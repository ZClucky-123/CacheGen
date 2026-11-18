"""Targeted, offline Mistral KV loss diagnostics; dry-run needs only Python.

This is a separate diagnostic, never a continuation of the archived experiment.
"""
import argparse
import hashlib
import json
import os
import csv
import importlib
import importlib.metadata
from pathlib import Path
import statistics
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_BATCH=ROOT/'results/mistral/larger/v3-larger-stable-20261025'
DEFAULT_SOURCES=(3,5,16,7)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def schedules(spec):
    layers=spec['layers'];b1,b2,_=spec['key_boundaries'];v=spec['value_boundary']
    k1=[16 if i<b2 else 12 for i in range(layers)]
    v1=[16 if i<v else 12 for i in range(layers)]
    return {
        'cachegen1':{'key_bins':k1,'value_bins':v1},
        'cachegen2':{'key_bins':[32 if i<b1 else 16 for i in range(layers)],
                     'value_bins':[32 if i<v else 16 for i in range(layers)]},
        'cachegen3':{'key_bins':[32]*layers,'value_bins':[32]*layers},
        'cachegen1_k32':{'key_bins':[32]*layers,'value_bins':v1.copy()},
        'cachegen1_v32':{'key_bins':k1.copy(),'value_bins':[32]*layers},
    }


def make_plan(batch, sources=DEFAULT_SOURCES):
    batch=Path(batch).resolve()
    manifest=json.loads((batch/'artifacts.json').read_text())
    if manifest.get('status')!='completed':raise ValueError('Use a completed archived batch')
    fingerprints={}
    for name in ('config.json','selection.json','metadata.json','records.jsonl'):
        info=manifest['files'][name];path=batch/name
        digest=sha256(path)
        if path.stat().st_size!=info['bytes'] or digest!=info['sha256']:
            raise ValueError('Archive integrity failed: '+name)
        fingerprints[name]=digest
    cfg=json.loads((batch/'config.json').read_text())
    if cfg['model']['name']!='mistral' or cfg['model']['layers']!=32:
        raise ValueError('This diagnostic is for the archived 32-layer Mistral model')
    if not sources or len(set(sources))!=len(sources):raise ValueError('Choose unique source IDs')
    selected=json.loads((batch/'selection.json').read_text())['quality']
    cases=[]
    for source in sources:
        found=[r for r in selected if r['source_id']==source and r['context_tokens']==4096]
        if len(found)!=1:raise ValueError(f'Missing or duplicate 4K source: {source}')
        case=dict(found[0])
        if len(case['context_ids'])!=4096 or len(case['continuation_ids'])!=32:
            raise ValueError('Keep the original 4096 context and 32 target tokens')
        cases.append(case)
    originals={}
    with (batch/'records.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r['kind']=='quality' and r['context_tokens']==4096 and r['source_id'] in sources:
                key=(r['source_id'],r['method'])
                if key in originals:raise ValueError('Duplicate archived source/method')
                originals[key]=r
    for case in cases:
        previous={}
        for method in ('raw','cachegen1','cachegen2','cachegen3'):
            r=originals[(case['source_id'],method)]
            if r['scored_ids']!=case['continuation_ids']:raise ValueError('Archived target tokens differ')
            if abs(statistics.mean(r['token_nll'])-r['nll'])>1e-6:
                raise ValueError('Archived token/document NLL mismatch')
            previous[method]={'nll':r['nll'],'token_nll':r['token_nll']}
        case['archived_scores']=previous
    variants=schedules(cfg['model'])
    return {'protocol':'mistral-4k-loss-diagnostic-v1','batch':str(batch),
            'input_sha256':fingerprints,'config':cfg,'cases':cases,'variants':variants,
            'score_rows':len(cases)*(1+2*len(variants))}


def reserve_output(output,batch):
    output=Path(output).resolve();batch=Path(batch).resolve()
    if output.is_relative_to(batch):raise ValueError('Do not write inside the original batch')
    output.mkdir(parents=True,exist_ok=False)
    return output


def compare_tensors(expected,actual):
    import torch
    if expected.shape!=actual.shape:raise ValueError('Diagnostic tensor shape mismatch')
    finite=bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
    different=int(torch.count_nonzero(expected!=actual))
    return {'identical':finite and different==0,'finite':finite,'different_elements':different,
            'elements':expected.numel()}


def make_codecs(engine,variant):
    from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
    from lmcache.storage_backend.serde.cachegen_basics import CacheGenConfig
    from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
    from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
    spec=engine.spec; b1,b2,_=spec['key_boundaries'];v=spec['value_boundary']
    k,vb=variant['key_bins'],variant['value_bins']
    schedule=CacheGenConfig(*spec['key_boundaries'],k[0],k[b1],k[b2],v,vb[0],vb[v])
    config=LMCacheEngineConfig.from_defaults(chunk_size=4096)
    metadata=LMCacheEngineMetadata(spec['id'],1,0,'huggingface')
    return (CacheGenSerializer(config,metadata,cachegen_config=schedule),
            CacheGenDeserializer(config,metadata,cachegen_config=schedule))


def quantized_reference(blob,encoder):
    """Use the original quantizer/dequantizer, with no entropy encoding."""
    import torch
    from lmcache.storage_backend.serde.cachegen_encoder import _split_kv, torch_quant_vectorized
    from lmcache.storage_backend.serde.cachegen_decoder import do_dequantize
    k,v=_split_kv(blob.permute(0,1,3,2,4))
    qk,sk=torch_quant_vectorized(encoder.key_bins,k)
    qv,sv=torch_quant_vectorized(encoder.value_bins,v)
    rk=do_dequantize(qk.float(),encoder.key_bins,sk).half()
    rv=do_dequantize(qv.float(),encoder.value_bins,sv).half()
    layers,_,heads,tokens,dim=blob.shape
    restored=torch.stack((rk,rv),dim=1).reshape(layers,2,tokens,heads,dim).permute(0,1,3,2,4)
    symbols=torch.cat((qk,qv),dim=0).view(torch.uint8)
    return restored,symbols,(sk,sv)


def encode_probe_payload(blob,encoder,symbols,scales):
    """Keep original baseline encoding; align CDF alphabets for K/V interventions.

    The original serializer concatenates K/V CDFs and requires equal alphabet
    sizes. A K-only or V-only precision change breaks that assumption. For these
    diagnostic cases only, use one common CDF range for the already verified
    symbols, retaining their actual quantization bins and reconstruction scales.
    These encode times have different scopes and are not performance baselines.
    """
    import torch
    import torchac_cuda
    from lmcache.storage_backend.serde import cachegen_basics as basics
    from lmcache.storage_backend.serde.cachegen_encoder import encode_ntokens
    key_max=int(encoder.key_bins.max());value_max=int(encoder.value_bins.max())
    if key_max==value_max:
        return encoder.to_bytes(blob),{'encoder_path':'original_serializer','cdf_bins':key_max,
                                      'encode_timing_scope':'quantization_and_encoding'}
    common_max=max(key_max,value_max)
    signed=symbols.view(torch.int8)
    cdf=torchac_cuda.calculate_cdf(signed,common_max)
    layers,tokens,channels=symbols.shape
    buffer=torch.zeros((layers,channels,basics.CACHEGEN_GPU_OUTPUT_BYTES_PER_CHANNEL),
                       dtype=torch.uint8,device=symbols.device)
    lengths=torch.zeros((layers,channels),dtype=torch.int32,device=symbols.device)
    parts=[]
    for start in range(0,tokens,basics.CACHEGEN_SAFE_ENCODE_TOKENS):
        end=min(start+basics.CACHEGEN_SAFE_ENCODE_TOKENS,tokens)
        data=encode_ntokens(cdf,signed[:,start:end,:],buffer,lengths)
        parts.append(basics.CacheGenGPUBytestream(data,lengths.clone(),end-start))
    obj=basics.CacheGenGPUEncoderOutput(parts,cdf,scales[0],scales[1],blob.shape[2],blob.shape[4])
    return obj.to_bytes(),{'encoder_path':'shared_alphabet_diagnostic','cdf_bins':common_max,
                          'encode_timing_scope':'entropy_encoding_from_verified_symbols'}


def audit_encoded(payload,expected_symbols,scales):
    """Decode the actual serialized arithmetic stream before dequantization."""
    import torch
    from lmcache.storage_backend.serde.cachegen_basics import CacheGenGPUEncoderOutput
    from lmcache.storage_backend.serde.cachegen_decoder import decode_chunk
    # This payload was just produced in this process, not supplied externally.
    obj=CacheGenGPUEncoderOutput.from_bytes(payload)
    output=torch.empty_like(expected_symbols)
    start=0
    for part in obj.data_chunks:
        end=start+part.ntokens
        if end>output.shape[1]:raise ValueError('Encoded segments exceed target length')
        decode_chunk(obj.cdf,part,output[:,start:end,:])
        start=end
    if start!=output.shape[1]:raise ValueError('Encoded segments do not cover all tokens')
    return {'symbols':compare_tensors(expected_symbols,output),
            'key_scales':compare_tensors(scales[0],obj.max_tensors_key),
            'value_scales':compare_tensors(scales[1],obj.max_tensors_value),
            'internal_segment_tokens':[p.ntokens for p in obj.data_chunks]}


def head_errors(raw,restored,source,variant):
    import torch
    rows=[]
    for layer in range(raw.shape[0]):
        difference=restored[layer].float()-raw[layer].float()
        mae=difference.abs().mean(dim=(-2,-1)).cpu().tolist()
        rmse=difference.square().mean(dim=(-2,-1)).sqrt().cpu().tolist()
        maximum=difference.abs().amax(dim=(-2,-1)).cpu().tolist()
        for kind in range(2):
            for head in range(raw.shape[2]):
                rows.append({'source_id':source,'variant':variant,'kv':'K' if kind==0 else 'V',
                             'layer_index_0_based':layer,'head_index_0_based':head,
                             'mae':mae[kind][head],'rmse':rmse[kind][head],'max_abs':maximum[kind][head]})
        del difference
    return rows


def save(path,value):
    # Replace small metadata atomically; measurements are separately fsynced.
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def append_row(path,row):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())


def append_rows(path,rows):
    with Path(path).open('a',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        f.flush();os.fsync(f.fileno())


def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def verify_runtime(plan):
    meta=json.loads((Path(plan['batch'])/'metadata.json').read_text())
    for relative,digest in meta['identity']['inputs']['code'].items():
        if sha256(ROOT/relative)!=digest:raise ValueError('Archived execution source changed: '+relative)
    for package in ('torch','transformers','numpy','fschat','accelerate','lmcache','torchac','torchac_cuda'):
        if importlib.metadata.version(package)!=meta['runtime']['packages'][package]:
            raise ValueError('Use archived runtime package version: '+package)
    for module_name,digest in meta['dependency_sources_sha256'].items():
        module=importlib.import_module(module_name)
        if sha256(module.__file__)!=digest:raise ValueError('Dependency source changed: '+module_name)
    return meta


def full_nll(engine,case):
    import torch
    from torch.nn import functional as F
    context,continuation=case['context_ids'],case['continuation_ids']
    with torch.inference_mode():
        output=engine.model(input_ids=torch.tensor([context+continuation[:-1]],device='cuda'),use_cache=False)
        losses=F.cross_entropy(output.logits[0,len(context)-1:].float(),
                               torch.tensor(continuation,device='cuda'),reduction='none')
        return {'nll':float(losses.mean()),'token_nll':losses.tolist()}


def run_probe(plan,output):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for this probe; use --dry-run locally')
    if output:
        destination=Path(output).resolve()
        if destination.is_relative_to(Path(plan['batch']).resolve()):
            raise ValueError('Do not write inside the original batch')
        if destination.exists():raise FileExistsError(destination)
    meta=verify_runtime(plan)
    cfg=plan['config'];torch.set_num_threads(cfg['cpu_threads']);torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=meta['runtime']['tf32_matmul']
    from experiments.engine import Engine, timed
    engine=Engine(cfg['model'],load=False,execution=cfg['execution'])
    for name,digest in meta['model_files_sha256'].items():
        print('Checking pinned model file:',name,flush=True)
        if sha256(engine.snapshot/name)!=digest:raise ValueError('Model file differs: '+name)
    import torchac_cuda
    if sha256(torchac_cuda.__file__)!=meta['runtime']['codec_binary_sha256']:
        raise ValueError('Use the archived CUDA extension binary')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=reserve_output(output or ROOT/'.local/diagnostics'/('4k-loss-'+stamp),plan['batch'])
    save(output/'plan.json',plan)
    runtime={'gpu':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),
             'torch':torch.__version__,'cuda':torch.version.cuda,'diagnostic_script_sha256':sha256(__file__),
             'original_metadata_sha256':plan['input_sha256']['metadata.json']}
    save(output/'runtime.json',runtime)
    state={'status':'running','completed_score_rows':0,'expected_score_rows':plan['score_rows']}
    save(output/'status.json',state)
    scores=[];all_errors=[]
    try:
        print('Loading fixed Mistral weights; output:',output,flush=True)
        engine.load()
        for case in plan['cases']:
            source=case['source_id'];context=case['context_ids'];targets=case['continuation_ids']
            print('Prefill and raw NLL validation, source',source,flush=True)
            full=full_nll(engine,case)
            blob,prefill_s=timed(lambda:engine.prefill(context))
            unchanged=blob.clone()
            raw=engine.nll(blob,context,targets)
            full_error=abs(full['nll']-raw['nll'])
            raw_error=abs(raw['nll']-case['archived_scores']['raw']['nll'])
            validation={'source_id':source,'full_forward':full,'raw_reuse':raw,
                        'full_vs_raw_abs_nll':full_error,'new_vs_archived_raw_abs_nll':raw_error,
                        'reference_tolerance':.01,'offline_prefill_seconds':prefill_s}
            append_row(output/'raw_checks.jsonl',validation)
            if max(full_error,raw_error)>.01:raise ValueError('Raw baseline NLL differs beyond archived .01 tolerance')
            base={'source_id':source,'variant':'raw','path':'raw','nll':raw['nll'],'delta_nll':0.,
                  'token_nll':raw['token_nll'],'scored_ids':targets,'archived_nll':case['archived_scores']['raw']['nll']}
            scores.append(base);append_row(output/'scores.jsonl',base)
            for name,variant in plan['variants'].items():
                print('Source',source,'variant',name,flush=True)
                encoder,decoder=make_codecs(engine,variant)
                direct,symbols,scales=quantized_reference(blob,encoder)
                direct_score=engine.nll(direct,context,targets)
                encoded,encode_s=timed(lambda:encode_probe_payload(blob,encoder,symbols,scales))
                payload,encoding_metadata=encoded
                checks=audit_encoded(payload,symbols,scales)
                decoded,decode_s=timed(lambda:decoder.from_bytes(payload))
                checks['dequantized_kv']=compare_tensors(direct,decoded)
                checks['raw_cache_unchanged']=compare_tensors(unchanged,blob)
                checks.update(source_id=source,variant=name,bytes=len(payload),encode_seconds=encode_s,
                              decode_seconds=decode_s,**encoding_metadata)
                append_row(output/'codec_checks.jsonl',checks)
                if not all(checks[k]['identical'] for k in ('symbols','key_scales','value_scales','dequantized_kv','raw_cache_unchanged')):
                    raise ValueError('Codec or cache identity failed; inspect codec_checks.jsonl')
                decoded_score=engine.nll(decoded,context,targets)
                if direct_score['token_nll']!=decoded_score['token_nll']:
                    raise ValueError('Identical restored KV produced different token NLL; inference is not repeatable')
                for path,score in (('quantize_only',direct_score),('arithmetic_roundtrip',decoded_score)):
                    row={'source_id':source,'variant':name,'path':path,'nll':score['nll'],
                         'delta_nll':score['nll']-raw['nll'],'token_nll':score['token_nll'],
                         'scored_ids':targets,'archived_nll':case['archived_scores'].get(name,{}).get('nll')}
                    scores.append(row);append_row(output/'scores.jsonl',row)
                errors=head_errors(blob,decoded,source,name)
                append_rows(output/'head_errors.jsonl',errors)
                all_errors.extend(errors)
                state.update(completed_score_rows=len(scores),current_source=source,current_variant=name)
                save(output/'status.json',state)
                del encoder,decoder,direct,symbols,scales,payload,decoded,encoded
            if not torch.equal(unchanged,blob):raise ValueError('Original KV mutated during diagnostic')
            del unchanged,blob
            torch.cuda.empty_cache()
        for name,digest in plan['input_sha256'].items():
            if sha256(Path(plan['batch'])/name)!=digest:raise ValueError('Original evidence changed during diagnostic')
        assert len(scores)==plan['score_rows']
        write_csv(output/'summary.csv',[{k:r[k] for k in ('source_id','variant','path','nll','delta_nll','archived_nll')} for r in scores])
        write_csv(output/'head_errors.csv',all_errors)
        state.update(status='completed',completed_score_rows=len(scores),raw_reference_checks=len(plan['cases']),
                     note='Targeted diagnostic sources; do not pool with the formal experiment or infer population quality.')
        save(output/'status.json',state)
        print('COMPLETED:',output,flush=True)
    except BaseException as exc:
        state.update(status='failed',completed_score_rows=len(scores),error_type=type(exc).__name__,error=str(exc))
        save(output/'status.json',state)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,default=DEFAULT_BATCH)
    parser.add_argument('--sources',type=int,nargs='+',default=list(DEFAULT_SOURCES))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    plan=make_plan(args.batch,args.sources)
    if args.dry_run:
        print(json.dumps({'protocol':plan['protocol'],'source_ids':[c['source_id'] for c in plan['cases']],
                          'context_tokens':4096,'target_tokens':32,'variants':plan['variants'],
                          'score_rows':plan['score_rows'],'gpu_started':False,
                          'output_created':False},ensure_ascii=False,indent=2))
        return
    run_probe(plan,args.output)


if __name__=='__main__':main()
