"""Server-only raw KV locality pilot; no quantizer tuning or coding claims."""
import argparse
import json
from pathlib import Path
import statistics
import time

from tools.full_length_qa import ROOT, split_prompt
from tools.full_length_adaptive import save, append, digest


def measures(values):
    """Input [nonanchor tokens, channels], no stride subsampling."""
    import torch
    values=values.double()
    var=values.var(dim=0,unbiased=False)
    absolute=values.float().abs().flatten()
    quantiles=torch.quantile(absolute,torch.tensor([.5,.9,.99],device=values.device,dtype=torch.float32))
    return {'mean_channel_variance':float(var.mean()),'median_channel_variance':float(var.median()),
            'global_variance':float(values.var(unbiased=False)),
            'mean_absolute':float(absolute.mean()),'rms':float(values.square().mean().sqrt()),
            'abs_p50_p90_p99':quantiles.tolist()},var


def layer_stats(values,group_size):
    """Compare the same nonanchor positions in raw and anchor-delta form."""
    import torch
    positions=torch.arange(values.shape[0],device=values.device)
    anchors=values[positions%group_size==0].float()
    idx=positions[positions%group_size!=0]
    parent=idx//group_size
    raw=values[idx].float()
    scales=anchors.abs().amax(-1,keepdim=True)/127.
    scales=torch.where(scales>0,scales,torch.ones_like(scales))
    restored=torch.round(anchors/scales).clamp(-127,127)*scales
    result={}
    reference_var=None
    for name,reference in [('raw',None),('delta_original',anchors),('delta_reconstructed',restored)]:
        sample=raw if reference is None else raw-reference[parent]
        metrics,var=measures(sample)
        if reference_var is None:
            reference_var=var
        mask=reference_var>1e-12
        ratios=var[mask]/reference_var[mask]
        result[name]={'tokens':len(idx),'channels':values.shape[1],**metrics,
            'ratio_of_mean_channel_variances':float(var.mean()/reference_var.mean()) if float(reference_var.mean())>0 else None,
            'median_channel_variance_ratio':float(ratios.median()) if ratios.numel() else None,
            'fraction_channels_variance_lower':float((ratios<1).double().mean()) if ratios.numel() else None,
            'valid_ratio_channels':int(mask.sum()),'per_channel_variance':var.tolist()}
    return result


def main():
    import torch
    from experiments.engine import Engine,snapshot_path
    from paper_compression.provenance import provenance
    from tools.prepare_delta_diagnostics import sha
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--readiness',type=Path,required=True)
    parser.add_argument('--all',action='store_true',help='Run all200 prompts, keeping the same pilot statistics')
    args=parser.parse_args()
    config_path=ROOT/'paper_compression/configs/delta-diagnostics.json'
    cfg=json.loads(config_path.read_text());core=json.loads((ROOT/cfg['core_config']).read_text())
    readiness=json.loads(args.readiness.read_text())
    assert cfg['inference_authorized_by_this_config'] and torch.cuda.is_available()
    assert core['model']==readiness['model'] and core['model']['weight_loading']=='fp16'
    dataset=ROOT/cfg['dataset']
    assert digest(dataset)==readiness['dataset_sha256']
    snapshot=snapshot_path(core['model'])
    for filename,checksum in readiness['weights_sha256'].items():
        assert sha(snapshot/filename)==checksum,'Model weights changed'
    rows={r['test_id']:r for r in map(json.loads,dataset.read_text().splitlines())}
    selected=sorted(rows) if args.all else cfg['pilot_source_ids']
    if args.all and len(selected)!=cfg['expected_samples']:
        raise ValueError('Expanded dataset is incomplete')
    torch.set_num_threads(core['cpu_threads']);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    def fingerprint():
        hashes=provenance('stage2')
        for p in [Path(__file__),config_path,ROOT/'tools/prepare_delta_diagnostics.py',ROOT/'tools/full_length_qa.py',dataset]:
            hashes[str(p.relative_to(ROOT))]=digest(p)
        return hashes
    hashes=fingerprint()
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'run_meta.json',{'config':cfg,'model':core['model'],'execution':core['execution'],
        'weights_sha256':readiness['weights_sha256'],'source_sha256':hashes,
        'readiness_sha256':digest(args.readiness),'gpu':torch.cuda.get_device_name(),
        'source_ids':selected,'effective_stage':'all_documents' if args.all else 'pilot',
        'scope':'raw statistics only; not final byte counts or generation quality'})
    started=time.monotonic();completed=0
    def status(state,**extra):
        save(args.output/'status.json',{'status':state,'completed_documents':completed,
            'expected_documents':len(selected),'wall_seconds':time.monotonic()-started,**extra})
    status('running')
    aggregate=[]
    try:
        engine=Engine(core['model'],execution=core['execution'])
        for source in selected:
            context,suffix=split_prompt(engine.tokenizer,rows[source]['prompt'])['clean_question']
            assert len(context)+len(suffix)<core['model']['context_limit']
            host=engine.prefill(context).cpu()
            torch.cuda.empty_cache()
            for kind_index,kind in enumerate(['K','V']):
                for layer in range(host.shape[0]):
                    values=host[layer,kind_index].permute(1,0,2).reshape(host.shape[3],-1).to('cuda')
                    stats=layer_stats(values,cfg['group_size'])
                    for mode,metrics in stats.items():
                        append(args.output/'per_layer.jsonl',{'source_id':source,'kind':kind,
                            'layer':layer,'mode':mode,**metrics})
                    del values
                print('MEASURED',source,kind,len(context),flush=True)
            del host
            torch.cuda.empty_cache()
            completed+=1;status('running',source_id=source)
        # Detailed channel arrays remain on disk; summaries do not need them in RAM.
        full=[]
        with (args.output/'per_layer.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line)
                del row['per_channel_variance']
                full.append(row)
        assert len(full)==len(selected)*2*core['model']['layers']*3
        keys={(r['source_id'],r['kind'],r['layer'],r['mode']) for r in full}
        expected={(source,kind,layer,mode) for source in selected for kind in ['K','V']
                  for layer in range(core['model']['layers'])
                  for mode in ['raw','delta_original','delta_reconstructed']}
        if len(keys)!=len(full) or keys!=expected:
            raise ValueError('Missing or duplicate layer statistics')
        for source in selected:
            for kind in ['K','V']:
                for mode in ['delta_original','delta_reconstructed']:
                    ratios=[r['ratio_of_mean_channel_variances'] for r in full
                            if r['source_id']==source and r['kind']==kind and r['mode']==mode]
                    aggregate.append({'source_id':source,'kind':kind,'mode':mode,
                        'mean_layer_variance_ratio':statistics.mean(ratios),
                        'median_layer_variance_ratio':statistics.median(ratios),
                        'layers_with_lower_variance':sum(x<1 for x in ratios),'layers':len(ratios)})
        if fingerprint()!=hashes:raise ValueError('Sources changed during pilot')
        save(args.output/'summary.json',{'per_document_kind':aggregate,
            'interpretation':'ratio<1 means lower per-channel token variance; not evidence of smaller encoded packets'})
        status('completed')
    except BaseException as e:
        status('failed',error_type=type(e).__name__,error=str(e));raise


if __name__=='__main__':
    main()
