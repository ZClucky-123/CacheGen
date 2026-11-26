"""Server-only paired full-context real-socket chunk/pipeline study.

No experiment runs on import. Warm quality checks are outside timed repeats.
Serial and pipeline replay the exact same independently encoded chunk messages.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics
import subprocess
import time

from tools.full_length_qa import ROOT, load_rows, split_prompt, score
from tools.full_length_adaptive import save, append, digest, generate_native, predict_nll


def matrix(cfg, ids):
    variants = [(m, mode) for m in cfg['methods'] for mode in cfg['modes']] + [('text', 'whole')]
    return [(source, method, mode, bandwidth, repeat)
            for source in ids for method, mode in variants
            for bandwidth in cfg['bandwidth_gbps'] for repeat in range(cfg['repeats'])]


def cache_hash(blob):
    host = blob.detach().contiguous().cpu()
    return hashlib.sha256(memoryview(host.numpy())).hexdigest()


def replay(engine, cfg, packets, mode, bandwidth, suffix):
    import torch
    from experiments.chunks import restore_chunk
    from experiments.engine import timed
    from mini_repro.transport import transfer_chunks, overlap_timeline
    merged = None
    def restore(index, wire):
        nonlocal merged
        merged = restore_chunk(engine, packets[index], wire, merged)
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    start = time.perf_counter()
    network = transfer_chunks([c.wire for c in packets], bandwidth, restore,
        mode=mode, queue_capacity=cfg['queue_capacity'],
        chunk_bytes=cfg['pacing_quantum_bytes'], origin=start, analyze_timeline=False)
    initial, first_seconds = timed(lambda: engine.first_state([], suffix, merged))
    wall = time.perf_counter() - start
    # Postprocessing and byte comparisons MUST remain outside TTFT.
    if network['received'] != [c.wire for c in packets]:
        raise ValueError('Transport byte identity failed')
    if merged.shape[3] != packets[-1].end:
        raise ValueError('Missing full-context tail')
    overlap = overlap_timeline(network['chunks'])
    if mode != 'chunk_pipeline' and overlap['overlap_seconds'] > 1e-6:
        raise ValueError('Serial transfer unexpectedly overlaps restoration')
    network_s = sum(r['effective_transfer_seconds'] for r in network['chunks'])
    restore_s = sum(r['restore_seconds'] for r in network['chunks'])
    return merged, initial, {'ttft_seconds': wall, 'first_token_seconds': first_seconds,
        'first_token_id': initial[0], 'network_active_seconds': network_s,
        'restore_sum_seconds': restore_s, **overlap,
        'other_wall_seconds': wall - (network_s + restore_s - overlap['overlap_seconds'] + first_seconds),
        'transport_wall_seconds': network['transport_wall_seconds'],
        'chunks': network['chunks'], 'bytes': sum(len(c.wire) for c in packets),
        'byte_identity_checked': True, 'application_chunks': len(packets)}


def summarize(output):
    rows = [json.loads(s) for s in (output / 'records.jsonl').read_text().splitlines()]
    # Repeats reduce timing noise; the document remains the independent unit.
    docs = {}
    for r in rows:
        docs.setdefault((r['source_id'], r['method'], r['mode'], r['bandwidth_gbps']), []).append(r)
    per_doc = []
    for key, group in sorted(docs.items()):
        per_doc.append(dict(zip(('source_id', 'method', 'mode', 'bandwidth_gbps'), key)) | {
            'median_ttft_seconds': statistics.median(r['ttft_seconds'] for r in group),
            'median_restore_seconds': statistics.median(r['restore_sum_seconds'] for r in group),
            'median_overlap_seconds': statistics.median(r['overlap_seconds'] for r in group),
            'repeat_max_min_ratio': max(r['ttft_seconds'] for r in group) / min(r['ttft_seconds'] for r in group)})
    groups = {}
    for r in per_doc:
        groups.setdefault((r['method'], r['mode'], r['bandwidth_gbps']), []).append(r)
    conditions = [dict(zip(('method','mode','bandwidth_gbps'), key)) | {
        'documents': len(group), 'median_ttft_seconds': statistics.median(r['median_ttft_seconds'] for r in group),
        'median_overlap_seconds': statistics.median(r['median_overlap_seconds'] for r in group)}
        for key, group in sorted(groups.items())]
    lookup = {(r['source_id'],r['method'],r['mode'],r['bandwidth_gbps']):r for r in per_doc}
    paired = []
    for (method, mode, bw), group in sorted(groups.items()):
        if mode != 'chunk_pipeline':
            continue
        for baseline in ('whole', 'chunk_serial'):
            ratios = [lookup[(r['source_id'],method,baseline,bw)]['median_ttft_seconds'] / r['median_ttft_seconds'] for r in group]
            paired.append({'method':method,'bandwidth_gbps':bw,'baseline':baseline,
                'median_paired_speedup':statistics.median(ratios),'per_document_speedup':ratios})
    save(output/'summary.json', {'conditions':conditions,'paired_speedups':paired,
        'per_document':per_doc,'scope':'actual paced local socket; whole includes encoding granularity differences; serial/pipeline use identical bytes'})


def run(args):
    import torch
    from experiments.engine import Engine
    from experiments.chunks import prepare, preparation_metadata
    from paper_compression.provenance import provenance
    cfg = json.loads(args.config.read_text())
    ids = args.ids if args.ids is not None else cfg['source_ids']
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(cfg['source_ids']):
        raise ValueError('Distinct preregistered source IDs required')
    if not torch.cuda.is_available():
        raise RuntimeError('Run only on the CUDA server')
    core = json.loads((ROOT/'paper_compression/configs/core-study.json').read_text())
    torch.set_num_threads(core['cpu_threads'])
    torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    def fingerprints():
        result = provenance('stage2')
        for p in (Path(__file__), args.config, ROOT/'tools/full_length_adaptive.py', ROOT/'tools/full_length_qa.py',ROOT/'test_data/longchat.jsonl'):
            result[str(p.relative_to(ROOT))] = digest(p)
        return result
    sources = fingerprints()
    expected = set(matrix(cfg, ids))
    save(args.output/'run_meta.json', {'config':cfg,'source_ids':ids,'expected_records':len(expected),
        'source_sha256':sources,'model':core['model'],'execution':core['execution'],
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'started_utc':datetime.now(timezone.utc).isoformat(),
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'scope':'real paced socket, not WAN; fixed profiles, no probe or adaptation; preencoding excluded; 1 warm-up per representation/mode, 3 timed repetitions',
        'quality_scope':'full greedy answer and raw-answer conditional NLL on unmeasured warm-ups; timed repeats verify exact received bytes and matching first token'})
    started = time.monotonic()
    count = 0
    def status(state, **extra):
        save(args.output/'status.json', {'status':state,'completed_records':count,
            'expected_records':len(expected),'wall_seconds':time.monotonic()-started,**extra})
    status('running')
    try:
        engine = Engine(core['model'], execution=core['execution'])
        rows = load_rows(ids)
        for source in ids:
            context,suffix = split_prompt(engine.tokenizer,rows[source]['prompt'])['clean_question']
            if len(context)+len(suffix)+cfg['generation_tokens'] > core['model']['context_limit']:
                raise ValueError('Full context exceeds window')
            print('SOURCE',source,'TOKENS',len(context),flush=True)
            raw = engine.prefill(context)
            baseline = generate_native(engine,suffix,raw,cfg['generation_tokens'])
            raw_nll = predict_nll(engine,suffix,raw,baseline['generated_ids'])
            host = raw.cpu()
            del raw
            torch.cuda.empty_cache()
            append(args.output/'raw.jsonl',{'source_id':source,'kv_tokens':len(context),**baseline,'answer_nll':raw_nll})
            variants = {}
            for method in cfg['methods']:
                blob = host.to('cuda')
                whole = prepare(engine,context,blob,method,len(context))
                # Whole and chunk encodings are different wire representations.
                chunked = prepare(engine,context,blob,method,cfg['chunk_tokens'])
                del blob
                variants[(method,'whole')] = whole
                variants[(method,'chunk_serial')] = chunked
                variants[(method,'chunk_pipeline')] = chunked
                torch.cuda.empty_cache()
            dummy = torch.empty(host.shape,dtype=host.dtype,device='meta')
            variants[('text','whole')] = prepare(engine,context,dummy,'text',len(context))
            del host, dummy
            save(args.output/f'preparation-{source}.json', {m+'/'+mode:preparation_metadata(p) for (m,mode),p in variants.items()})
            warm = {}
            for (method,mode),packets in variants.items():
                restored,initial,timing = replay(engine,cfg,packets,mode,None,suffix)
                sha = cache_hash(restored)
                answer = generate_native(engine,suffix,restored,cfg['generation_tokens'],initial=initial)
                del initial
                nll = predict_nll(engine,suffix,restored,baseline['generated_ids'])
                warm[(method,mode)] = {'cache_sha256':sha,'first_token_id':timing['first_token_id']}
                append(args.output/'quality.jsonl',{'source_id':source,'method':method,'mode':mode,
                    'cache_sha256':sha,**answer,**score(answer['prediction'],rows[source]['label']),
                    'identical_to_raw':answer['prediction']==baseline['prediction'],
                    'answer_nll':nll,'delta_answer_nll':nll-raw_nll if nll is not None else None})
                del restored
                torch.cuda.empty_cache()
                print('WARMED',source,method,mode,flush=True)
            for method in cfg['methods']:
                if warm[(method,'chunk_serial')] != warm[(method,'chunk_pipeline')]:
                    raise ValueError('Serial/pipeline did not reconstruct identical KV/first token')
            schedule = matrix(cfg,[source])
            random.Random(cfg['seed']+source).shuffle(schedule)
            save(args.output/f'schedule-{source}.json',schedule)
            for _,method,mode,bw,repeat in schedule:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                restored,initial,timing = replay(engine,cfg,variants[(method,mode)],mode,bw,suffix)
                if timing['first_token_id'] != warm[(method,mode)]['first_token_id']:
                    raise ValueError('Timed first token differs from verified warm-up')
                append(args.output/'records.jsonl',{'source_id':source,'kv_tokens':len(context),
                    'method':method,'mode':mode,'bandwidth_gbps':bw,'repetition':repeat,
                    'verified_cache_sha256':warm[(method,mode)]['cache_sha256'],**timing,
                    'peak_gpu_bytes':torch.cuda.max_memory_allocated()})
                del initial, restored
                count += 1
                status('running',source_id=source)
                print('RESULT',count,'/',len(expected),source,method,mode,bw,repeat,
                      f'{timing["ttft_seconds"]:.3f}s','OVERLAP',f'{timing["overlap_seconds"]:.3f}s',flush=True)
            del variants, whole, chunked, packets
            torch.cuda.empty_cache()
        result=[json.loads(s) for s in (args.output/'records.jsonl').read_text().splitlines()]
        keys=[(r['source_id'],r['method'],r['mode'],r['bandwidth_gbps'],r['repetition']) for r in result]
        if len(keys)!=len(set(keys)) or set(keys)!=expected or fingerprints()!=sources:
            raise ValueError('Incomplete/duplicate matrix or sources changed')
        summarize(args.output)
        status('completed')
    except BaseException as error:
        status('failed',error_type=type(error).__name__,error=str(error))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'paper_compression/configs/full-length-pipeline.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ids',type=int,nargs='+')
    args=parser.parse_args()
    args.config=args.config.resolve()
    run(args)


if __name__=='__main__':
    main()
