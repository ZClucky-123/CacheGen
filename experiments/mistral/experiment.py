"""Pinned, offline Mistral CacheGen experiment using the existing author codec."""

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import re
import statistics
import string
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.workspace import Workspace

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from mini_repro.experiment_common import load_longchat_segments, pack_uniform, unpack_uniform
from mini_repro.run import blob_to_legacy, legacy_to_blob
from mini_repro.ttft import measurement_schedule, rate_limited_transfer

ROOT = Path(__file__).resolve().parent
REPO = Workspace.discover().repo
MODEL_ID = 'mistral-community/Mistral-7B-v0.2'
REVISION = '80cf49b18de6354fd8a6d1e81c177dd830e05eea'
HF_CACHE = Path(os.environ.get('HF_HUB_CACHE', str(Path(os.environ.get('HF_HOME', REPO / '.venv/huggingface')) / 'hub')))
SNAPSHOT = HF_CACHE / 'models--mistral-community--Mistral-7B-v0.2/snapshots' / REVISION
METHODS = ('raw', 'uniform4', 'uniform8', 'cachegen1', 'cachegen2', 'cachegen3')


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(piece)
    return h.hexdigest()


def log(event, **values):
    print(json.dumps({'time': now(), 'event': event, **values}, ensure_ascii=False), flush=True)


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def validate_blob(restored, original):
    if restored.shape != original.shape or restored.dtype != original.dtype:
        raise RuntimeError('Decoded KV shape/dtype mismatch')
    if not bool(torch.isfinite(restored).all()):
        raise RuntimeError('Decoded KV contains NaN/Inf')


def codec(tokens, level):
    from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
    from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
    from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
    prior = os.environ.get('QUANT_LEVEL')
    os.environ['QUANT_LEVEL'] = str(level)
    try:
        cfg = LMCacheEngineConfig.from_defaults(chunk_size=tokens)
        meta = LMCacheEngineMetadata(MODEL_ID, 1, 0, 'huggingface')
        return CacheGenSerializer(cfg, meta), CacheGenDeserializer(cfg, meta)
    finally:
        if prior is None:
            os.environ.pop('QUANT_LEVEL', None)
        else:
            os.environ['QUANT_LEVEL'] = prior


def raw_payload(blob):
    stream = io.BytesIO()
    torch.save(blob.cpu(), stream)
    return stream.getvalue()


def encoded(blob, method):
    if method == 'raw':
        payload, elapsed = timed(lambda: raw_payload(blob))
        return payload, elapsed, lambda value: torch.load(io.BytesIO(value), map_location='cuda', weights_only=True)
    if method.startswith('uniform'):
        payload, elapsed = timed(lambda: pack_uniform(blob, int(method[7:])))
        return payload, elapsed, unpack_uniform
    encoder, decoder = codec(blob.shape[-2], int(method[-1]))
    payload, elapsed = timed(lambda: encoder.to_bytes(blob))
    return payload, elapsed, decoder.from_bytes


@torch.inference_mode()
def prefill(model, ids):
    out = model(input_ids=torch.tensor([ids], device='cuda'), use_cache=True)
    blob = legacy_to_blob(out.past_key_values)
    del out
    return blob


def fresh_cache(blob):
    # Each forward gets a fresh container; cache updates must not contaminate repeats.
    return DynamicCache.from_legacy_cache(blob_to_legacy(blob))


@torch.inference_mode()
def continuation_nll(model, blob, ids):
    out = model(input_ids=torch.tensor([ids], device='cuda'), past_key_values=fresh_cache(blob), use_cache=False)
    targets = torch.tensor(ids[1:], device='cuda')
    result = float(F.cross_entropy(out.logits[0, :-1].float(), targets))
    if not math.isfinite(result):
        raise RuntimeError('Nonfinite continuation NLL')
    return result


@torch.inference_mode()
def greedy_from_cache(model, blob, suffix_ids, limit, eos_id):
    cache = fresh_cache(blob)
    inputs = torch.tensor([suffix_ids], device='cuda')
    generated = []
    for _ in range(limit):
        out = model(input_ids=inputs, past_key_values=cache, use_cache=True)
        token = int(out.logits[0, -1].argmax())
        if token == eos_id:
            break
        generated.append(token)
        cache = out.past_key_values
        inputs = torch.tensor([[token]], device='cuda')
    return generated


@torch.inference_mode()
def first_token(model, context_ids, suffix_ids, blob=None):
    ids = suffix_ids if blob is not None else context_ids + suffix_ids
    kwargs = {} if blob is None else {'past_key_values': fresh_cache(blob)}
    out = model(input_ids=torch.tensor([ids], device='cuda'), use_cache=True, **kwargs)
    return int(out.logits[0, -1].argmax())


def normalize(text):
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def score_qa(prediction, answers):
    def one(answer):
        p, a = normalize(prediction), normalize(answer)
        common = sum((Counter(p.split()) & Counter(a.split())).values())
        f1 = 2 * common / (len(p.split()) + len(a.split())) if common else float(p == a)
        return float(p == a), f1
    scores = [one(answer) for answer in answers]
    return {'em': max(x[0] for x in scores), 'f1': max(x[1] for x in scores)}


def qa_cases(tokenizer, cfg):
    cases = []
    # File iteration preserves Unicode line separators inside valid JSON strings.
    with (REPO / 'test_data/tqa.jsonl').open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            ids = tokenizer.encode(row['prompt'], add_special_tokens=False)
            if len(ids) <= cfg['max_prompt_tokens']:
                cases.append({'source_index': index, 'ids': ids, 'answers': row['answers'],
                              'prompt_sha256': hashlib.sha256(row['prompt'].encode()).hexdigest()})
                if len(cases) == cfg['samples']:
                    return cases
    raise RuntimeError('Insufficient complete TriviaQA examples')


def metadata(config, config_path=None):
    import torchac_cuda
    log('verify_model_start')
    expected = json.loads(Path(os.environ.get('CACHEGEN_MODEL_CHECKSUMS', REPO / '.venv/mistral-download-verified.json')).read_text())
    hashes = {}
    for row in expected['files']:
        path = SNAPSHOT / row['file']
        digest = sha256(path)
        if path.stat().st_size != row['bytes'] or digest != row['sha256']:
            raise RuntimeError(f'Model checksum mismatch: {path.name}')
        hashes[path.name] = digest
        log('model_shard_verified', file=path.name)
    for path in SNAPSHOT.iterdir():
        if path.suffix == '.json' or path.name == 'tokenizer.model':
            hashes[path.name] = sha256(path)
    source = list(ROOT.glob('*.py')) + [Path(config_path) if config_path else ROOT / 'config.json', ROOT / 'run.sh']
    source += list((REPO / 'experiments').glob('*.py')) + [REPO / 'scripts/experiment-env.sh']
    source += list((REPO / 'LMCache/lmcache/storage_backend/serde').glob('cachegen*.py'))
    source += [REPO / 'src/utils.py', REPO / 'test_data/longchat.jsonl', REPO / 'test_data/tqa.jsonl']
    return {
        'time_utc': now(), 'platform': platform.platform(), 'python': platform.python_version(),
        'gpu': {'name': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
                'memory_bytes': torch.cuda.get_device_properties(0).total_memory},
        'cuda_runtime': torch.version.cuda,
        'packages': {n: importlib.metadata.version(n) for n in ['torch','transformers','numpy','fschat','lmcache','torchac']},
        'git_head': subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'], text=True).strip(),
        'git_status': subprocess.check_output(['git','-C',str(REPO),'status','--short'], text=True).splitlines(),
        'model_id': MODEL_ID, 'model_revision': REVISION, 'model_snapshot': str(SNAPSHOT),
        'model_sha256': hashes, 'codec_binary': str(torchac_cuda.__file__),
        'codec_sha256': sha256(torchac_cuda.__file__),
        'source_and_data_sha256': {str(p): sha256(p) for p in source},
        'configuration_file': str(config_path or ROOT / 'config.json'),
        'source_and_data_relative_sha256': {str(p.relative_to(REPO)): sha256(p) for p in source if p.is_relative_to(REPO)},
        'cgroup_memory_limit': Path('/sys/fs/cgroup/memory.max').read_text().strip(),
        'cgroup_cpu_quota': Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        'configuration': config,
        'limitations': ['FastChat groupwise 8bit model weights shared by all methods; FP16 KV',
                        'author repository codec has no paper anchor/delta path',
                        'single-host paced socket, receive then decode; no transfer/decode overlap',
                        'continuation NLL scores tokens 2..32; first held-out token omitted',
                        '10 complete short TriviaQA inputs; not the full paper benchmark',
                        'pooled P95 is descriptive; timing repeats are not independent contexts']}


@torch.inference_mode()
def smoke(model, tokenizer, output):
    ids = tokenizer.encode('The museum is in London. Its collection includes paintings and sculptures. ' * 12,
                           add_special_tokens=False)[:192]
    suffix = tokenizer.encode('\nQuestion: Where is the museum?\nAnswer:', add_special_tokens=False)
    blob = prefill(model, ids)
    direct = model(input_ids=torch.tensor([ids + suffix], device='cuda'), use_cache=True).logits[0, -1].float()
    reused = model(input_ids=torch.tensor([suffix], device='cuda'), past_key_values=fresh_cache(blob), use_cache=True).logits[0, -1].float()
    max_error = float((direct - reused).abs().max())
    if int(direct.argmax()) != int(reused.argmax()) or not torch.allclose(direct, reused, atol=0.1, rtol=0.02):
        raise RuntimeError(f'Raw cache/full forward mismatch: {max_error}')
    expected_ids = model.generate(torch.tensor([ids + suffix], device='cuda'), max_new_tokens=4,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)[0, len(ids + suffix):].tolist()
    manual_ids = greedy_from_cache(model, blob, suffix, 4, tokenizer.eos_token_id)
    expected_ids = expected_ids[:expected_ids.index(tokenizer.eos_token_id)] if tokenizer.eos_token_id in expected_ids else expected_ids
    if manual_ids != expected_ids:
        raise RuntimeError('Cache-based greedy generation differs from full generate')
    rows = []
    for method in METHODS:
        payload, encode_s, restore = encoded(blob, method)
        restored, decode_s = timed(lambda: restore(payload))
        validate_blob(restored, blob)
        rows.append({'method': method, 'bytes': len(payload), 'encode_seconds': encode_s,
                     'decode_seconds': decode_s, 'kv_mae': float((restored-blob).abs().float().mean())})
        del restored, payload, restore
    result = {'status':'passed','time_utc':now(),'context_tokens':len(ids),'raw_logits_max_error':max_error,
              'generation_matches_full_forward':True,'prediction':tokenizer.decode(manual_ids),
              'kv_shape':list(blob.shape),'methods':rows}
    save(output, result)
    log('smoke_passed', raw_logits_max_error=max_error, prediction=result['prediction'])


def quality(model, tokenizer, cfg, run_dir, manifest):
    data = {'status':'running','nll_definition':'teacher-forced continuation tokens 2..32 (31 targets)',
            'methods':list(METHODS),'continuation':[],'qa':[]}
    cases = load_longchat_segments(REPO/'test_data/longchat.jsonl',tokenizer,
                                  lengths=cfg['quality']['lengths'],sample_count=cfg['quality']['samples'],
                                  continuation_tokens=cfg['quality']['continuation'])
    for case in cases:
        blob = prefill(model, case['context_ids'])
        row = {'case_id':case['case_id'],'source_id':case['source_id'],'context_tokens':len(case['context_ids']),
               'continuation_tokens_scored':len(case['continuation_ids'])-1,
               'raw_tensor_bytes':blob.numel()*blob.element_size(),'methods':[]}
        for method in METHODS:
            payload, encode_s, restore = encoded(blob,method)
            restored, decode_s = timed(lambda:restore(payload))
            validate_blob(restored,blob)
            row['methods'].append({'method':method,'bytes':len(payload),
                'nll':continuation_nll(model,restored,case['continuation_ids']),
                'kv_mae':float((restored-blob).abs().float().mean()),
                'offline_encode_seconds':encode_s,'decode_seconds':decode_s})
            del payload,restored,restore
        data['continuation'].append(row)
        save(run_dir/'quality.json',data)
        manifest['progress']={'stage':'quality','continuation_complete':len(data['continuation']),'continuation_total':len(cases)}
        save(run_dir/'manifest.json',manifest)
        log('quality_context_complete',case_id=row['case_id'],complete=len(data['continuation']),total=len(cases))
        del blob
        cleanup()
    qas = qa_cases(tokenizer,cfg['qa'])
    save(run_dir/'qa_selection.json',[{k:v for k,v in row.items() if k!='ids'} | {'prompt_tokens':len(row['ids'])} for row in qas])
    for case in qas:
        # Keep the complete prompt: prefill all but its final token, replay that token.
        blob = prefill(model,case['ids'][:-1])
        row={'case_id':f"tqa-{case['source_index']}",'source_index':case['source_index'],
             'prompt_tokens':len(case['ids']),'answers':case['answers'],'methods':[]}
        for method in METHODS:
            payload,encode_s,restore=encoded(blob,method)
            restored,decode_s=timed(lambda:restore(payload))
            validate_blob(restored,blob)
            generated=greedy_from_cache(model,restored,case['ids'][-1:],cfg['qa']['max_new_tokens'],tokenizer.eos_token_id)
            prediction=tokenizer.decode(generated,skip_special_tokens=True).strip()
            row['methods'].append({'method':method,'bytes':len(payload),'prediction':prediction,
                                  **score_qa(prediction,case['answers']),
                                  'offline_encode_seconds':encode_s,'decode_seconds':decode_s})
            del payload,restored,restore
        data['qa'].append(row)
        save(run_dir/'quality.json',data)
        manifest['progress']={'stage':'qa','complete':len(data['qa']),'total':len(qas)}
        save(run_dir/'manifest.json',manifest)
        log('qa_complete',case_id=row['case_id'],complete=len(data['qa']),scores={r['method']:{'em':r['em'],'f1':r['f1']} for r in row['methods']})
        del blob
        cleanup()
    data['status']='completed'
    save(run_dir/'quality.json',data)
    return data


def ttft(model,tokenizer,cfg,run_dir,manifest):
    params=cfg['ttft']
    data={'status':'running','measurement':'paced local socket; complete receive, restore, first token; excludes offline encoding',
          'parameters':params,'cases':[]}
    suffix=tokenizer.encode('\nQuestion: What is the main topic?\nAnswer:',add_special_tokens=False)
    cases=load_longchat_segments(REPO/'test_data/longchat.jsonl',tokenizer,lengths=params['lengths'],
                                sample_count=params['samples'],continuation_tokens=2)
    for case_index,case in enumerate(cases):
        context=case['context_ids']
        text=tokenizer.decode(context,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        if tokenizer.encode(text,add_special_tokens=False)!=context:
            raise RuntimeError('Text/token roundtrip mismatch')
        blob=prefill(model,context)
        # Raw first-token parity guards positions and cache reuse at each tested length.
        raw_token=first_token(model,context,suffix,blob)
        direct_token=first_token(model,context,suffix)
        if raw_token!=direct_token:
            raise RuntimeError(f"Raw cache first-token mismatch at {case['case_id']}")
        payloads={'text':(text.encode(),0.0,None)}
        for method in ('uniform8','cachegen2','cachegen3'):
            payloads[method]=encoded(blob,method)
        del blob
        cleanup()
        row={'case_id':case['case_id'],'source_id':case['source_id'],'context_tokens':len(context),
             'raw_matches_direct_first_token':True,'direct_first_token_id':direct_token,'measurements':[]}
        schedule=measurement_schedule(list(payloads),params['bandwidth_gbps'],warmup=params['warmup'],
                                      repeats=params['repeats'],seed=cfg['seed']+case_index)
        for step_index,step in enumerate(schedule):
            method=step['method']; payload,encode_s,restore=payloads[method]
            torch.cuda.synchronize()
            start=time.perf_counter()
            received,transfer_s=rate_limited_transfer(payload,step['bandwidth_gbps'])
            if method=='text':
                t=time.perf_counter()
                ids=tokenizer.encode(received.decode(),add_special_tokens=False)
                if ids!=context:
                    raise RuntimeError('Text transport changed token IDs')
                restore_s=time.perf_counter()-t
                token,first_s=timed(lambda:first_token(model,ids,suffix))
            else:
                restored,restore_s=timed(lambda:restore(received))
                token,first_s=timed(lambda:first_token(model,context,suffix,restored))
                del restored
            elapsed=time.perf_counter()-start
            if step['phase']=='measure':
                row['measurements'].append({**step,'bytes':len(payload),'offline_encode_seconds':encode_s,
                    'transfer_seconds':transfer_s,'reconstruct_seconds':restore_s,'first_token_seconds':first_s,
                    'ttft_wall_seconds':elapsed,'effective_bandwidth_gbps':len(payload)*8/transfer_s/1e9,
                    'first_token_id':token,'matches_direct_first_token':token==direct_token})
            del received
            # Persist partial timing records as well as each completed context.
            if (step_index+1)%len(payloads)==0:
                save(run_dir/'ttft.json',{**data,'current_case':row})
                manifest['progress']={'stage':'ttft','contexts_complete':len(data['cases']),'contexts_total':len(cases),
                                      'current_case':case['case_id'],'steps_complete':step_index+1,'steps_total':len(schedule)}
                save(run_dir/'manifest.json',manifest)
        data['cases'].append(row)
        save(run_dir/'ttft.json',data)
        log('ttft_context_complete',case_id=row['case_id'],complete=len(data['cases']),total=len(cases),
            observations=len(row['measurements']))
        del payloads,payload,restore
        cleanup()
    data['status']='completed'
    save(run_dir/'ttft.json',data)
    return data


def quantile(values,q):
    a=sorted(values); rank=(len(a)-1)*q; lo=int(rank); hi=min(lo+1,len(a)-1)
    return a[lo]+(a[hi]-a[lo])*(rank-lo)


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(run_dir,cfg):
    quality_data=json.loads((run_dir/'quality.json').read_text())
    timing=json.loads((run_dir/'ttft.json').read_text())
    if quality_data['status']!='completed' or timing['status']!='completed':
        raise RuntimeError('Incomplete results cannot generate a completed report')
    if len(quality_data['continuation'])!=cfg['quality']['samples']*len(cfg['quality']['lengths']) or len(quality_data['qa'])!=cfg['qa']['samples']:
        raise RuntimeError('Quality case count mismatch')
    qa_summary=[]
    raw_correct={r['case_id'] for r in quality_data['qa'] if next(x for x in r['methods'] if x['method']=='raw')['em']==1}
    for method in METHODS:
        rows=[next(x for x in r['methods'] if x['method']==method) for r in quality_data['qa']]
        kept=sum(next(x for x in r['methods'] if x['method']==method)['em'] for r in quality_data['qa'] if r['case_id'] in raw_correct)
        qa_summary.append({'method':method,'questions':len(rows),'mean_em':statistics.mean(r['em'] for r in rows),
                           'mean_f1':statistics.mean(r['f1'] for r in rows),'raw_em_correct':len(raw_correct),
                           'retained_raw_em_correct':kept,'raw_correct_retention':kept/len(raw_correct) if raw_correct else None})
    qa_by_method={r['method']:r for r in qa_summary}
    quality_rows=[]
    for length in cfg['quality']['lengths']:
        cases=[r for r in quality_data['continuation'] if r['context_tokens']==length]
        for method in METHODS:
            pairs=[(next(x for x in r['methods'] if x['method']=='raw'),next(x for x in r['methods'] if x['method']==method)) for r in cases]
            delta=statistics.mean(m['nll']-r['nll'] for r,m in pairs)
            retention=qa_by_method[method]['raw_correct_retention']
            quality_rows.append({'context_tokens':length,'method':method,'contexts':len(cases),
                'mean_bytes':statistics.mean(m['bytes'] for _,m in pairs),
                'raw_serialized_over_method':sum(r['bytes'] for r,_ in pairs)/sum(m['bytes'] for _,m in pairs),
                'raw_tensor_over_method':sum(r['raw_tensor_bytes'] for r in cases)/sum(m['bytes'] for _,m in pairs),
                'mean_nll':statistics.mean(m['nll'] for _,m in pairs),'paired_nll_delta':delta,
                'mean_kv_mae':statistics.mean(m['kv_mae'] for _,m in pairs),
                'mean_offline_encode_seconds':statistics.mean(m['offline_encode_seconds'] for _,m in pairs),
                'mean_decode_seconds':statistics.mean(m['decode_seconds'] for _,m in pairs),
                'passes_descriptive_threshold':retention is not None and retention>=cfg['quality_tolerances']['minimum_raw_correct_qa_retention'] and delta<=cfg['quality_tolerances']['max_nll_delta']})
    groups=defaultdict(list)
    if len(timing['cases'])!=cfg['ttft']['samples']*len(cfg['ttft']['lengths']):
        raise RuntimeError('TTFT context count mismatch')
    expected=cfg['ttft']['repeats']*4*len(cfg['ttft']['bandwidth_gbps'])
    for case in timing['cases']:
        if len(case['measurements'])!=expected:
            raise RuntimeError('TTFT observation count mismatch')
        counts=Counter((r['method'],r['bandwidth_gbps']) for r in case['measurements'])
        if len(counts)!=4*len(cfg['ttft']['bandwidth_gbps']) or any(n!=cfg['ttft']['repeats'] for n in counts.values()):
            raise RuntimeError('Unbalanced method/bandwidth measurements')
        for row in case['measurements']:
            groups[(case['context_tokens'],row['bandwidth_gbps'],row['method'])].append(row)
    ttft_rows=[]
    for (length,bw,method),rows in sorted(groups.items()):
        values=[r['ttft_wall_seconds'] for r in rows]
        ttft_rows.append({'context_tokens':length,'bandwidth_gbps':bw,'method':method,
            'contexts':cfg['ttft']['samples'],'observations':len(rows),'median_seconds':statistics.median(values),
            'p95_seconds':quantile(values,.95),'mean_transfer_seconds':statistics.mean(r['transfer_seconds'] for r in rows),
            'mean_restore_seconds':statistics.mean(r['reconstruct_seconds'] for r in rows),
            'mean_first_token_seconds':statistics.mean(r['first_token_seconds'] for r in rows),
            'mean_effective_bandwidth_gbps':statistics.mean(r['effective_bandwidth_gbps'] for r in rows),
            'first_token_match_fraction':statistics.mean(r['matches_direct_first_token'] for r in rows)})
    for row in ttft_rows:
        base=next(r for r in ttft_rows if r['context_tokens']==row['context_tokens'] and r['bandwidth_gbps']==row['bandwidth_gbps'] and r['method']=='text')
        uniform=next(r for r in ttft_rows if r['context_tokens']==row['context_tokens'] and r['bandwidth_gbps']==row['bandwidth_gbps'] and r['method']=='uniform8')
        row['speedup_vs_text']=base['median_seconds']/row['median_seconds']
        row['speedup_vs_uniform8']=uniform['median_seconds']/row['median_seconds']
    write_csv(run_dir/'quality.csv',quality_rows);write_csv(run_dir/'qa.csv',qa_summary);write_csv(run_dir/'ttft.csv',ttft_rows)
    def table(rows,cols):
        def fmt(value):
            return f'{value:.5g}' if isinstance(value,float) else str(value)
        return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+
                         ['| '+' | '.join(fmt(r.get(c)) for c in cols)+' |' for r in rows])
    report=['# Mistral-7B CacheGen 小规模复现',
        f'模型：{MODEL_ID}；revision：{REVISION}。当前批次使用 RTX 4090 D，FastChat 8bit 权重，FP16 KV。',
        '## 实验口径',
        '30 个自然文本上下文（10 条源文本×3 长度）、10 条完整短 TriviaQA、540 次正式 TTFT 计时。种子为 0。',
        '自然文本 NLL 对 32-token 后续文本中的第 2–32 个 token 评分。TriviaQA 按完整 32-token 生成结果计算标准化 EM/F1，取答案别名最大分。',
        '体积包含实际序列化元数据。uniform4 使用位打包。TTFT 通过本机限速 socket 传送实际字节，收完后恢复并生成首 token；离线编码不计入 TTFT。',
        '作者仓库编码路径没有论文 anchor/delta 阶段；本实验没有跨服务器传输和传输—解码流水线，不构成完整论文系统复现。',
        '## 配对体积与续写质量',
        table(quality_rows,['context_tokens','method','contexts','raw_serialized_over_method','paired_nll_delta','mean_kv_mae','passes_descriptive_threshold']),
        '描述性门槛：平均配对 NLL 增量≤0.02，且保留≥90% 原始 KV 的 EM 正确题。QA 来自独立的混合长度样本，不能解释为每个长度的 QA 等价性证明；小样本门槛也不构成统计非劣证明。',
        '## 完整 TriviaQA',table(qa_summary,['method','questions','mean_em','mean_f1','raw_em_correct','raw_correct_retention']),
        '逐题输入选择见 qa_selection.json，原始答案与预测见 quality.json。',
        '## 传输与首 token 延迟',
        table(ttft_rows,['context_tokens','bandwidth_gbps','method','observations','median_seconds','p95_seconds','speedup_vs_text','speedup_vs_uniform8']),
        '速度比>1 表示更快。中位数和 P95 汇总同条件的 3 个上下文×5 次重复；重复不增加独立上下文数量。缓存方法与文本的首 token 一致率是诊断指标。',
        '逐项传输、恢复、首 token 开销和有效带宽见 ttft.csv。高带宽设定不等同于实际吞吐。',
        '## 可追溯信息','metadata.json 保存模型校验、源码/数据指纹、CUDA 扩展和环境；manifest.json 保存进度与运行状态。run.log 保存完整日志。']
    (run_dir/'report.md').write_text('\n\n'.join(report)+'\n')
    save(run_dir/'summary.json',{'quality':quality_rows,'qa':qa_summary,'ttft':ttft_rows})


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'config.json');args=p.parse_args()
    run_dir=args.run_dir
    cfg=json.loads(args.config.read_text())
    manifest={'status':'running','started_utc':now(),'pid':os.getpid(),
              'runner_pid':int(os.environ.get('CACHEGEN_SUPERVISOR_PID',os.getpid())),
              'configuration':cfg,'stages':{},'progress':{'stage':'metadata'}}
    save(run_dir/'manifest.json',manifest)
    try:
        torch.set_num_threads(8);torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        meta=metadata(cfg,args.config.resolve());save(run_dir/'metadata.json',meta)
        manifest['stages']['metadata']='completed'
        manifest['progress']={'stage':'load_model'};save(run_dir/'manifest.json',manifest)
        from src.utils import define_model_and_tokenizer
        log('load_model_start',snapshot=str(SNAPSHOT))
        model,tokenizer=define_model_and_tokenizer(str(SNAPSHOT),num_gpus=1,max_gpu_memory=20)
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        meta['attention_implementation']=model.config._attn_implementation
        meta['attention_class']=type(model.model.layers[0].self_attn).__name__
        save(run_dir/'metadata.json',meta)
        log('model_loaded',attention=meta['attention_class'])
        manifest['progress']={'stage':'smoke'};save(run_dir/'manifest.json',manifest)
        smoke(model,tokenizer,run_dir/'smoke.json');cleanup()
        manifest['stages']['smoke']='completed'
        manifest['progress']={'stage':'quality'};save(run_dir/'manifest.json',manifest)
        quality(model,tokenizer,cfg,run_dir,manifest)
        manifest['stages']['quality']='completed';save(run_dir/'manifest.json',manifest)
        ttft(model,tokenizer,cfg,run_dir,manifest)
        manifest['stages']['ttft']='completed';save(run_dir/'manifest.json',manifest)
        summarize(run_dir,cfg)
        manifest.update(status='completed',finished_utc=now(),progress={'stage':'completed'},
                        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        manifest['stages']['report']='completed';save(run_dir/'manifest.json',manifest)
        log('experiment_completed',run_dir=str(run_dir))
    except BaseException as error:
        manifest.update(status='failed',finished_utc=now(),error=repr(error),traceback=traceback.format_exc())
        save(run_dir/'manifest.json',manifest);traceback.print_exc();raise


if __name__=='__main__':
    with torch.inference_mode():
        main()
