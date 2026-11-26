"""Server readiness and tokenizer-only audit. Does not load weights or run inference."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    from transformers import AutoTokenizer
    from experiments.engine import snapshot_path
    from tools.full_length_qa import QUESTION_MARKER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg_path = ROOT/'paper_compression/configs/delta-diagnostics.json'
    cfg = json.loads(cfg_path.read_text())
    core = json.loads((ROOT/cfg['core_config']).read_text())
    model = core['model']
    assert (model['id'], model['revision'], model['weight_loading']) == (
        cfg['model_id'],cfg['model_revision'],cfg['weight_loading'])
    assert model['weight_loading'] == 'fp16'
    data_path = ROOT/cfg['dataset']
    manifest = json.loads((ROOT/cfg['dataset_manifest']).read_text())
    assert sha(data_path) == manifest['output_sha256']
    data = [json.loads(s) for s in data_path.read_text().splitlines() if s.strip()]
    old = [json.loads(s) for s in (ROOT/'test_data/longchat.jsonl').read_text().splitlines() if s.strip()]
    assert len(data) == cfg['expected_samples'] and data[:50] == old
    assert [r['test_id'] for r in data] == list(range(200))
    assert len({r['prompt'] for r in data}) == 200
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    snapshot = snapshot_path(model)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
    index = json.loads((snapshot/'model.safetensors.index.json').read_text())
    weights = sorted(set(index['weight_map'].values()))
    weight_hashes = {name:sha(snapshot/name) for name in weights}
    lengths=[]
    for row in data:
        prompt=row['prompt']
        assert prompt.count(QUESTION_MARKER)==1
        cut=prompt.index(QUESTION_MARKER)
        context=tokenizer.encode(prompt[:cut])
        question=tokenizer.encode(prompt[cut:],add_special_tokens=False)
        assert context[0]==tokenizer.bos_token_id
        assert len(row['label'])==len(set(row['label']))==15
        assert len(context)+len(question)+cfg['generation_tokens']<=model['context_limit']
        lengths.append({'source_id':row['test_id'],'origin':'historical' if row['test_id']<50 else 'new_permutation',
                        'context_tokens':len(context),'question_tokens':len(question),
                        'context_token_sha256':hashlib.sha256(json.dumps(context).encode()).hexdigest(),
                        'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
    summary={}
    for name in ['historical','new_permutation','all']:
        values=[r['context_tokens'] for r in lengths if name=='all' or r['origin']==name]
        summary[name]={'n':len(values),'min':min(values),'median':statistics.median(values),'max':max(values)}
    result={'status':'ready_for_diagnostic_pilot','preparation_only':True,
            'inference_executed':False,'model':model,'snapshot':str(snapshot),
            'weights_sha256':weight_hashes,'weight_file_count':len(weights),
            'tokenizer_files_sha256':{p.name:sha(p) for p in sorted(snapshot.glob('*'))
                if p.is_file() and p.suffix=='.json' and 'safetensors.index' not in p.name},
            'config_sha256':sha(cfg_path),'dataset_sha256':sha(data_path),
            'original_dataset_sha256':sha(ROOT/'test_data/longchat.jsonl'),
            'length_summary':summary,'samples':lengths,
            'wall_seconds':time.monotonic()-started,
            'note':'200 same-source prompt permutations, not recovered paper-original200; no fresh independent source documents or new model weights'}
    (args.output/'readiness.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'length_summary':summary,
                      'weight_files':len(weights),'inference_executed':False},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
