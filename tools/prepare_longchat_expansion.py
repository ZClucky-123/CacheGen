"""Assemble a separately named 200-prompt candidate set; stdlib only, no inference.

Preserve the 50 CacheGen prompts and sample 150 new 15-topic permutations from
the same official pool. This does NOT recover the paper's original 200 samples.
"""
from pathlib import Path
import hashlib
import json
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'test_data/longchat-expansion'
COMMIT = '4b5faa52fda961b8ea5125b15ae984ff0d1c0f9d'
SEED = 20261125


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main():
    output = FOLDER / 'longchat_200_expanded_candidate.jsonl'
    manifest = FOLDER / 'manifest.json'
    if output.exists() or manifest.exists():
        raise FileExistsError('Preserve existing candidate and manifest; do not overwrite')
    original_path = ROOT / 'test_data/longchat.jsonl'
    original = read(original_path)
    upstream = read(FOLDER/'sources/15_topics.jsonl')
    pool = read(FOLDER/'sources/conversations.jsonl')
    assert len(original) == len(upstream) == 50
    assert len(pool) == len({r['topic'] for r in pool}) == 30
    by_topic = {r['topic']:r for r in pool}
    body = ''.join(by_topic[t]['conversation'] for t in upstream[0]['topics'])
    pos = upstream[0]['prompt'].index(body)
    prefix = upstream[0]['prompt'][:pos]
    suffix = upstream[0]['prompt'][pos+len(body):]
    pos = original[0]['prompt'].index(upstream[0]['prompt'])
    wrap_prefix = original[0]['prompt'][:pos]
    wrap_suffix = original[0]['prompt'][pos+len(upstream[0]['prompt']):]
    mapping = []
    for index,(row,ref) in enumerate(zip(original,upstream)):
        assert row['test_id'] == ref['test_id'] == index
        assert row['label'] == ref['topics']
        assert ref['prompt'] == prefix + ''.join(by_topic[t]['conversation'] for t in ref['topics']) + suffix
        assert row['prompt'] == wrap_prefix + ref['prompt'] + wrap_suffix
        mapping.append({'test_id':index,'origin':'cachegen_published_50',
            'official_topic_ids':[by_topic[t]['topic_id'] for t in row['label']]})
    rows = list(original)
    seen = {r['prompt'] for r in rows}
    rng = random.Random(SEED)
    while len(rows) < 200:
        selection = rng.sample(pool,15)
        prompt = wrap_prefix + prefix + ''.join(r['conversation'] for r in selection) + suffix + wrap_suffix
        if prompt in seen:
            continue
        seen.add(prompt)
        index = len(rows)
        rows.append({'test_id':index,'prompt':prompt,'label':[r['topic'] for r in selection]})
        mapping.append({'test_id':index,'origin':'new_same_source_permutation',
            'official_topic_ids':[r['topic_id'] for r in selection]})
    assert len(seen) == 200
    output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    model = json.loads((ROOT/'paper_compression/configs/core-study.json').read_text())['model']
    metadata = {'status':'candidate_only_not_paper_original_200','count':200,
        'published_original_count':50,'new_permutation_count':150,
        'topics_per_prompt':15,'base_conversations':30,'seed':SEED,
        'sampling':'uniform ordered sampling without replacement within each prompt; Python random.sample',
        'python_version':sys.version,'model_fixed':model,
        'upstream_repo':'https://github.com/DachengLi1/LongChat',
        'upstream_commit':COMMIT,'upstream_branch':'longeval',
        'sources_sha256':{p.name:sha(p) for p in sorted((FOLDER/'sources').iterdir()) if p.is_file()},
        'original_cachegen_sha256':sha(original_path),'output_sha256':sha(output),
        'builder_sha256':sha(Path(__file__)), 'original50_mapping_verified':True,
        'splits_assigned':False,'token_lengths_verified_with_pinned_tokenizer':False,
        'warnings':['Paper original 200 selection/seed not recovered.',
                    'Prompt permutations reuse a pool of only 30 conversations; not 200 independent source documents.',
                    'Original IDs30-49 were used for historical calibration; do not label all200 as untouched held-out data.',
                    'Original input file and existing experiment configuration remain unchanged.'],
        'samples':mapping}
    manifest.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print('Prepared',output,'with 200 unique prompts; original50 matched; new150 explicitly labeled.')


if __name__ == '__main__':
    main()
