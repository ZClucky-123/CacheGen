"""Deterministic selections shared by every model and task."""
import hashlib
import json
from pathlib import Path
from mini_repro.experiment_common import build_positioned_case, load_longchat_segments
from mini_repro.adaptive import load_bandwidth_traces


def load_triviaqa(path, tokenizer, cfg):
    cases = []
    # JSONL physical file lines: U+2028/U+0085 inside JSON strings remain intact.
    with Path(path).open(encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            row = json.loads(line)
            ids = tokenizer.encode(row['prompt'], add_special_tokens=False, verbose=False)
            if len(ids) <= cfg['max_prompt_tokens']:
                cases.append({'case_id': f'tqa-{index}', 'source_id': index, 'input_ids': ids,
                              'answers': row['answers'], 'context_tokens': len(ids) - 1,
                              'prompt_sha256': hashlib.sha256(row['prompt'].encode()).hexdigest()})
                if len(cases) == cfg['samples']:
                    return cases
    raise ValueError(f'Insufficient complete TriviaQA rows: {len(cases)}/{cfg["samples"]}')


def selection(repo, cfg, tokenizer):
    data = {k: repo / v for k, v in cfg['data'].items()}
    result = {}
    for section in ('quality', 'ttft'):
        p = cfg[section]
        result[section] = load_longchat_segments(data['longchat'], tokenizer, p['lengths'], p['samples'],
                                               p.get('continuation', 2))
        for row in result[section]:
            row['context_tokens'] = len(row['context_ids'])
    result['controlled'] = []
    for length in cfg['quality']['lengths']:
        for position in ('start', 'middle', 'end'):
            for index in range(cfg['quality']['qa_per_position']):
                row = build_positioned_case(tokenizer, index, length, position)
                row.update(source_id=index, context_tokens=length, answers=[row['answer']])
                result['controlled'].append(row)
    result['triviaqa'] = load_triviaqa(data['triviaqa'], tokenizer, cfg['triviaqa'])
    result['adaptive'] = []
    for index in range(cfg['adaptive']['cases']):
        row = build_positioned_case(tokenizer, index, cfg['adaptive']['length'], ('start', 'middle', 'end')[index % 3])
        row.update(source_id=index, context_tokens=cfg['adaptive']['length'], answers=[row['answer']])
        result['adaptive'].append(row)
    p = cfg['adaptive']
    result['traces'] = load_bandwidth_traces(data['bandwidth'], p['traces'],
                                            (p['length'] + p['chunk_size'] - 1) // p['chunk_size'], 1.0)
    for row in result['controlled'] + result['adaptive']:
        row['suffix_ids'] = tokenizer.encode(f'\nQuestion: {row["question"]}\nAnswer:', add_special_tokens=False)
    for row in result['ttft']:
        row['suffix_ids'] = tokenizer.encode('\nQuestion: What is the main topic?\nAnswer:', add_special_tokens=False)
    return result
