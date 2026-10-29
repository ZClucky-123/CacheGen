"""Runtime source fingerprints: compression evidence and network code are separate.

Tests and shell orchestration do not determine stage1 numerical results. Shared
model, codec, data/scoring helpers and the stage1 runner remain protected.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_paths(scope='stage1', root=None):
    root = Path(root) if root is not None else ROOT
    if scope not in ('stage1', 'stage2'):
        raise ValueError('Expected stage1 or stage2 provenance scope')
    names = [
        'paper_compression/' + name + '.py' for name in (
            '__init__', 'config', 'quantization', 'probability', 'entropy', 'codec',
            'cuda_backend', 'theory', 'study_config', 'insights', 'provenance')]
    names += ['paper_compression/configs/core-study.json', 'tools/paper_compression.py',
              'tools/paper_study.py', 'scripts/experiment-env.sh']
    names += ['experiments/' + name + '.py' for name in ('__init__', 'engine', 'config', 'stable', 'scoring')]
    names += ['mini_repro/' + name + '.py' for name in ('__init__', 'run', 'data', 'experiment_common')]
    names += ['LMCache/lmcache/' + name + '.py' for name in ('__init__', 'config', 'logging', 'utils')]
    paths = [root / name for name in names]
    paths += list((root / 'LMCache/lmcache/storage_backend/serde').glob('*.py'))
    paths += list((root / 'LMCache/third_party/torchac_cuda').glob('*.cu'))
    paths += list((root / 'LMCache/third_party/torchac_cuda').glob('*.cuh'))
    if scope == 'stage2':
        paths += [root / name for name in (
            'paper_compression/network.py', 'paper_compression/policy.py', 'tools/paper_stages.py',
            'tools/run_paper_core.sh', 'tools/run_paper_study.sh')]
        paths += list((root / 'experiments').glob('*.py'))
        paths += list((root / 'mini_repro').glob('*.py'))
    return sorted(set(paths))


def provenance(scope='stage1', root=None):
    root = Path(root) if root is not None else ROOT
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths(scope, root)}
