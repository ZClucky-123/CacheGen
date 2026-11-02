"""Independent joint-v2 settings and repository-root output confinement."""
import copy
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / 'optimization/configs/joint-v2.json'


def repository_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def run_directory(path=None, *, run_id=None):
    root = (REPO_ROOT / 'optimization/output').resolve()
    if path is None:
        name = run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
            raise ValueError('Invalid run_id')
        target = root / name
    else:
        target = repository_path(path)
    if target == root or root not in target.parents:
        raise ValueError('Optimization run-dir must be optimization/output/<run_id>')
    return target


def load_joint_config(path=None):
    cfg = json.loads(repository_path(path or DEFAULT_CONFIG).read_text(encoding='utf-8'))
    cfg = copy.deepcopy(cfg)
    required = {
        'protocol': 'cachegen-joint-v2', 'preparation_policy': 'select_then_materialize',
        'source_policy': 'gpu_master', 'symbol_retention': 'current_job_only',
        'receive_capacity': 1, 'gpu_execution': 'one_default_stream_executor',
        'gpu_ready_priority': 'restore_before_encode', 'clock': 'perf_counter_ns',
        'deadline_unit': 'absolute_monotonic_ns',
    }
    for key, value in required.items():
        if cfg.get(key) != value:
            raise ValueError(f'joint.{key} must be {value!r}')
    if not cfg.get('encodings') or not set(cfg['encodings']) <= {'ac_count', 'ac_cdf', 'fixed_bits'}:
        raise ValueError('Invalid encodings')
    if not set(cfg.get('qualities', [])) <= {1, 2, 3} or not cfg['qualities']:
        raise ValueError('Invalid qualities')
    if cfg.get('quality_mode') not in ('adaptive', 'fixed') or cfg.get('min_quality') not in (1, 2, 3):
        raise ValueError('Invalid quality mode')
    if cfg.get('execution_mode') not in ('serial', 'pipeline') or cfg.get('generation_lookahead') not in (0, 1):
        raise ValueError('Invalid execution mode/lookahead')
    if cfg.get('chunk_tokens') != 256:
        raise ValueError('Joint-v2 application chunk size is 256')
    for key in ('packet_cache_bytes_per_context', 'max_frame_bytes'):
        if type(cfg.get(key)) is not int or cfg[key] <= 0:
            raise ValueError(f'Invalid {key}')
    for key in ('io_idle_timeout_s', 'bandwidth_initial_gbps'):
        if not isinstance(cfg.get(key), (float, int)) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if cfg.get('residual_min_sources') != 5 or cfg.get('residual_min_requests') != 20:
        raise ValueError('Residual minimums must be 5 sources and 20 requests')
    if not set(cfg.get('benchmark_cache_conditions', cfg['cache_conditions'])) <= {'cold','warm','mixed','quality_change','representation_miss'}:
        raise ValueError('Invalid benchmark cache condition')
    return cfg
