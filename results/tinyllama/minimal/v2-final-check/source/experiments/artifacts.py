"""Version checks, complete source snapshots, preflight and artifact integrity."""
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from experiments.records import fingerprint, save


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def source_paths(repo):
    paths = []
    for folder in ('experiments', 'mini_repro', 'src', 'scripts', 'tests', 'LMCache/lmcache', 'LMCache/third_party/torchac_cuda'):
        paths += [p for p in (repo / folder).rglob('*') if p.is_file()
                  and p.suffix in ('.py', '.sh', '.json', '.cu', '.cuh', '.cpp', '.h')
                  and not p.is_symlink() and not any(x in p.parts for x in ('build', '__pycache__', 'runs', 'deliverables'))]
    paths += [repo / 'env.yaml', repo / 'mini_repro/requirements-mini.txt']
    return sorted(set(paths))


def code_data_identity(repo, cfg):
    return {'config': fingerprint(cfg),
            'code': {str(p.relative_to(repo)): sha256(p) for p in source_paths(repo)},
            'data': {str(p): sha256(repo / p) for p in cfg['data'].values()}}


def runtime_identity():
    import torch
    import torchac_cuda
    packages = ('torch', 'transformers', 'numpy', 'fschat', 'accelerate', 'lmcache', 'torchac', 'torchac_cuda', 'matplotlib')
    return {'packages': {p: importlib.metadata.version(p) for p in packages},
            'python': platform.python_version(), 'cuda': torch.version.cuda,
            'gpu_name': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
            'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip(),
            'gpu_uuid_inventory': subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip(),
            'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'all'),
            'codec_binary_sha256': sha256(torchac_cuda.__file__),
            'threads': torch.get_num_threads(),
            'cpu': platform.processor(), 'cpu_quota': read_limit('/sys/fs/cgroup/cpu.max'),
            'memory_limit': read_limit('/sys/fs/cgroup/memory.max'),
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}


def read_limit(path):
    return Path(path).read_text().strip() if Path(path).is_file() else None


def preflight(engine, cfg, selected, run):
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    figure.savefig(io.BytesIO(), format='png')
    plt.close(figure)
    spec = cfg['model']
    model_config = json.loads((engine.snapshot / 'config.json').read_text())
    geometry = (model_config['num_hidden_layers'], model_config['num_key_value_heads'],
                model_config['hidden_size'] // model_config['num_attention_heads'])
    if geometry != tuple(spec[k] for k in ('layers', 'kv_heads', 'head_dim')) or spec['context_limit'] > model_config['max_position_embeddings']:
        raise ValueError('Model window/geometry does not match the adapter before weight loading')
    max_tokens = 0
    for field in ('quality', 'controlled', 'triviaqa', 'ttft', 'adaptive'):
        for case in selected[field]:
            if field == 'quality':
                total = len(case['context_ids']) + len(case['continuation_ids'])
            elif field == 'triviaqa':
                total = len(case['input_ids']) + cfg['triviaqa']['max_new_tokens']
            else:
                limit = cfg['quality']['max_new_tokens'] if field == 'controlled' else cfg['adaptive']['max_new_tokens'] if field == 'adaptive' else 1
                total = case['context_tokens'] + len(case['suffix_ids']) + limit
            if total > spec['context_limit']:
                raise ValueError(f'{case["case_id"]}: {total} tokens exceed window {spec["context_limit"]}')
            max_tokens = max(max_tokens, total)
    text_wire_checks = []
    for case in selected['ttft']:
        payload = engine.text_payload(case['context_ids'])
        text_wire_checks.append(engine.text_metadata(payload))
    for case in selected['adaptive']:
        for start in range(0, len(case['input_ids']), cfg['adaptive']['chunk_size']):
            payload = engine.text_payload(case['input_ids'][start:start + cfg['adaptive']['chunk_size']])
            text_wire_checks.append(engine.text_metadata(payload))
    kv_bytes = spec['layers'] * 2 * spec['kv_heads'] * spec['head_dim'] * max_tokens * 2
    weights = sum(p.stat().st_size for p in engine.snapshot.glob('*.safetensors'))
    if not weights:
        weights = sum(p.stat().st_size for p in engine.snapshot.glob('pytorch_model*.bin'))
    if not weights:
        raise ValueError('Pinned local model weights are missing')
    loaded_weights = weights if spec['weight_loading'] == 'fp16' else weights * 0.6
    # Conservative reserve for decode CDF/indices, attention activations, allocator.
    gpu_required = int(loaded_weights + kv_bytes * 8 + 3 * 2**30)
    free_gpu, total_gpu = torch.cuda.mem_get_info()
    host_limit = read_limit('/sys/fs/cgroup/memory.max')
    host_total = int(host_limit) if host_limit and host_limit.isdigit() else os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    host_used = int(read_limit('/sys/fs/cgroup/memory.current') or 0)
    memory_stat = read_limit('/sys/fs/cgroup/memory.stat')
    reclaimable = dict(line.split() for line in memory_stat.split('\n')).get('inactive_file', '0') if memory_stat else '0'
    host_used = max(0, host_used - int(reclaimable))
    host_required = int(weights * 1.2 + kv_bytes * 4 + 2 * 2**30)
    disk_free = shutil.disk_usage(run).free
    disk_required = max(512 << 20, len(selected['quality']) * 100000 + len(selected['controlled']) * 50000)
    report = {'maximum_total_tokens': max_tokens, 'largest_fp16_kv_bytes': kv_bytes,
              'estimated_gpu_required_bytes': gpu_required, 'gpu_free_bytes': free_gpu, 'gpu_total_bytes': total_gpu,
              'estimated_host_required_bytes': host_required, 'host_limit_bytes': host_total,
              'host_used_bytes': host_used, 'disk_free_bytes': disk_free, 'disk_required_bytes': disk_required,
              'host_reclaimable_inactive_file_bytes': int(reclaimable),
              'plot_preflight': 'passed', 'matplotlib': matplotlib.__version__,
              'text_wire_checks': {'conditions': len(text_wire_checks),
                  'patched_conditions': sum(r['token_patch_count'] > 0 for r in text_wire_checks),
                  'maximum_inserted_patch_tokens': max((r['patch_inserted_tokens'] for r in text_wire_checks), default=0)},
              'resource_estimates_are_conservative_not_allocations': True}
    save(run / 'preflight.json', report)
    if gpu_required > free_gpu or host_required > host_total - host_used or disk_free < disk_required:
        raise ValueError(f'Preflight resource estimate exceeds available resources; see {run / "preflight.json"}')
    return report


def collect_metadata(repo, run, engine, cfg, identity):
    import torch
    import torchac_cuda
    model_files = {p.name: sha256(p) for p in engine.snapshot.iterdir() if p.is_file()}
    # Copy all measured repository helpers with directory structure, not flattened basenames.
    for path in source_paths(repo):
        destination = run / 'source' / path.relative_to(repo)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    for role, name in cfg['data'].items():
        original = (repo / name).resolve()
        relative = original.relative_to(repo) if original.is_relative_to(repo) else Path('external-data') / f'{role}.jsonl'
        destination = run / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, destination)
    dependency_sources = {}
    import transformers.models.mistral.modeling_mistral as mistral
    import transformers.models.llama.modeling_llama as llama
    import transformers.cache_utils as cache_utils
    import fastchat.model.compression as compression
    import fastchat.model.model_adapter as model_adapter
    import transformers.modeling_attn_mask_utils as masks
    import transformers.modeling_utils as modeling
    import transformers.generation.utils as generation
    for module in (mistral, llama, cache_utils, compression, model_adapter, masks, modeling, generation):
        path = Path(module.__file__)
        dependency_sources[module.__name__] = sha256(path)
        destination = run / 'source/dependencies' / (module.__name__.replace('.', '/') + '.py')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    meta = {'protocol': cfg['protocol'], 'model': cfg['model'], 'model_files_sha256': model_files,
            'model_snapshot': str(engine.snapshot), 'tokenizer_class': type(engine.tokenizer).__name__,
            'runtime': runtime_identity(), 'identity': identity,
            'dependency_sources_sha256': dependency_sources, 'platform': platform.platform(),
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
            'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=repo, text=True).splitlines(),
            'codec_binary': str(torchac_cuda.__file__),
            'nvcc': subprocess.check_output([str(Path(os.environ['CUDA_HOME']) / 'bin/nvcc'), '--version'], text=True)}
    (run / 'source/local.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD', '--binary'], cwd=repo))
    (run / 'dependencies.txt').write_text(subprocess.check_output([os.sys.executable, '-m', 'pip', 'freeze'], text=True))
    (run / 'hardware.txt').write_text(subprocess.check_output(['nvidia-smi', '-q'], text=True) + '\n' + Path('/proc/cpuinfo').read_text())
    save(run / 'metadata.json', meta)
    return meta


def seal(run):
    files = {str(p.relative_to(run)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
             for p in sorted(run.rglob('*')) if p.is_file()
             and p.name not in ('artifacts.json', 'records.sqlite3-wal', 'records.sqlite3-shm')}
    save(run / 'artifacts.json', {'status': 'completed', 'files': files})


def verify(run):
    saved = json.loads((run / 'artifacts.json').read_text())
    for name, row in saved['files'].items():
        path = run / name
        if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError(f'Archive integrity failed: {name}')
    return len(saved['files'])
