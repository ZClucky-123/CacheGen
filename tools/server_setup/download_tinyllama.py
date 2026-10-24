"""Complete the public, pinned TinyLlama weight download; never prints credentials."""
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ['HF_HUB_OFFLINE'] = '0'
from huggingface_hub import hf_hub_download
target = REPO / '.local/tinyllama-download.json'
target.parent.mkdir(parents=True, exist_ok=True)
try:
    path = Path(hf_hub_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'model.safetensors',
        revision='fe8a4ea1ffedaf415f4da2f062534de366a451e6',
        cache_dir=Path(os.environ.get('HF_HOME', REPO / '.venv/huggingface')) / 'hub',
        local_files_only=False, token=False, resume_download=True))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for piece in iter(lambda: stream.read(8 << 20), b''):
            digest.update(piece)
    expected = '6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933'
    if digest.hexdigest() != expected or path.stat().st_size != 2200119864:
        raise RuntimeError('Pinned TinyLlama weight size/SHA-256 mismatch')
    record = {'status': 'verified', 'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    target.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
except BaseException as error:
    target.write_text(json.dumps({'status': 'failed', 'error_type': type(error).__name__}) + '\n')
    raise
