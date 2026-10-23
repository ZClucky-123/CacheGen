# CacheGen environment for Mistral-7B

Project: /home/aaa/CacheGen
Model: mistral-community/Mistral-7B-v0.2
Pinned model revision: 80cf49b18de6354fd8a6d1e81c177dd830e05eea

## Activate

```bash
cd /home/aaa/CacheGen
source .venv/cachegen-env.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

The environment uses Python 3.10.12, PyTorch 2.3.1 / CUDA 12.1, and Transformers 4.42.3. Python packages are in .venv; the CUDA compiler and Python development headers are in .cuda-12.1. The activation script configures both directories and the model cache.

The model cache is in .venv/huggingface. Its three weight shards are checked against Hugging Face SHA-256 metadata; see .venv/mistral-download-verified.json. The cached main reference points to the fixed revision so the original model ID works offline.

## Verify after migration

```bash
python -m pip check
python -m unittest discover -s tests -v
python .venv/tools/mistral_smoke.py
```

The model smoke check follows the author's FastChat load_8bit=True loader, checks inference, CacheGen encoding/decoding of real Mistral KV, reuse of decoded KV, and short generation. Its result is written to .venv/mistral-smoke.json. This is environment validation, not a full paper reproduction.

The custom CUDA extension is configured for T4 (sm_75) and RTX 4090D (sm_89). Repeat the checks after restoring the image.

## Save the image

Preserve /home/aaa/CacheGen including the hidden .venv and .cuda-12.1 directories on the system disk. This setup uses the platform system image and keeps absolute paths, so restore with the same paths. Model files are on the system disk rather than bayes-tmp. The platform console must perform image saving/restoration.

## Installation references

- https://pytorch.org/get-started/previous-versions/#v231
- https://docs.pytorch.org/docs/2.3/cpp_extension.html
- Original repository env.yaml and mini_repro/requirements-mini.txt

## WaveDrom build compatibility

WaveDrom 2.0.3.post3 has an unpinned setuptools_scm build dependency. A recent setuptools_scm requests packaging>=26.2, conflicting with the repository's packaging==24.1. The helper .venv/tools/build_wavedrom.sh builds the original, SHA-256-checked source in a new temporary environment with setuptools-scm==8.1.0, setuptools==69.5.1, wheel==0.43.0, and packaging==24.1. It installs the resulting wheel into the project environment without changing the runtime packaging version.

## Completed validation (2026-10-23, Tesla T4)

- pip check: no broken requirements.
- Repository tests: 41 passed, including the GPU codec roundtrip.
- Mistral offline smoke: inference, real KV encoding/decoding, decoded KV reuse, and generation passed.
- CUDA binary inspection: both sm_75 and sm_89 confirmed.
- Runtime packaging: 24.1; WaveDrom: 2.0.3.post3; protobuf: 5.27.2.

Records: environment-verification.json, mistral-download-verified.json, mistral-smoke.json, cuda-extension-architectures.txt, requirements-server.lock, requirements-server-versions.txt, and logs/ in this .venv directory. Environment activation defaults to offline model use. Repeat the model smoke check on the 4090D after restoring the image.
