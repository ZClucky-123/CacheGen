#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="$HOME/.venvs/cachegen-mini"
export PATH="$venv/bin:/usr/local/cuda-12.1/bin:$PATH"
export PYTHONPATH="$repo_root:$repo_root/LMCache${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$HOME/.cache/huggingface"
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_HOME=/usr/local/cuda-12.1
export MAX_JOBS=2

exec "$venv/bin/python" -m mini_repro.run "$@"
