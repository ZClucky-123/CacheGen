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

experiment="${1:-mini}"
case "$experiment" in
  mini|quality|ttft|adaptive) shift || true ;;
  -*) experiment="mini" ;;
  *) echo "Unknown experiment: $experiment" >&2; exit 2 ;;
esac
case "$experiment" in
  mini) module="mini_repro.run" ;;
  quality) module="mini_repro.quality_size" ;;
  ttft) module="mini_repro.ttft" ;;
  adaptive) module="mini_repro.adaptive" ;;
esac
exec "$venv/bin/python" -m "$module" "$@"
