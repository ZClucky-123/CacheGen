#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="$repo_root/.venv"
export CUDA_HOME="$venv/cuda-12.1"
export PATH="$venv/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$repo_root:$repo_root/LMCache${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$venv/huggingface"
export TORCH_EXTENSIONS_DIR="$venv/torch_extensions"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=2

if [[ ! -x "$venv/bin/python" || ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "Missing local Python environment or CUDA compiler under $venv" >&2
  exit 1
fi

experiment="${1:-mini}"
case "$experiment" in
  mini|quality|ttft|adaptive|test) shift || true ;;
  -*) experiment="mini" ;;
  *) echo "Unknown experiment: $experiment" >&2; exit 2 ;;
esac
case "$experiment" in
  mini) module="mini_repro.run"; output="metrics.json" ;;
  quality) module="mini_repro.quality_size"; output="quality_size.json" ;;
  ttft) module="mini_repro.ttft"; output="ttft.json" ;;
  adaptive) module="mini_repro.adaptive"; output="adaptive.json" ;;
  test) exec "$venv/bin/python" -m unittest discover -s "$repo_root/tests" -v "$@" ;;
esac
exec "$venv/bin/python" -m "$module" --results "$repo_root/results/local4070/$output" "$@"
