#!/usr/bin/env bash
# Source from experiment entrypoints. Reading this file creates no directories.
repo_root="${CACHEGEN_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
repo_root="$(cd -- "$repo_root" && pwd -P)"

cachegen_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$repo_root" "$1" ;;
  esac
}

export CACHEGEN_REPO_ROOT="$repo_root"
export CACHEGEN_RESULTS_DIR="$(cachegen_path "${CACHEGEN_RESULTS_DIR:-results}")"
export CACHEGEN_REPORTS_DIR="$(cachegen_path "${CACHEGEN_REPORTS_DIR:-reports}")"
export CACHEGEN_LOCAL_DIR="$(cachegen_path "${CACHEGEN_LOCAL_DIR:-.local}")"
if [[ -z "${CACHEGEN_PYTHON:-}" ]]; then
  if [[ -x "$repo_root/.venv/bin/python" ]]; then
    export CACHEGEN_PYTHON="$repo_root/.venv/bin/python"
  else
    export CACHEGEN_PYTHON="$(command -v python3)"
  fi
fi
if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -x "$repo_root/.cuda-12.1/bin/nvcc" ]]; then
    export CUDA_HOME="$repo_root/.cuda-12.1"
  elif [[ -x "$repo_root/.venv/cuda-12.1/bin/nvcc" ]]; then
    export CUDA_HOME="$repo_root/.venv/cuda-12.1"
  else
    export CUDA_HOME=/usr/local/cuda-12.1
  fi
fi
export PATH="$(dirname -- "$CACHEGEN_PYTHON"):$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$repo_root:$repo_root/LMCache${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$repo_root/.venv/huggingface}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CACHEGEN_LOCAL_DIR/torch_extensions}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}" MAX_JOBS="${MAX_JOBS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
if [[ -d "$CUDA_HOME/include/python3.10" ]]; then
  export CPLUS_INCLUDE_PATH="$CUDA_HOME/include/python3.10${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
fi

cachegen_lock_gpu() {
  mkdir -p -- "$CACHEGEN_LOCAL_DIR"
  exec 9>"$CACHEGEN_LOCAL_DIR/gpu.lock"
  if ! flock -n 9; then
    printf '%s\n' 'Another CacheGen GPU measurement is running; try again after it finishes.' >&2
    return 75
  fi
}
