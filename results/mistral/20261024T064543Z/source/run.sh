#!/usr/bin/env bash
set -uo pipefail
root=/root/cachegen-repro
repo=/home/aaa/CacheGen
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="$root/runs/$run_id"
mkdir -p "$run_dir"
ln -sfn "$run_dir" "$root/latest"
printf '%s\n' "$run_dir" > "$root/latest-run.txt"
export CUDA_HOME="$repo/.cuda-12.1"
export PATH="$repo/.venv/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$root:$repo:$repo/LMCache${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$repo/.venv/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_EXTENSIONS_DIR="$root/torch_extensions"
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 OMP_NUM_THREADS=8
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include/python3.10${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
printf '%s\n' "Running: $run_dir"
"$repo/.venv/bin/python" -m unittest discover -s "$root" -p 'test_*.py' -v 2>&1 | tee "$run_dir/tests.log"
code=${PIPESTATUS[0]}
if [ "$code" -eq 0 ]; then
  "$repo/.venv/bin/python" -u "$root/experiment.py" --run-dir "$run_dir" 2>&1 | tee "$run_dir/run.log"
  code=${PIPESTATUS[0]}
fi
printf '%s\n' "$code" > "$run_dir/exit_code"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/process_finished_utc"
printf '%s\n' "Experiment exit code: $code; results: $run_dir"
exit "$code"
