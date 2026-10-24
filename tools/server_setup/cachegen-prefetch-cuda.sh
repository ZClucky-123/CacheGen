#!/usr/bin/env bash
set -euo pipefail
setup_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$setup_dir/../../scripts/experiment-env.sh"
export CACHEGEN_WHEELHOUSE="$repo_root/.venv/wheelhouse"
mkdir -p -- "$CACHEGEN_WHEELHOUSE"
cat "$setup_dir/.cachegen-cuda-wheels.txt" | xargs -P 4 -n 1 bash -c 'for attempt in 1 2 3; do "$CACHEGEN_PYTHON" -m pip download --index-url https://pypi.org/simple --no-deps --dest "$CACHEGEN_WHEELHOUSE" "$1" && exit 0; done; exit 1' _
printf 'CUDA_WHEELS_PREFETCHED\n'
