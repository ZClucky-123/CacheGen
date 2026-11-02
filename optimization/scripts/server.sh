#!/usr/bin/env bash
# Explicit stages; no automatic server provisioning or background launch.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
stage="${1:?Usage: server.sh plan|build|gpu-check|calibrate|fit-residual|verify|freeze|benchmark RUN_ID}"
run_id="${2:?Provide a unique run ID}"
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo 'Invalid run ID' >&2; exit 2; }
run="$repo/optimization/output/$run_id"
export CACHEGEN_LOCAL_DIR="$run/local"
export TORCH_EXTENSIONS_DIR="$run/build/torch_extensions"
source "$repo/scripts/experiment-env.sh"
cd -- "$repo"
export PYTHONPATH="$run/build/lib:$PYTHONPATH"
command=("$CACHEGEN_PYTHON" -m experiments.joint --base-config experiments/configs/larger-mistral.json
         --config optimization/configs/joint-v2.json --run-dir "optimization/output/$run_id")
if [[ "$stage" == plan ]]; then
  "${command[@]}" --task inspect --plan
  exit
fi
mkdir -p -- "$run/logs" "$run/build/tmp" "$run/tmp"
export TMPDIR="$run/tmp"
# Use the same coordination inode as the original framework. Experiment
# logs/builds remain in the run directory; this shared lock is coordination.
mkdir -p -- "$repo/.local"
exec 9>"$repo/.local/gpu.lock"
flock -n 9 || { echo 'Another CacheGen GPU task holds the shared lock' >&2; exit 75; }
case "$stage" in
  build)
    "$CACHEGEN_PYTHON" -c 'from experiments.joint_calibration import require_gpu; require_gpu()'
    if ! "$CACHEGEN_PYTHON" -c 'import torchac_cuda'; then
      (
        cd -- "$repo/LMCache/third_party/torchac_cuda"
        "$CACHEGEN_PYTHON" setup.py build_ext --build-lib "$run/build/lib" --build-temp "$run/build/torchac"
      ) 2>&1 | tee "$run/logs/torchac-build.log"
    fi
    "$CACHEGEN_PYTHON" LMCache/third_party/cachegen_optim_cuda/setup.py build_ext \
      --build-lib "$run/build/lib" --build-temp "$run/build/tmp" 2>&1 | tee "$run/logs/build.log"
    ;;
  gpu-check)
    "$CACHEGEN_PYTHON" -c 'from experiments.joint_calibration import require_gpu; require_gpu()'
    "$CACHEGEN_PYTHON" - "$run" 2>&1 <<'PY' | tee "$run/logs/gpu-correctness.log"
import sys, unittest
from pathlib import Path
from experiments.joint_config import load_joint_config
from experiments.config import load_config
from experiments.joint_calibration import code_fingerprint, save_json
suite = unittest.defaultTestLoader.discover('tests', pattern='test_joint_*gpu.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful() or result.testsRun == 0 or result.skipped:
    raise SystemExit('GPU correctness requires actual executed passing tests with zero skips')
fingerprint, environment = code_fingerprint(load_config('experiments/configs/larger-mistral.json'), load_joint_config(), device=True)
save_json(Path(sys.argv[1]) / 'gpu-correctness.json', {'passed': True, 'fingerprint': fingerprint, 'environment': environment, 'tests':result.testsRun,'skipped':0,'test_log': 'logs/gpu-correctness.log'})
PY
    ;;
  calibrate) "${command[@]}" --task calibrate 2>&1 | tee "$run/logs/calibrate.log" ;;
  fit-residual) "${command[@]}" --task check --phase fit-residual 2>&1 | tee "$run/logs/fit-residual.log" ;;
  verify) "${command[@]}" --task check --phase verify 2>&1 | tee "$run/logs/verify.log" ;;
  freeze) "${command[@]}" --task inspect --freeze 2>&1 | tee "$run/logs/freeze.log" ;;
  benchmark) "${command[@]}" --task benchmark 2>&1 | tee "$run/logs/benchmark.log" ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
