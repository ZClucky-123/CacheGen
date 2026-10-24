#!/usr/bin/env bash
# Internal measurement body. Public callers use run.sh for signal supervision.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$root/../../scripts/experiment-env.sh"
config="$1"
cachegen_lock_gpu
runs="$CACHEGEN_RESULTS_DIR/mistral"
mkdir -p -- "$runs"
run_id="$(python3 -c 'from datetime import datetime,timezone; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))')"
run_dir="$runs/$run_id"
mkdir -- "$run_dir"
finish() {
  code=$?
  trap - EXIT INT TERM
  set +e
  python3 -m experiments.lifecycle finish "$run_dir" "$code"
  exit $?
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -- "$run_dir/source"
cp -- "$root/"*.py "$root/"*.sh "$root/README.md" "$run_dir/source/"
cp -- "$config" "$run_dir/source/config.json"
python3 -m experiments.lifecycle initialize "$run_dir" "$run_dir/source/config.json"
ln -sfnT -- "$run_id" "$runs/latest"
printf '%s\n' "$run_dir" > "$runs/latest-run.txt"
printf 'Running: %s\n' "$run_dir"
"$CACHEGEN_PYTHON" -m unittest discover -s "$root" -p 'test_*.py' -v 2>&1 | tee "$run_dir/tests.log"
"$CACHEGEN_PYTHON" -u "$root/experiment.py" --run-dir "$run_dir" --config "$run_dir/source/config.json" 2>&1 | tee "$run_dir/run.log"
