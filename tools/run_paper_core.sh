#!/usr/bin/env bash
# Server only. Stop on failure; never rerun or relax the gate automatically.
set -euo pipefail
core_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$core_root"
core_stage="${1:?Usage: bash tools/run_paper_core.sh stage1|stage2 OUTPUT_ROOT}"
core_run="${2:?Usage: bash tools/run_paper_core.sh stage1|stage2 OUTPUT_ROOT}"
source scripts/experiment-env.sh
case "$core_stage" in
stage1)
if [[ -e "$core_run" ]]; then
  printf '%s\n' "Output root already exists: $core_run" >&2
  exit 2
fi
mkdir -p -- "$core_run"
(
  source scripts/experiment-env.sh
  cachegen_lock_gpu
  "$CACHEGEN_PYTHON" -m unittest discover -s tests -p test_paper_compression.py -v
  "$CACHEGEN_PYTHON" -m unittest discover -s tests -p test_paper_study.py -v
) 2>&1 | tee "$core_run/tests.log"
bash tools/run_paper_study.sh insights --output "$core_run/insights" 2>&1 | tee "$core_run/insights.log"
bash tools/run_paper_study.sh calibrate --output "$core_run/profiles" 2>&1 | tee "$core_run/calibrate.log"
bash tools/run_paper_study.sh scan --profiles "$core_run/profiles" --output "$core_run/validation" 2>&1 | tee "$core_run/scan.log"
bash tools/run_paper_study.sh plot --input "$core_run/validation" --output "$core_run/validation-plots"
bash tools/run_paper_study.sh final --profiles "$core_run/profiles" --selection "$core_run/validation/selected.json" --output "$core_run/test" 2>&1 | tee "$core_run/final.log"
bash tools/run_paper_study.sh plot --input "$core_run/test" --output "$core_run/test-plots"
"$CACHEGEN_PYTHON" -m tools.paper_stages seal "$core_run"
;;
stage2)
"$CACHEGEN_PYTHON" -m tools.paper_stages verify "$core_run"
if [[ -e "$core_run/ttft" || -e "$core_run/adaptive" || -e "$core_run/ttft.log" || -e "$core_run/adaptive.log" ]]; then
  printf '%s\n' 'Stage2 output/logs already exist; refusing to overwrite.' >&2
  exit 2
fi
bash tools/run_paper_study.sh ttft --profiles "$core_run/profiles" --selection "$core_run/validation/selected.json" --output "$core_run/ttft" 2>&1 | tee "$core_run/ttft.log"
bash tools/run_paper_study.sh adaptive --profiles "$core_run/profiles" --selection "$core_run/validation/selected.json" --output "$core_run/adaptive" 2>&1 | tee "$core_run/adaptive.log"
;;
*)
  printf '%s\n' 'Usage: bash tools/run_paper_core.sh stage1|stage2 OUTPUT_ROOT' >&2
  exit 2
;;
esac
