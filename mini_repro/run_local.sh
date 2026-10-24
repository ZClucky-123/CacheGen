#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/experiment-env.sh"
cd -- "$repo_root"

experiment="${1:-mini}"
case "$experiment" in
  mini|quality|ttft|adaptive|reproduce|test) shift || true ;;
  -*) experiment="mini" ;;
  *) echo "Unknown experiment: $experiment" >&2; exit 2 ;;
esac
if [[ "$experiment" != test && " $* " != *' --dry-run '* && " $* " != *' --help '* ]]; then
  cachegen_lock_gpu
fi
case "$experiment" in
  mini) module="mini_repro.run"; output="metrics.json" ;;
  quality) module="mini_repro.quality_size"; output="quality_size.json" ;;
  ttft) module="mini_repro.ttft"; output="ttft.json" ;;
  adaptive) module="mini_repro.adaptive"; output="adaptive.json" ;;
  reproduce) exec "$CACHEGEN_PYTHON" -u -m mini_repro.reproduce --output-root "$CACHEGEN_RESULTS_DIR/reproduction" "$@" ;;
  test) exec "$CACHEGEN_PYTHON" -m unittest discover -s "$repo_root/tests" -v "$@" ;;
esac
exec "$CACHEGEN_PYTHON" -m "$module" --results "$CACHEGEN_RESULTS_DIR/local4070/$output" "$@"
