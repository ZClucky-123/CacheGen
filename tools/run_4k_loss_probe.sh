#!/usr/bin/env bash
set -euo pipefail
probe_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$probe_root/scripts/experiment-env.sh"
cd -- "$probe_root"
probe_preview=false
for probe_arg in "$@"; do
  case "$probe_arg" in --dry-run|--help|-h) probe_preview=true ;; esac
done
if [[ "$probe_preview" == false ]]; then
  cachegen_lock_gpu
fi
exec "$CACHEGEN_PYTHON" -u -m tools.diagnose_4k_loss "$@"
