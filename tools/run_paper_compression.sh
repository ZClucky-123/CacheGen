#!/usr/bin/env bash
set -euo pipefail
paper_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$paper_root/scripts/experiment-env.sh"
cd -- "$paper_root"
if [[ "${1:-}" != plan && "${1:-}" != --help && "${1:-}" != -h ]]; then
  cachegen_lock_gpu
fi
exec "$CACHEGEN_PYTHON" -u -m tools.paper_compression "$@"
