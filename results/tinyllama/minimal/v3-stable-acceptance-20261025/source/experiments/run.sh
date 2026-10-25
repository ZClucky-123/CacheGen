#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$root/scripts/experiment-env.sh"
cd -- "$root"
exec "$CACHEGEN_PYTHON" -u -m experiments.suite "$@"
