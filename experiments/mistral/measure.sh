#!/usr/bin/env bash
# Compatibility for the old internal wrapper; current lifecycle lives in suite.py.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ $# -eq 1 && "$1" != --* ]]; then
  exec bash "$root/run.sh" --config "$1"
fi
exec bash "$root/run.sh" "$@"
