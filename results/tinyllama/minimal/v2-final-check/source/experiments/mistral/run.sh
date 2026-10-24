#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
for argument in "$@"; do
  if [[ "$argument" == --resume ]]; then
    exec bash "$root/../run.sh" "$@"
  fi
done
exec bash "$root/../run.sh" --model mistral "$@"
