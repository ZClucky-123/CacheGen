#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$root/../../scripts/experiment-env.sh"
config="$root/config.json"
dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) [[ $# -ge 2 ]] || { echo '--config requires a file' >&2; exit 2; }; config="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) echo 'Usage: bash experiments/mistral/run.sh [--config FILE] [--dry-run]'; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -f "$config" ]] || { printf 'Configuration missing: %s\n' "$config" >&2; exit 2; }
config="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$config")"
if [[ "$dry_run" -eq 1 ]]; then
  exec python3 -m experiments run mistral --dry-run --config "$config"
fi
if [[ "${CACHEGEN_SUPERVISOR_PID:-}" == "$PPID" ]]; then
  exec bash "$root/measure.sh" "$config"
fi
exec python3 -m experiments.process -- bash "$root/measure.sh" "$config"
