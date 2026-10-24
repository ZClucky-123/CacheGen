#!/usr/bin/env bash
set -euo pipefail
setup_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$setup_dir/../../scripts/experiment-env.sh"
export CACHEGEN_WHEELHOUSE="$repo_root/.venv/wheelhouse"
mkdir -p -- "$CACHEGEN_WHEELHOUSE"
cd -- "$repo_root"
export PIP_CONSTRAINT="$setup_dir/.cachegen-original-constraints.txt"
awk '/^nvidia-/ {print}' "$PIP_CONSTRAINT" | xargs -P 4 -n 1 bash -c 'env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy python -m pip download --index-url https://pypi.nvidia.com --find-links "$CACHEGEN_WHEELHOUSE" --no-deps --dest "$CACHEGEN_WHEELHOUSE" "$1" || python -m pip download --index-url https://pypi.org/simple --no-deps --dest "$CACHEGEN_WHEELHOUSE" "$1"' _
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy python -m pip download --index-url https://download.pytorch.org/whl/cu121 --no-deps --dest .venv/wheelhouse triton==2.3.1
python -m pip install --no-index --no-deps --find-links .venv/wheelhouse -r "$setup_dir/.cachegen-cuda-wheels.txt"
python -m pip install --index-url https://pypi.org/simple --find-links .venv/wheelhouse .venv/wheelhouse/torch-2.3.1+cu121-cp310-cp310-linux_x86_64.whl
python -m pip install --index-url https://pypi.org/simple --find-links .venv/wheelhouse -r mini_repro/requirements-mini.txt huggingface-hub==0.23.4
bash "$setup_dir/cachegen-finish-setup.sh"
