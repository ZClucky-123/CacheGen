#!/usr/bin/env bash
set -euo pipefail
setup_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$setup_dir/../../scripts/experiment-env.sh"
cd -- "$repo_root"
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'
bash "$repo_root/.venv/tools/build_wavedrom.sh"
python -m pip install --index-url https://pypi.org/simple -c "$setup_dir/.cachegen-original-constraints.txt" -c mini_repro/requirements-mini.txt fschat==0.2.36 openai==0.28.0 accelerate==0.32.1 bitsandbytes==0.43.3 peft==0.11.1 einops==0.8.0 protobuf==5.27.2
python -m pip install --no-build-isolation --no-deps ./LMCache
python -m pip install --no-build-isolation --no-deps ./LMCache/third_party/torchac_cuda
python -m pip check
python -m unittest discover -s tests -v
python -m pip freeze > .venv/requirements-server.lock
python -c 'import torch,torchac_cuda; from src.utils import define_model_and_tokenizer; print("GPU:",torch.cuda.get_device_name(0)); print("CUDA extensions and original model loader imported successfully")'
printf 'CACHEGEN_ENVIRONMENT_READY\n'
