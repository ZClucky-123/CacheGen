# Source this file before running CacheGen on this server.
source /home/aaa/CacheGen/.venv/bin/activate
export CUDA_HOME=/home/aaa/CacheGen/.cuda-12.1
export PATH="/home/aaa/CacheGen/.venv/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="/home/aaa/CacheGen:/home/aaa/CacheGen/LMCache${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/home/aaa/CacheGen/.venv/huggingface
export TORCH_EXTENSIONS_DIR=/home/aaa/CacheGen/.venv/torch_extensions
export TORCH_CUDA_ARCH_LIST="7.5;8.9"
export MAX_JOBS=2

export CPLUS_INCLUDE_PATH="$CUDA_HOME/include/python3.10${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
