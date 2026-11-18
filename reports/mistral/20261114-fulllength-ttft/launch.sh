#!/usr/bin/env bash
cd /home/aaa/CacheGen; source scripts/experiment-env.sh
exec 9>.local/gpu.lock; flock -n 9 || { echo GPU_LOCKED; exit 75; }
python -m tools.full_length_ttft --output .local/ttft-fulllength-20261114/$1 --ids ${@:2}
echo "EXIT=$?"
