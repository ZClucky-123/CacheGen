#!/usr/bin/env bash
cd /home/aaa/CacheGen; source scripts/experiment-env.sh
exec 9>.local/gpu.lock; flock -n 9 || { echo GPU_LOCKED; exit 75; }
python tools/full_length_qa.py --output .local/paper-fulllength-qa-20261102/full --ids $(seq 0 49)
echo "EXIT=$?" 
