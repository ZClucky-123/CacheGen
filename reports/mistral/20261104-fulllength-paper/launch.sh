#!/usr/bin/env bash
cd /home/aaa/CacheGen; source scripts/experiment-env.sh
exec 9>.local/gpu.lock; flock -n 9 || { echo GPU_LOCKED; exit 75; }
python -m tools.full_length_paper --output .local/paper-fulllength-paper-20261104/full --calibration-ids $(seq 30 49) --test-ids $(seq 0 19)
echo "EXIT=$?"
