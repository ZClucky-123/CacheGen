#!/usr/bin/env bash
cd /home/aaa/CacheGen; source scripts/experiment-env.sh
exec 9>.local/gpu.lock; flock -n 9 || { echo GPU_LOCKED; exit 75; }
python -m tools.full_length_paper --output .local/paper-fulllength-paper-20261114/pilot --calibration-ids 30 31 --test-ids 0 --configs paper-r4 nolayer-r4 nodelta-r4 paper-abs nolayer-abs nodelta-abs
echo "EXIT=$?"
