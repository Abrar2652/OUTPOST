#!/bin/bash
# Restart the GPU-6 (cs) queue. Same shape as restart_gpu7.sh: stop the runner,
# stop its orphaned children, relaunch. Safe to repeat - finished rotations are
# skipped by their shard file.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
pkill -f "run_campaign.py campaigns/gpu6_cs.json"
sleep 2
ps -eo pid,args | grep "main.py --dataset cs" | grep -v grep \
  | awk '{print $1}' | xargs -r kill -9
sleep 4
python3 campaigns/build_gpu6_cs.py
nohup python3 -u analysis/scripts/run_campaign.py campaigns/gpu6_cs.json \
  --gpus 6 --max-parallel 1 --gpu-gb 22 > runs/gpu6_cs.log 2>&1 &
echo "gpu6 runner pid $!"
exit 0
