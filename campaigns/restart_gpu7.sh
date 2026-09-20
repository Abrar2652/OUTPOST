#!/bin/bash
# Restart the GPU-7 queue: stop the runner, stop its orphaned children (killing
# the runner leaves them running), rebuild the job list, relaunch. Safe to
# repeat - finished runs are skipped on resume.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
pkill -f "run_campaign.py campaigns/gpu7.json"
sleep 2
# Kill only THIS campaign's orphans - children survive the runner being killed.
# Selecting them by dataset name was wrong: it also killed jobs the GPU-6 runner
# had just started, since both queues train the same datasets. The log directory
# is the reliable discriminator, since run_job writes one log per job under it.
for pid in $(pgrep -f "main.py --dataset"); do
  if tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -q '^CUDA_VISIBLE_DEVICES=7$'; then
    kill -9 "$pid"
  fi
done
sleep 4
python3 campaigns/build_gpu7.py
nohup python3 -u analysis/scripts/run_campaign.py campaigns/gpu7.json \
  --gpus 7 --max-parallel 3 > runs/gpu7.log 2>&1 &
echo "gpu7 runner pid $!"
exit 0
