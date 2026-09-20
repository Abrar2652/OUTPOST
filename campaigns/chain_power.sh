#!/bin/bash
# Start the power top-up as soon as the ogbn-mag runner releases GPUs 3-7.
# Chained rather than launched now because run_campaign holds a per-GPU lockfile
# and correctly refuses a second runner on a held card - so a manual launch would
# either be rejected or, without the lock, over-subscribe the cards.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
while pgrep -f "run_campaign.py campaigns/mag_seeds.json" > /dev/null; do sleep 300; done
sleep 60                      # let the last workers tear down and free memory
python3 analysis/scripts/merge_rotations.py >> runs/mag_seeds.log 2>&1
exec python3 -u analysis/scripts/run_campaign.py campaigns/power_seeds.json \
     --gpus 3 4 5 6 7 --max-parallel 12 --gpu-gb 45
