#!/bin/bash
# P25 first (10 jobs, ~25 min), then resume gate-drop on the same cards.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
python3 -u analysis/scripts/run_campaign.py campaigns/amazon_p25.json --gpus 3 4 5 6 7 --max-parallel 10 --gpu-gb 45
python3 analysis/scripts/merge_rotations.py
sleep 30
exec python3 -u analysis/scripts/run_campaign.py campaigns/gatedrop.json --gpus 3 4 5 6 7 --max-parallel 12 --gpu-gb 45
