#!/bin/bash
# NSReg first, gate-drop after. Stop gate-drop's RUNNER now so nothing new is
# dispatched, but let its in-flight workers finish - they are minutes from
# writing their shards and killing them discards ~3 GPU-hours. Then stop the
# bare timing probes (their jobs are in the campaign and will be re-run under
# the scheduler), run NSReg on all five cards, merge, and resume gate-drop,
# which skips every rotation that already has a shard.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
pkill -f "campaigns/chain_gatedrop.sh"; pkill -f "run_campaign.py campaigns/gatedrop.json"; sleep 3
echo "waiting for in-flight gate-drop workers to drain..."
while pgrep -f "main.py --dataset" > /dev/null; do sleep 30; done
echo "drained at $(date)"
pkill -9 -f "baselines/run_nsreg.py"; sleep 5
rm -f runs/.gpu*.lock
python3 -u analysis/scripts/run_campaign.py campaigns/nsreg.json --gpus 3 4 5 6 7 --max-parallel 15 --gpu-gb 45
python3 analysis/scripts/merge_rotations.py
sleep 30
exec python3 -u analysis/scripts/run_campaign.py campaigns/gatedrop.json --gpus 3 4 5 6 7 --max-parallel 12 --gpu-gb 45
