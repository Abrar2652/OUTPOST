#!/bin/bash
# The OOM-failed C_noconformal seed 42 released its claim, but both drivers had
# already passed that position in their one-pass queues, so nothing will retry it
# and the arm would finish at 9 of 10 seeds - enough to block P38/P39 scoring.
#
# Waits on SHARD COUNT, never on a process pattern: this script's own argv
# contains "tfinance_decomp", so any pgrep for that would match the waiter itself.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
[ "$(hostname)" = "$(cat runs/.chain_host 2>/dev/null)" ] || exit 0

count() { ls results/rotations/tfinance_outpost_s*_C_nogate_rot1.json \
             results/rotations/tfinance_outpost_s*_C_noconformal_rot1.json \
             2>/dev/null | wc -l; }

# 19 = everything the two running drivers will produce; the 20th is the orphan
while [ "$(count)" -lt 19 ]; do sleep 300; done
echo "[$(date -u +%H:%M)] sweeper: 19 shards present, running the orphan" >> runs/chain_sel.log
# a fresh driver re-reads the campaign and skips finished shards, so it picks up
# the orphan and anything else that slipped through
python3 analysis/scripts/run_small.py tfinance_decomp --gpu 4 \
    --headroom-gb 6 --threads 4 >> runs/tfinance_sweep.out 2>&1
echo "[$(date -u +%H:%M)] sweeper done: $(count)/20 shards" >> runs/chain_sel.log
