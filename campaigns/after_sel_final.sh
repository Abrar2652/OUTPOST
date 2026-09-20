#!/bin/bash
# When sel_final finishes its 38 shards its drivers exit and four cards free up.
# The corrected 400-epoch DEMO grid (63 jobs) is the long pole after that, so put
# the freed capacity there. Waits on shard COUNT, never on a process pattern:
# a pattern that appears in the waiting shell's own argv matches itself.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
[ "$(hostname)" = "$(cat runs/.chain_host 2>/dev/null)" ] || exit 0

count() {
  ls results/rotations/amazon_outpost_s{2,3,4,5,6,7,8}_T_hidden16_rot1.json \
     results/rotations/amazon_outpost_s{0,1,2,3,4,5,6,7,8,42}_C_nopl_hidden16_rot1.json \
     results/rotations/photo_nsreg_s{2,3,4,5,6,7,8}_NT_lr0.003_wd0.0_rot{0,7}.json \
     results/rotations/amazon_nsreg_s{2,3,4,5,6,7,8}_NT_lr0.003_wd0.0_rot1.json \
     2>/dev/null | wc -l
}
while [ "$(count)" -lt 38 ]; do sleep 120; done
echo "[$(date -u +%H:%M)] sel_final complete; adding demo_tune400 drivers" >> runs/chain_sel.log
for g in 5 6; do
  nohup python3 analysis/scripts/run_small.py demo_tune400 --gpu $g \
      --headroom-gb 5 --threads 4 > "runs/demo_tune400_g$g.out" 2>&1 &
  sleep 3
done
echo "[$(date -u +%H:%M)] launched demo_tune400 on gpus 5,6" >> runs/chain_sel.log
