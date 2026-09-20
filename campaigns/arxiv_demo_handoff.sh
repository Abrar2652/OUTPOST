#!/bin/bash
# The four pre-fix arxiv DEMO drivers loaded a spec without the epoch override and
# run at 200. They were stood down by claiming every remaining job for pid 1, so
# they will exit once their in-flight jobs finish. This waits for that, clears the
# stand-down claims, and restarts against the corrected 400-epoch spec.
#
# Waits on driver "done:" lines, never on a process pattern: this script's own argv
# contains the campaign name and would match any pgrep for it.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
[ "$(hostname)" = "$(cat runs/.chain_host 2>/dev/null)" ] || exit 0
LOG=runs/arxiv_demo/driver.log

started() { grep -ac 'on gpu' "$LOG" 2>/dev/null || echo 0; }
finished() { grep -ac 'done:' "$LOG" 2>/dev/null || echo 0; }

while :; do
  s=$(started); f=$(finished)
  [ "$s" -gt 0 ] && [ "$f" -ge "$s" ] && break
  sleep 120
done
echo "[$(date -u +%H:%M)] handoff: all $(started) pre-fix drivers exited" >> runs/chain_sel.log

# their output is half-budget; sweep it before anything can merge it
python3 analysis/scripts/quarantine_wrong_budget.py ogbn-arxiv demo 400 0 >> runs/chain_sel.log 2>&1
grep -l 'stand-down' runs/arxiv_demo/claims/*.claim 2>/dev/null | xargs -r rm -f
echo "[$(date -u +%H:%M)] handoff: stand-down claims cleared" >> runs/chain_sel.log

for g in 3 7; do
  nohup python3 analysis/scripts/run_small.py arxiv_demo --gpu $g \
      --headroom-gb 5 --threads 6 >> "runs/arxiv_demo_fixed_g$g.out" 2>&1 &
  sleep 3
done
echo "[$(date -u +%H:%M)] handoff: relaunched arxiv_demo at 400 epochs" >> runs/chain_sel.log
