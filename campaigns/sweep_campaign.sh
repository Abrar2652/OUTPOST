#!/bin/bash
# Re-run whatever a campaign's drivers left behind.
#
# run_small.py drivers walk their queue ONCE. A job that fails releases its claim
# so another driver can retry it, but only a driver that has not yet passed that
# position will. Anything failing behind the queue head is simply orphaned - three
# Yelp DEMO jobs and one T-Finance job on 2026-09-08.
#
# Waits until every driver for the campaign has logged its "done:" line, then runs
# one more driver: it re-reads the campaign, skips finished shards, and claims only
# what is missing. Claims make this safe to start even if a driver is still going.
#
#   bash campaigns/sweep_campaign.sh <campaign> [gpu]
#
# Counts log lines rather than matching processes: this script's own argv contains
# the campaign name, so any pgrep for it would match the waiter itself.
set -u
C="${1:?usage: sweep_campaign.sh <campaign> [gpu]}"
GPU="${2:-4}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
[ "$(hostname)" = "$(cat runs/.chain_host 2>/dev/null)" ] && : || exit 0
LOG="runs/$C/driver.log"

started() { grep -ac 'on gpu' "$LOG" 2>/dev/null || echo 0; }
finished() { grep -ac 'done:' "$LOG" 2>/dev/null || echo 0; }

while :; do
  s=$(started); f=$(finished)
  [ "$s" -gt 0 ] && [ "$f" -ge "$s" ] && break
  sleep 300
done
echo "[$(date -u +%H:%M)] sweep $C: all $(started) driver(s) finished; sweeping" >> runs/chain_sel.log
python3 analysis/scripts/run_small.py "$C" --gpu "$GPU" --headroom-gb 6 --threads 4 \
  >> "runs/sweep_$C.out" 2>&1
echo "[$(date -u +%H:%M)] sweep $C: complete" >> runs/chain_sel.log
