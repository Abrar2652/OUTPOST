#!/bin/bash
# Everything that remains, in value order, each followed by a full refresh and
# a LaTeX-table diff. Waits for the running gate-drop campaign to finish first.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
export PYTHONPATH=.stubs
snap() { mkdir -p runs/tex_snap; cp analysis/tables/*.tex analysis/tables/tex/*.tex runs/tex_snap/ 2>/dev/null; }
refresh() {
  echo "=== refresh after $1 at $(date) ===" >> runs/tex_diff.log
  bash analysis/scripts/refresh_all.sh > runs/refresh_$1.log 2>&1
  for f in analysis/tables/*.tex analysis/tables/tex/*.tex; do
    b=$(basename $f); [ -f runs/tex_snap/$b ] && { d=$(diff runs/tex_snap/$b $f | grep -c '^[<>]'); [ "$d" -gt 0 ] && echo "  $b: $d changed lines" >> runs/tex_diff.log; }
  done
  grep -E "numeric claims|STALE" runs/refresh_$1.log | head -2 >> runs/tex_diff.log
  snap
}
run() { python3 -u analysis/scripts/run_campaign.py campaigns/$1.json --gpus 3 4 5 6 7 --max-parallel ${2:-15} --gpu-gb 45; python3 analysis/scripts/merge_rotations.py; }
echo "waiting for gate-drop to finish..." ; while ps -eo args | grep -q '[g]atedrop.json'; do sleep 300; done
sleep 60; rm -f runs/.gpu*.lock; snap; refresh gatedrop
run decomp_fill; run arxiv_n10; refresh cheap
[ "$(hostname)" = "$(cat runs/.chain_host 2>/dev/null)" ] || { echo "chain_all: not the designated host ($(hostname)); exiting" >> runs/tex_diff.log; exit 0; }
run hparam400; refresh hparam400
run nsreg_tune; refresh nsreg_tune
if [ -f data/tfinance/tfinance.zip ] && [ -f campaigns/tfinance.json ]; then run tfinance; refresh tfinance; fi
run mag_scores 15; refresh mag_scores
echo "ALL CAMPAIGNS DONE $(date)" >> runs/tex_diff.log
