#!/bin/bash
# NSReg-on-arxiv is GPU-bound at ~1 core a job; mag is CPU-bound at ~4.9. While
# ten NSReg jobs run they hold ~10 cores and slow mag. When NSReg finishes, hand
# that CPU to mag. Waits on the job-accurate count, never on a process pattern.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
[ "$(hostname)" = "$(cat runs/.chain_host 2>/dev/null)" ] || exit 0
while :; do
  st=$(python3 analysis/scripts/campaign_remaining.py nsreg_arxiv 2>/dev/null | awk '{print $2}')
  [ -n "$st" ] && [ "${st%/*}" = "${st#*/}" ] && break
  sleep 300
done
echo "[$(date -u +%H:%M)] NSReg-arxiv complete ($st); adding mag drivers" >> runs/chain_sel.log
# two more mag drivers on the cards whose memory allows an 11.8 GB job
for g in 3 5; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g | tr -d ' ')
  if [ "${free:-0}" -gt 16000 ]; then
    nohup python3 analysis/scripts/run_small.py mag_scores --gpu $g \
        --headroom-gb 3 --threads 6 >> "runs/mag_boost_g$g.out" 2>&1 &
    echo "[$(date -u +%H:%M)]   + mag driver on gpu$g (${free} MiB free)" >> runs/chain_sel.log
    sleep 3
  fi
done
