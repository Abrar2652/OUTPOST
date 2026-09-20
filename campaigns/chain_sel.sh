#!/bin/bash
# Runs after the selection re-runs land: refresh, then the T-Finance decomposition.
#
# Waits on MARKER FILES, never on a pgrep pattern: a wait loop whose pattern
# matches its own argv holds itself open forever, which cost hours once already.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
[ "$(hostname)" = "$(cat runs/.chain_host 2>/dev/null)" ] || {
  echo "chain_sel: not the designated host ($(hostname)); exiting" >> runs/tex_diff.log; exit 0; }

log() { echo "[$(date -u +%H:%M)] chain_sel: $*" >> runs/chain_sel.log; }

refresh() {
  mkdir -p runs/tex_snap
  cp -f analysis/tables/tex/*.tex analysis/tables/*.tex runs/tex_snap/ 2>/dev/null
  bash analysis/scripts/refresh_all.sh > "runs/refresh_$1.log" 2>&1
  echo "=== refresh after $1 at $(date) ===" >> runs/tex_diff.log
  for f in analysis/tables/tex/*.tex analysis/tables/*.tex; do
    b=$(basename "$f"); [ -f "runs/tex_snap/$b" ] || continue
    n=$(diff "runs/tex_snap/$b" "$f" 2>/dev/null | grep -c '^[<>]')
    [ "$n" -gt 0 ] && echo "  $b: $n changed lines" >> runs/tex_diff.log
  done
  grep -a -A3 'numeric claims match' "runs/refresh_$1.log" >> runs/tex_diff.log 2>/dev/null
}

# 1. wait for every sel_final shard to exist. Counting result files is the only
#    honest completion test: driver logs appear when a job STARTS, and a claim
#    file says a job was taken, not that it finished.
log "waiting for sel_final (38 shards)"
count_shards() {
  ls results/rotations/amazon_outpost_s{2,3,4,5,6,7,8}_T_hidden16_rot1.json \
     results/rotations/amazon_outpost_s{0,1,2,3,4,5,6,7,8,42}_C_nopl_hidden16_rot1.json \
     results/rotations/photo_nsreg_s{2,3,4,5,6,7,8}_NT_lr0.003_wd0.0_rot{0,7}.json \
     results/rotations/amazon_nsreg_s{2,3,4,5,6,7,8}_NT_lr0.003_wd0.0_rot1.json \
     2>/dev/null | wc -l
}
while [ "$(count_shards)" -lt 38 ]; do
  sleep 300
  [ -f runs/.chain_sel_stop ] && { log "stop file present; exiting"; exit 0; }
done
log "sel_final finished; refreshing"
refresh sel_final
# score the predictions this campaign was run to test; the scorer refuses a
# partial campaign, so a non-zero exit here means shards are still missing
python3 analysis/scripts/score_selection_final.py >> runs/score_selection_final.txt 2>&1
log "scored P32-P35, P40 -> runs/score_selection_final.txt (exit $?)"

# 2. the T-Finance decomposition, once the box is no longer saturated
log "starting tfinance_decomp"
python3 analysis/scripts/run_small.py tfinance_decomp --gpu 4 --headroom-gb 5 --threads 4 \
  >> runs/chain_sel.log 2>&1
refresh tfinance_decomp
log "ALL DONE"
