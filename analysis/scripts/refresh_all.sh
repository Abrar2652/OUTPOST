#!/bin/bash
# One refresh at a time. Two concurrent runs write the same tables and the same
# appendix.json, and a half-written file is worse than a stale one. flock is
# advisory and costs nothing when uncontended.
if [ -z "$REFRESH_LOCKED" ]; then
  export REFRESH_LOCKED=1
  # exec through bash, not the script path: an editor that rewrites this file
  # atomically (write temp + rename) drops its executable bit, and `flock ... "$0"`
  # then dies with "Permission denied" the moment it wins the lock.
  # runs/ is not in a fresh clone (it is gitignored), and flock will not create
  # the directory for its own lock file, so refresh_all died on line one there
  mkdir -p "$(dirname "$0")/../../runs"
  exec flock -w 3600 "$(dirname "$0")/../../runs/.refresh.lock" bash "$0" "$@"
fi
# Rebuild every analysis artifact from whatever has finished so far.
#
# Safe to run at any point during a campaign: each step reads the run records
# and reports what is present rather than assuming the campaign is complete.
# Nothing here trains, so it costs no GPU and can be run as often as wanted.
set -u
cd "$(dirname "$0")/../.." || exit 1

echo "=============================================================="
echo " 1. assemble sharded rotations into results rows"
echo "=============================================================="
# sweep out any shard whose budget does not match the protocol before merging;
# merge_rotations refuses mismatched main-table arms anyway, but leaving them in
# results/rotations/ means every later command has to keep stepping around them
python3 analysis/scripts/quarantine_wrong_budget.py ogbn-arxiv demo 400 0 2>/dev/null || true
# Every generator below is guarded. build_tables.py once raised AttributeError
# partway through (a renamed column its caller never followed), stopped writing
# table_compression, and this script still reached "done." -- an unguarded step
# that dies quietly is indistinguishable from one that succeeded.
die() { echo; echo "####################################################################"; echo "  # $1 FAILED -- its tables are now STALE. Fix and re-run.";  echo "####################################################################"; exit 1; }

python3 analysis/scripts/merge_rotations.py || die merge_rotations.py

echo
echo "=============================================================="
echo " 2. main tables, both selection protocols"
echo "=============================================================="
python3 analysis/scripts/make_paper_tables.py --metric best || die make_paper_tables.py
python3 analysis/scripts/make_paper_tables.py --metric valsel || die make_paper_tables.py
python3 analysis/scripts/make_paper_tables.py --metric best --selected-arm || die make_paper_tables.py
python3 analysis/scripts/make_paper_tables.py --metric valsel --selected-arm || die make_paper_tables.py
python3 analysis/scripts/make_paper_tables.py --metric best --gate-off || die make_paper_tables.py
python3 analysis/scripts/make_paper_tables.py --metric valsel --gate-off || die make_paper_tables.py

python3 analysis/scripts/build_tables.py || die build_tables.py        # main comparison, OUTPOST row from results.csv
python3 analysis/scripts/build_epoch_table.py || die build_epoch_table.py   # 200 vs 400 epochs, DEMO gap by budget

echo
echo "=============================================================="
echo " 3. statistics: cells, paired tests, ablations"
echo "=============================================================="
python3 analysis/scripts/stats.py --json analysis/tables/stats.json

echo
echo "=============================================================="
echo " 4. detectability law, measured cost, provenance, environment"
echo "=============================================================="
python3 analysis/scripts/detectability_law.py
python3 analysis/scripts/law_crossval.py > /dev/null 2>&1 || echo "  [law crossval] failed"
python3 analysis/scripts/detectability_law.py --holdout amazon \
  --json analysis/tables/detectability_law_holdout_amazon.json   # P23/P24, out of sample
python3 analysis/scripts/efficiency.py
python3 analysis/scripts/provenance.py
python3 analysis/scripts/record_environment.py > /dev/null && \
  echo "-> results/environment.json"

echo
echo "=============================================================="
echo " 5. figures"
echo "=============================================================="
python3 analysis/scripts/simsample_by_dataset.py   # regime figure data
echo "=============================================================="
# figures are built in step 6b by make_figures_paper.py; make_figures.py is
# now data helpers only and no longer draws (it wrote the same filenames)

echo
echo "=============================================================="
echo " 6. appendix: every table behind the claims, as JSON and as a page"
echo "=============================================================="
for ds in photo:12 computers:9; do n=${ds#*:}; ds=${ds%:*}; python3 analysis/scripts/select_hparams.py --dataset $ds --expect-arms $n --seeds 0 1 42 --prefix E400_T_ --default-tag E400_outpost --out-suffix _400 --reveal > /dev/null 2>&1 || echo "  [hparam400] $ds: sweep not complete yet"; done
python3 analysis/scripts/select_hparams.py --dataset amazon --expect-arms 12 --seeds 0 1 42 --default-tag $(python3 -c "import json;a=json.load(open('analysis/tables/arm_selection.json'))['amazon'];print(a['arm'] if isinstance(a,dict) else a)" 2>/dev/null || echo A_sim0.0) --reveal > /dev/null 2>&1 || echo "  [hparam amazon] sweep not complete yet"
python3 analysis/scripts/sensitivity_table.py > /dev/null 2>&1 || echo "  [sensitivity table] failed"
python3 analysis/scripts/inflation_ordering.py > /dev/null 2>&1 || echo "  [inflation ordering] failed"
python3 analysis/scripts/selection_noise.py > /dev/null 2>&1 || echo "  [selection noise] failed"
python3 analysis/scripts/select_nsreg.py > /dev/null 2>&1 || echo "  [nsreg selection] grid not complete yet"
# build_appendix is the head of the chain: render_appendix, make_appendix_tex,
# check_tex_consistency and build_audit_page all read the JSON it writes. When it
# died on a scipy error on 2026-09-07 the script sailed past it and every
# downstream step happily rebuilt itself from an eight-hour-old appendix.json,
# while the tex tables moved on. A failure here must be loud and must stop the
# section, not be inferred later from a consistency warning.
if ! python3 analysis/scripts/build_appendix.py; then
  echo
  echo "  ####################################################################"
  echo "  # build_appendix.py FAILED. analysis/appendix.json is now STALE.   #"
  echo "  # Every table below would be built from it, so they are skipped.   #"
  echo "  # Fix the traceback above and re-run refresh_all.sh.               #"
  echo "  ####################################################################"
  exit 1
fi
# and prove it: a JSON older than the run record it claims to summarise is stale
python3 - <<'FRESH' || exit 1
import os, sys
a, r = "analysis/appendix.json", "results/results.csv"
if os.path.getmtime(a) < os.path.getmtime(r):
    sys.exit(f"appendix.json ({a}) is older than {r}: stale build, refusing to continue")
FRESH
python3 analysis/scripts/render_appendix.py      # -> analysis/appendix.html (publish separately)
python3 analysis/scripts/render_ranking_note.py  # -> analysis/tables/three_method_ranking.md
python3 analysis/scripts/prereg_ledger.py        # -> analysis/tables/prereg_ledger.md
python3 analysis/scripts/make_appendix_tex.py    # -> analysis/tables/tex/*.tex
python3 analysis/scripts/check_appendix_complete.py || echo 'APPENDIX INCOMPLETE (see above) - a rendered table lost rows'
python3 analysis/scripts/check_tex_consistency.py || echo 'TEX CONSISTENCY PROBLEMS (see above)'   # structure + numbers vs appendix.json
python3 analysis/scripts/table_stories.py   # -> analysis/tables/TABLE_STORIES.md
python3 analysis/scripts/build_audit_page.py     # -> analysis/submission_audit.html

echo
echo "=============================================================="
echo " 6b. paper figures and their captions"
echo "=============================================================="
# figures first, then the tex: every caption number is read from the artifacts the
# figures were drawn from, so a rescored campaign can never leave a caption behind
python3 analysis/scripts/make_figures_paper.py || echo 'FIGURES FAILED (see above)'
python3 analysis/scripts/make_figure_tex.py || echo 'FIGURE CAPTIONS FAILED (see above)'
# no chart titles, and no text sitting on other text or on the plotted data
python3 analysis/scripts/check_figure_text.py || echo 'FIGURE TEXT PROBLEMS (see above)'
# figures/method_overview.png is the ONE figure this script does not rebuild: it is
# an AI-generated raster (scientific-schematics skill), so it is neither regenerated
# nor text-checked here. See METHODOLOGY 12.2.
test -f figures/method_overview.pdf || echo 'NOTE: figures/method_overview.pdf missing (method schematic, generated by hand)'

echo
echo "=============================================================="
echo " 7. audit: every number in the prose vs the run record"
echo "=============================================================="
python3 analysis/scripts/audit_claims.py

echo
echo "done."
