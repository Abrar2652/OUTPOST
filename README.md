# OUTPOST — Open-Set Graph Anomaly Detection

Detecting anomalies from classes that were never labelled during training.

This repository is the full record of the study, not a demo: the method, the
baselines' runners, every per-rotation result the tables are built from, the
scripts that turn those into tables and figures, and the checks that fail loudly
when any of it drifts. `METHODOLOGY.md` is the long form — protocol, ablations,
pre-registered predictions with their verdicts, and every correction made along
the way, including the ones that went against the method.

## Install

```bash
conda create -n outpost python=3.11 -y && conda activate outpost
pip install -r requirements.txt
```

## Reproduce

```bash
python reproduce.py --seeds 42 0 1
```

Checks the environment, unpacks the data, trains, and writes the tables and
figures. Interruption is safe; `--resume` continues. This covers six of the eight
graphs — Photo, Computers, CS, Yelp, ogbn-arxiv, ogbn-mag. Amazon and T-Finance
were added later and live in the campaigns below, because they only ever ran as
part of the full matrix. A three-epoch pipeline check, roughly ten minutes:

```bash
python reproduce.py --quick
```

Rebuild every table, figure and test from whatever has already finished. No GPU,
safe to run at any time, and the fastest way to see what this repository contains:

```bash
bash analysis/scripts/refresh_all.sh
```

## Data

Not in the repository. `ogbn-arxiv` and `ogbn-mag` download on first use (~480 MB
and ~1 GB unpacked). The other four — Photo, Computers, CS, Yelp — ship as a
132 MB `dataset.zip`, which exceeds GitHub's 100 MB file limit, so it is attached
to the GitHub release instead. Put it in the repository root and `reproduce.py`
unpacks it:

```bash
# from the Releases page, then:
python reproduce.py --quick      # verifies the unpack before any long run
```

## Baselines

Not vendored, for licensing reasons set out in `baselines/README.md` — GGAD's
upstream grants no licence at all. Fetch them:

```bash
bash baselines/fetch_baselines.sh
```

`baselines/run_nsreg.py` drives NSReg's trainer but loads graphs through
OUTPOST's own `utils.load_data`, so both methods see identical tensors, splits and
seeds. Without that they would differ by their data loaders before any model ran.

## What the results actually say

Read the generated tables rather than this paragraph, because they are rebuilt
from the run record and this paragraph is not:

```
analysis/tables/table1_small.tex     small graphs, oracle selection
analysis/tables/table2_large.tex     large graphs, oracle selection
analysis/tables/*_valsel.tex         the same under the deployable rule
analysis/tables/prereg_ledger.md     all 40 pre-registered predictions, with verdicts
analysis/appendix.html               every cell traced to the runs behind it
```

Two things worth knowing before you read them. The headline depends on the
selection rule: reporting the best epoch per metric flatters every method, and the
`*_valsel` tables, which select at the peak-validation epoch, are the ones that
describe a deployable system. And of the 40 pre-registered predictions, 18 came out
CONFIRMED, 12 FALSIFIED, 3 SPLIT and 1 NULL, with the rest superseded or restated —
several of the falsified ones are load-bearing. That is the point of having written
them down before running anything; `METHODOLOGY.md` §5.4, §6.5 and §11 are where
those negative results live.

## The full experiment matrix

`reproduce.py` runs six datasets in sequence. The paper's matrix is larger — ten
seeds, re-run DEMO and NSReg baselines at matched seeds and splits, ablation arms,
sensitivity sweeps — so it is expressed as *campaigns*: job lists a scheduler
spreads across free GPUs, skipping anything already finished.

```bash
python campaigns/build_gpu7.py
python analysis/scripts/run_campaign.py campaigns/gpu7.json --gpus 7
python analysis/scripts/merge_rotations.py --status     # progress, any time
```

Interrupting is safe and resuming is the same command: finished runs are skipped
by their results row, and each rotation of a long dataset is its own job, so an
interruption costs one rotation rather than a seed.

## Layout

```
reproduce.py            entry point
main.py                 one run   (--rotations shards a dataset across GPUs)
model.py trainer.py losses.py utils.py results_writer.py
config.json             defaults and per-dataset settings
METHODOLOGY.md          protocol, verdicts, corrections, known limits

campaigns/              the experiment matrix as job lists
baselines/              our runners; upstream code is fetched, not vendored
results/                results.csv and one JSON per rotation — the run record
figures/                every figure, .pdf and .png
analysis/
  REVIEWER_PROOFING.md  what was audited, by objection
  appendix.json/.html   every table cell traced to its runs
  figstyle.py           one palette, one label placer, one dataset spelling
  tables/               generated .csv and .tex
  scripts/
    refresh_all.sh          everything below, in order
    run_campaign.py         schedules a campaign over free GPUs
    merge_rotations.py      assembles sharded rotations into results rows
    stats.py                cells, paired tests, ablations, Holm correction
    prereg_ledger.py        scores the pre-registered predictions
    make_figures_paper.py   every figure, one function each
    make_figure_tex.py      figure captions, numbers read from artifacts
    check_figure_text.py    no chart titles, no overlapping text
    check_tex_consistency.py  every .tex cell against appendix.json
    check_appendix_complete.py  no rendered table has lost rows
    audit_claims.py         every number in the prose against the run record
```

## The checks are the point

Numbers in captions and tables are read from result files when they are generated,
never typed. `analysis/scripts/refresh_all.sh` regenerates all of it and then tries to break it:

```
appendix completeness   no rendered table silently lost rows
tex consistency         every numeric cell matches appendix.json
figure text             no chart titles, no text over text or over data
claims audit            every number in the prose still exists in the run record
```

Each of these was written because the corresponding mistake had already happened
once. `METHODOLOGY.md` §11 and §12 name them.

## Citation

See `CITATION.cff`.

## Licence

MIT, `LICENSE`. Datasets and fetched baselines keep their own terms.
