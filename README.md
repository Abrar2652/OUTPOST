# OUTPOST: Open-Set Graph Anomaly Detection

OUTPOST detects graph anomalies from classes that were never labeled during
training.

## Install

```bash
conda create -n outpost python=3.11 -y && conda activate outpost
pip install -r requirements.txt
```

## Reproduce the tables and figures

No GPU needed; takes a few minutes:

```bash
bash analysis/scripts/refresh_all.sh
```

This rebuilds `analysis/tables/` and `figures/` from the run records in
`results/`, then runs the checks listed below.

## Train from scratch

```bash
python reproduce.py --quick           # 3-epoch end-to-end check, about 10 min
python reproduce.py --seeds 42 0 1    # Photo, Computers, CS, Yelp, ogbn-arxiv, ogbn-mag
```

Runs can be interrupted; `--resume` continues. Amazon, T-Finance and the full
experiment matrix (ten seeds, baselines re-run at matched seeds and splits,
ablations, sensitivity sweeps) run as campaigns:

```bash
python campaigns/build_gpu7.py
python analysis/scripts/run_campaign.py campaigns/gpu7.json --gpus 7
python analysis/scripts/merge_rotations.py --status
```

## Data

Not included. ogbn-arxiv and ogbn-mag download on first use. Photo, Computers, CS
and Yelp come in `dataset.zip` (132 MB), distributed separately because it exceeds
the 100 MB per-file limit. Place it in the repository root and `reproduce.py`
unpacks it.

## Baselines

Not included, for licensing reasons. Fetch NSReg, ConsisGAD and GGAD with:

```bash
bash baselines/fetch_baselines.sh
```

`baselines/run_nsreg.py` runs NSReg's own trainer on graphs loaded through this
repository's `utils.load_data`, so both methods see identical tensors, splits and
seeds.

## Layout

```
reproduce.py, main.py      entry points
model.py, trainer.py, losses.py, utils.py
config.json                default and per-dataset settings
campaigns/                 full experiment matrix as job lists
baselines/                 baseline runners
results/                   results.csv and one JSON record per rotation
analysis/tables/           generated tables (CSV)
analysis/appendix.html     every table cell traced to the runs behind it
figures/                   all figures (PDF and PNG)
```

## Checks

Every number in a generated table or figure caption is read from a result file at
build time. After rebuilding, `refresh_all.sh` verifies:

```
appendix completeness   no table lost rows
table consistency       every table cell matches analysis/appendix.json
figure text             no chart titles, no overlapping text
claims audit            every reported number still exists in the run record
```

## License

MIT (see `LICENSE`). Datasets and fetched baselines keep their own licenses.
