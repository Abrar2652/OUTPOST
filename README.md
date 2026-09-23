# OUTPOST: Open-Set Graph Anomaly Detection

OUTPOST detects graph anomalies from classes that were never labeled during training.

## Install

```bash
conda create -n outpost python=3.11 -y && conda activate outpost
pip install -r requirements.txt
```

## Train from scratch

```bash
python reproduce.py --quick           # 3-epoch end-to-end check, about 10 min
python reproduce.py --seeds 42 0 1    # Photo, Computers, CS, Yelp, ogbn-arxiv, ogbn-mag
```

Runs can be interrupted; `--resume` continues.

## Data

Not included. `ogbn-arxiv` and `ogbn-mag` download on first use. Photo, Computers, CS and Yelp come in `dataset.zip` (132 MB), distributed separately because it exceeds the 100 MB per-file limit. Place it in the repository root and `reproduce.py` unpacks it.

## Layout

```text
reproduce.py, main.py      entry points
model.py, trainer.py, losses.py, utils.py
config.json                default and per-dataset settings
```

## License

MIT (see `LICENSE`). Datasets and fetched baselines keep their own licenses.
