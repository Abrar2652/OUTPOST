# OUTPOST: Open-Set Graph Anomaly Detection

OUTPOST detects graph anomalies from classes that were never labeled during training.

## Install

```bash
conda create -n outpost python=3.11 -y && conda activate outpost
pip install -r requirements.txt
```

## Run

Ensure `dataset.zip` is placed in the repository root if you are running Photo, Computers, CS, or Yelp. (ogbn datasets will download automatically).

```bash
python reproduce.py --quick           # 3-epoch end-to-end check
python reproduce.py --seeds 42 0 1    # Photo, Computers, CS, Yelp, ogbn-arxiv, ogbn-mag
```

Runs can be interrupted; `--resume` continues.
