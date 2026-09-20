"""Yelp ablation driver: runs OUTPOST v4(+SG) on Yelp with config overrides.

Mirrors driver.py but for the binary-fraud protocol (single pass,
known_anomaly=1, no unseen rotation) and NEVER writes results.csv --
ablation numbers must not clobber the method's official cells.

Usage: python driver_yelp.py '{"use_fview_gate": false}' A1-nogate
"""
import sys, os, json, logging, contextlib, time

REPO = r"E:\Papers\Graph Neural Network\Toufiq Bhai\Anomaly_2_Editable_FIXED-20260713T023823Z-2-001\Anomaly_2_Editable_FIXED"
sys.path.insert(0, REPO)
os.chdir(REPO)

import yaml
import numpy as np
from addict import Dict
from utils import set_seed, load_data, ad_split_num
from trainer import train_outpost_v4

overrides = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
tag = sys.argv[2] if len(sys.argv) > 2 else "yelp-run"
here = os.path.dirname(os.path.abspath(__file__))
runlog = os.path.join(here, f"{tag}.runlog")

seed = int(overrides.pop("seed", 42))
set_seed(seed)

with open("data/yelp/outpost.yaml") as f:
    cfg = yaml.safe_load(f)
cfg.update(overrides)
args = Dict(cfg)
args.dataname = "yelp"
args.num_classes = 2
args.train_normal_ratio = 0.05
args.train_anormaly_num = 50
args.val_normal_ratio = 0.01
args.val_anormaly_num = 30

logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger("driver_yelp")

graph, labels, dset_info = load_data("yelp")
labels_np = np.asarray(labels)
anomaly_info = {"known_anomaly": 1, "unknown_anomaly": [],
                "normal": [0], "all_anomaly": np.array([1])}
split_info = ad_split_num(labels_np, args, anomaly_info)

t0 = time.time()
with open(runlog, "w", encoding="utf-8") as lf, contextlib.redirect_stdout(lf):
    res = train_outpost_v4(split_info, labels_np, graph, args, anomaly_info, logger)
b, v = res["best"], res["val_selected"]
print(f"[{tag}] BEST: ROC {b['auroc_all']:.4f} PR {b['aupr_all']:.4f} | "
      f"val-selected: ROC {v.get('auroc_all', 0):.4f} PR {v.get('aupr_all', 0):.4f} | "
      f"{time.time()-t0:.0f}s", flush=True)
