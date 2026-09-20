"""Yelp compression sweep: does the lean model hold performance when shrunk?

Reference (lean, h=64): 3-seed mean 0.7364 ROC / 0.3599 PR; per-seed at the two
seeds used here -> s42 0.3293, s0 0.4079 (gate on).

Grid: hidden_dim in {32,16} x fview-gate {on,off}, seeds {42,0} = 8 runs.
Yelp yaml is already lean (mixup/halo off). Judged against the seed-variance
band (0.32-0.41 PR): a compressed config landing in-band = "maintained".

Logs trainable param count per config alongside the metrics. Keep-awake +
crash-safe result logging (same as batch_lean.py).
"""
import sys, os, time, ctypes, contextlib, logging

REPO = r"E:\Papers\Graph Neural Network\Toufiq Bhai\Anomaly_2_Editable_FIXED-20260713T023823Z-2-001\Anomaly_2_Editable_FIXED"
sys.path.insert(0, REPO); os.chdir(REPO)
HERE = os.path.dirname(os.path.abspath(__file__))

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
try:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    _awake = True
except Exception as e:
    _awake = False
    print(f"[batch] WARN keep-awake failed: {e}")

import yaml
import numpy as np
import torch
from addict import Dict
from utils import set_seed, load_data, ad_split_num
from trainer import train_outpost_v4

logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger("batch_compress")

def n_params(h, gate, d0=32, K_p=3):
    from model.outpost import OUTPOST_V4, OUTPOST_V4SG
    m = (OUTPOST_V4SG if gate else OUTPOST_V4)(d0, h, K_p, dropout=0.2)
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

DEMO_YELP = 34209
GRID = []
for h in (32, 16):
    for gate in (True, False):
        for seed in (42, 0):
            GRID.append((h, gate, seed))

graph, labels, dset_info = load_data("yelp")
labels_np = np.asarray(labels)
results_path = os.path.join(HERE, "batch_compress.results")
with open(results_path, "a", encoding="utf-8") as rf:
    rf.write(f"\n=== compression sweep {time.strftime('%Y-%m-%d %H:%M')} keepawake={_awake} ===\n")
    rf.write(f"reference lean h=64: params {n_params(64, True):,} (0.80x DEMO); "
             f"3-seed mean PR 0.3599\n")

for h, gate, seed in GRID:
    with open("data/yelp/outpost.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["hidden_dim"] = h
    cfg["use_fview_gate"] = gate
    args = Dict(cfg)
    args.dataname = "yelp"; args.num_classes = 2
    args.train_normal_ratio = 0.05; args.train_anormaly_num = 50
    args.val_normal_ratio = 0.01; args.val_anormaly_num = 30
    set_seed(seed)
    anomaly_info = {"known_anomaly": 1, "unknown_anomaly": [],
                    "normal": [0], "all_anomaly": np.array([1])}
    split_info = ad_split_num(labels_np, args, anomaly_info)
    p = n_params(h, gate)
    tag = f"h{h}-gate{'On' if gate else 'Off'}-s{seed}"
    t0 = time.time()
    try:
        with open(os.path.join(HERE, f"cmp_{tag}.runlog"), "w", encoding="utf-8") as lf, \
                contextlib.redirect_stdout(lf):
            res = train_outpost_v4(split_info, labels_np, graph, args, anomaly_info, logger)
        b, v = res["best"], res["val_selected"]
        line = (f"[{tag}] params {p:,} ({p/DEMO_YELP:.2f}x DEMO) | "
                f"BEST ROC {b['auroc_all']:.4f} PR {b['aupr_all']:.4f} | "
                f"val-sel ROC {v.get('auroc_all',0):.4f} PR {v.get('aupr_all',0):.4f} | {time.time()-t0:.0f}s")
    except Exception as e:
        line = f"[{tag}] params {p:,} FAILED: {type(e).__name__}: {e}"
    print(line, flush=True)
    with open(results_path, "a", encoding="utf-8") as rf:
        rf.write(line + "\n")

if _awake:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
print("[batch] compression sweep done", flush=True)
