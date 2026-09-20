"""Complete the gate 2x2 across seeds: lean + NO gate at seeds 0 and 1.

Pairs with batch_lean.py (lean + gate). Together they give, per seed,
the gate on/off contrast in the lean config -- deciding whether the
spectral gate is a genuine positive mechanism or within noise.

Same keep-awake + crash-safe result logging as batch_lean.py.
"""
import sys, os, time, ctypes, contextlib, logging

REPO = r"E:\Papers\Graph Neural Network\Toufiq Bhai\Anomaly_2_Editable_FIXED-20260713T023823Z-2-001\Anomaly_2_Editable_FIXED"
sys.path.insert(0, REPO)
os.chdir(REPO)
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
from addict import Dict
from utils import set_seed, load_data, ad_split_num
from trainer import train_outpost_v4

logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger("batch_gate")

RUNS = [
    ("A8-lean-nogate-s0", {"use_mixup": False, "use_halo": False, "use_fview_gate": False}, 0),
    ("A9-lean-nogate-s1", {"use_mixup": False, "use_halo": False, "use_fview_gate": False}, 1),
]

graph, labels, dset_info = load_data("yelp")
labels_np = np.asarray(labels)
results_path = os.path.join(HERE, "batch_gate.results")
with open(results_path, "a", encoding="utf-8") as rf:
    rf.write(f"\n=== gate-2x2 batch start {time.strftime('%Y-%m-%d %H:%M')} keepawake={_awake} ===\n")

for tag, ov, seed in RUNS:
    with open("data/yelp/outpost.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg.update(ov)
    args = Dict(cfg)
    args.dataname = "yelp"; args.num_classes = 2
    args.train_normal_ratio = 0.05; args.train_anormaly_num = 50
    args.val_normal_ratio = 0.01; args.val_anormaly_num = 30
    set_seed(seed)
    anomaly_info = {"known_anomaly": 1, "unknown_anomaly": [],
                    "normal": [0], "all_anomaly": np.array([1])}
    split_info = ad_split_num(labels_np, args, anomaly_info)
    t0 = time.time()
    try:
        with open(os.path.join(HERE, f"{tag}.runlog"), "w", encoding="utf-8") as lf, \
                contextlib.redirect_stdout(lf):
            res = train_outpost_v4(split_info, labels_np, graph, args, anomaly_info, logger)
        b, v = res["best"], res["val_selected"]
        line = (f"[{tag}] seed={seed} BEST ROC {b['auroc_all']:.4f} PR {b['aupr_all']:.4f} | "
                f"val-sel ROC {v.get('auroc_all',0):.4f} PR {v.get('aupr_all',0):.4f} | {time.time()-t0:.0f}s")
    except Exception as e:
        line = f"[{tag}] seed={seed} FAILED: {type(e).__name__}: {e}"
    print(line, flush=True)
    with open(results_path, "a", encoding="utf-8") as rf:
        rf.write(line + "\n")

if _awake:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
print("[batch] done", flush=True)
