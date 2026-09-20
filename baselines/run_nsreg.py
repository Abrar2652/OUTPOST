"""NSReg (Wang et al., ICLR 2025) run under OUTPOST's protocol.

Second reproduced baseline. Every head-to-head in the paper is OUTPOST vs DEMO,
and "single-baseline comparison" is a standard reject line. NSReg is the natural
second: it is the paper that defined this open-set protocol - the rotation over
anomaly classes, 5% labelled normals, 50 labelled anomalies of the seen class -
and its released code names the datasets amz_photo / amz_computers / mag_cs /
yelp, which is where this repository's protocol came from.

WHAT IS HELD IDENTICAL TO OUTPOST, AND HOW

  graph tensors     loaded with OUTPOST's utils.load_data, not NSReg's loader
  split             produced by the SAME function object, utils.ad_split_num,
                    after the SAME set_seed(seed) call, per rotation - the
                    exact sequence main.py executes. NSReg then receives it
                    through its own recorded-split path (a split_50.pkl written
                    per process), so NSReg's own code is not modified.
  metrics           auroc/aupr over test-all and test-unknown, per-metric max
                    over epochs ("best") and metrics at the peak-validation
                    epoch ("val_selected"), computed on OUTPOST's idx_val -
                    NSReg has no validation set of its own (its split makes
                    idx_val = idx_train), so this column is added rather than
                    read off.
  shard             written by main.write_rotation with method="nsreg", so
                    merge_rotations.py and results.csv need no changes.
  sampler           NSReg's legacy torch_geometric NeighborSampler needs
                    torch-sparse, which this environment stubs out; the
                    module-level name is rebound to OUTPOST's pure-torch
                    NeighborSamplerShim, which yields the same
                    (edge_index, e_id, size) triples NSReg's encoder consumes.
                    The shim is what OUTPOST itself trains with.

WHAT IS NSREG'S OWN

  Everything else: architecture, losses, edge labeller, optimiser, learning
  rates, weight decay, batch mode. The only released configuration is mag_cs;
  it is applied to every dataset with input_dim changed, which is a stated
  limitation - their per-dataset settings for photo/computers/yelp were not
  released. Two epoch budgets are run: 201 (their config, to check our
  reproduction against their published numbers) and 400 (the paper's uniform
  protocol, the same budget DEMO is granted).

    python baselines/run_nsreg.py --dataset photo --seed 42 --rotations 0 --epochs 3
"""
import argparse
import hashlib
import os
import pickle
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NSREG = os.path.join(ROOT, "baselines", "NSReg")
os.chdir(ROOT)

# --- OUTPOST first: bind what we need, then get its `utils` MODULE off the path.
# NSReg's `utils/` is a namespace package; a real module anywhere on sys.path
# beats a namespace package everywhere, so both cannot coexist by name.
sys.path.insert(0, ROOT)
import numpy as np
import torch
import utils as _ou
from utils import load_data, set_seed, ad_split_num, NeighborSamplerShim, split_fingerprint
from main import anomaly_classes, write_rotation, parse_set, load_config
from addict import Dict
sys.path.remove(ROOT)
for _m in [m for m in sys.modules if m == "utils" or m.startswith("utils.")]:
    del sys.modules[_m]
sys.path.insert(0, NSREG)
import yaml
from runners import train_runner as _tr
from runners.train_runner import TrainRunner
assert hasattr(sys.modules["utils"], "__path__"), "NSReg's utils package did not resolve"
_tr.NeighborSampler = NeighborSamplerShim          # see docstring: torch-sparse
from sklearn.metrics import roc_auc_score, average_precision_score

# PyG version skew, not an NSReg bug: their code (PyG 2.0.4) hands numpy index
# arrays to torch_geometric.utils.subgraph, and PyG 2.7's index_to_mask calls
# .view(-1) on the subset. The same arrays also go through torch.from_numpy
# elsewhere in their runner, so converting the split up front would break the
# other call sites. Coerce inside subgraph instead, on the module object their
# runner imported, and leave their code untouched.
import torch_geometric.utils as _pu
_orig_subgraph = _pu.subgraph


def _subgraph_np_ok(subset, *args, **kw):
    if isinstance(subset, np.ndarray):
        subset = torch.as_tensor(subset, dtype=torch.long)
    return _orig_subgraph(subset, *args, **kw)


_pu.subgraph = _subgraph_np_ok

KEYS = ("auroc_all", "aupr_all", "auroc_unknown", "aupr_unknown")


class Runner(TrainRunner):
    """NSReg's runner with metrics recorded instead of printed, a validation
    column, and checkpointing disabled."""

    def __init__(self, graph, labels, dset_info, anomaly_info, args, idx_val):
        self.idx_val = np.asarray(idx_val.cpu() if torch.is_tensor(idx_val) else idx_val)
        self.history = []
        super().__init__(graph, labels, dset_info, anomaly_info, args)
        # NSReg embeds the whole graph in ONE 50,000-node batch every epoch
        # (hard-coded in its __init__). On CS that scatters millions of
        # 6,805-dimensional messages and OOMs at ~25 GB even on an empty card.
        # The shim samples lazily in __iter__, so chunking the pass is a batch-
        # size change with no effect on what is computed: every target still
        # gets its own independently sampled neighbourhood, and
        # get_all_embedding concatenates the chunks. Recorded in the shard.
        self.dataloader.batch_size = int(getattr(args, "embed_batch_size", 50000))
        self.eval_loader = NeighborSamplerShim(
            self.edge_index, node_idx=None, sizes=args.sampling_sizes,
            batch_size=int(getattr(args, "eval_batch_size", 50000)),
            shuffle=False, drop_last=False)

    def create_ckpt_dir(self):
        pass

    def save_ckpt(self, epoch):
        pass

    def val(self):
        self.encoder.eval(), self.clf.eval(), self.proj.eval()
        xs = []
        with torch.no_grad():
            for _, n_id, adjs in self.eval_loader:
                adjs = [adj.to(self.device) for adj in adjs]
                out = self.encoder(self.x_all[n_id].to(self.device), adjs)
                xs.append(self.clf(self.proj(out)).cpu())
        pred = torch.sigmoid(torch.cat(xs, 0)).view(-1).numpy()
        labels = self.eval_binary_ad_labels.cpu().numpy()
        assert pred.shape[0] == labels.shape[0], (pred.shape, labels.shape)

        def auc(idx):
            y = labels[idx]
            if y.min() == y.max():          # binary graphs: test-unknown has no anomalies
                return 0.5, 0.0
            return (float(roc_auc_score(y, pred[idx])),
                    float(average_precision_score(y, pred[idx])))
        t = self.split_info["idx_test"]
        ra, pa = auc(t["all"])
        ru, pu = auc(t["unknown"])
        va, _ = auc(self.idx_val)
        rec = {"auroc_all": ra, "aupr_all": pa, "auroc_unknown": ru,
               "aupr_unknown": pu, "val_auc": va}
        self.history.append(rec)
        # Release cached blocks once per epoch. Fragmentation accumulated over
        # epochs is what pushed the tight-cap CS runs over at epoch 200-280
        # (17.4 GB allocated, 8.6 GB reserved-but-unallocated at the OOM); this
        # cannot reclaim blocks inside live segments, but it stops the
        # reclaimable part compounding into new segments.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return rec

    def n_params(self):
        return int(sum(p.numel() for m in (self.encoder, self.proj, self.clf,
                                           self.edge_labeller) for p in m.parameters()))


def nsreg_args(dataset, overrides, device):
    cfg_dir = os.path.join(NSREG, "exp", "config", "mag_cs")
    cfg = {}
    for fn in ("dset", "sampler", "encoder", "train", "exp"):
        cfg.update(yaml.safe_load(open(os.path.join(cfg_dir, fn + ".yaml"))) or {})
    cfg.update({
        "dset_name": dataset,
        "input_dim": load_config(dataset)["input_dim"],
        "eval_every": 1,                        # best-over-epochs needs every epoch
        "use_recorded_split": True,             # OUTPOST's split, via NSReg's own path
        "split_fn": "split_50.pkl",
        "num_train_anomaly": 50,
        "batch_id": None, "view": None,
        "out_dir": os.path.join(ROOT, "runs", "nsreg_ckpt"),
        # full-graph embedding chunk; only cs needs it small (see Runner)
        # Measured on cs, alone on an empty 47 GB card: NSReg's native 50,000-
        # node pass OOMs at 39 GB (it asked for 8.5 GB more), the 1,024-chunk
        # pass peaks at 23.3 GB. The retained autograd set is the same either
        # way; what scales with batch size is the TRANSIENT first-layer
        # tensors - ~335k sampled edges x 6,805 features, ~9 GB each, several
        # alive at once. Chunk the training pass. Evaluation runs under
        # no_grad after the training graph is freed, so it keeps one big batch.
        # CS needed chunking because of its 6,805-dim features, not its node
        # count. The OGB graphs are the opposite shape - 169k/736k nodes at 128
        # dims - so the first-layer transient is ~50x smaller and what matters is
        # the resident full-graph embedding. Chunk them for N, not for features.
        "embed_batch_size": 1024 if dataset == "cs"
        else (20000 if dataset.startswith("ogbn") else 50000),
        # eval at 50,000 on cs OOMed at 43 GB (two ~8.5 GB first-layer
        # transients on top of what stays resident); 1024/1024 measured 23.3 GB.
        # eval at 4096 was tried and OOMed under a 40 GB cap: allocated peak
        # 24.2 GB, fragmentation 10.8 GB, a 6.3 GB transient - transients scale
        # with the chunk. 1024/1024 is the measured configuration (23.3 GB) and
        # stays.
        "eval_batch_size": 1024 if dataset == "cs" \
        else (20000 if dataset.startswith("ogbn") else 50000),
        "mem_fraction_cap": float(os.environ.get("NSREG_MEM_FRACTION", "0.92")),
    })
    cfg.update(overrides)
    cfg["num_epochs"] = cfg["n_epochs"]      # the key every results.csv consumer filters on
    ns = argparse.Namespace(**cfg)
    ns.device = device
    ns.ts = f"{dataset}_{os.getpid()}"
    return ns, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["photo", "computers", "cs", "yelp", "amazon",
                             "tfinance", "ogbn-arxiv", "ogbn-mag"])
    ap.add_argument("--method", default="nsreg", choices=["nsreg"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="n_epochs override")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--rotations", type=int, nargs="*", default=None)
    a = ap.parse_args()

    ov = parse_set(a.set)
    if a.epochs is not None:
        ov["n_epochs"] = a.epochs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Cap what this process may RESERVE. The CS probe allocated 23.3 GB at its
    # peak, but PyTorch's caching allocator grows into whatever the card has
    # free, and in the campaign a CS job was seen holding 35 GB while a
    # neighbour OOMed asking for 3. The repo's MEM_GB comments record the same
    # lesson for OUTPOST. A hard cap forces the allocator to reuse instead of
    # grow. 0.58 (27.6 GB) was too tight: three CS jobs OOMed AGAINST THE CAP
    # with 17-18 GB allocated, 7-8 GB fragmentation and a 3.2 GB transient,
    # while 22 GB sat free on the card. CS is now budgeted a whole card in the
    # campaign file, so the cap only has to stop a runaway past the card: 0.92
    # of 47.5 GB is 43.7 GB, leaving the driver context its margin.
    mem_fraction = float(os.environ.get("NSREG_MEM_FRACTION", "0.92"))
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(mem_fraction, 0)

    # OUTPOST's label budget, as main.py sets it
    budget = Dict(dataname=a.dataset, train_normal_ratio=0.05, train_anormaly_num=50,
                  val_normal_ratio=0.01, val_anormaly_num=30)

    set_seed(a.seed)
    graph, labels, dset_info = load_data(a.dataset)
    labels_np = np.asarray(labels)
    acls_full = anomaly_classes(a.dataset, dset_info)
    acls = acls_full
    if a.rotations:
        unknown = set(a.rotations) - set(int(c) for c in acls_full)
        if unknown:
            raise SystemExit(f"--rotations {sorted(unknown)} not anomaly classes: {list(acls_full)}")
        acls = np.array([c for c in acls_full if int(c) in a.rotations])
    print(f"[{a.dataset}] NSReg, rotations {list(acls)} of {list(acls_full)}")

    tmp = tempfile.mkdtemp(prefix="nsreg_split_")
    ns, cfg = nsreg_args(a.dataset, ov, device)
    ns.saved_idx_dir = tmp
    print(f"[{a.dataset}] n_epochs={ns.n_epochs} eval_every={ns.eval_every} "
          f"input_dim={ns.input_dim} wd={ns.weight_decay} lr={ns.lr}")

    for idx in acls:
        info = {"known_anomaly": int(idx),
                "unknown_anomaly": [int(i) for i in acls_full if i != idx],
                "normal": [int(i) for i in dset_info["class_idx"] if i not in acls_full],
                "all_anomaly": [int(i) for i in acls_full]}
        # the exact sequence main.py runs: re-seed, then split
        set_seed(a.seed)
        split = ad_split_num(labels_np, budget, info)
        if a.train_seed is not None:
            set_seed(a.train_seed)
        with open(os.path.join(tmp, "split_50.pkl"), "wb") as f:
            pickle.dump({int(idx): split}, f)

        t0 = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        runner = Runner(graph, labels_np, dset_info, info, ns, idx_val=split["idx_val"])
        runner.train()
        hist = runner.history
        if not hist:
            raise SystemExit("no evaluations recorded")
        best = {k: max(h[k] for h in hist) for k in KEYS}
        vi = int(np.argmax([h["val_auc"] for h in hist]))
        val_sel = dict(hist[vi])
        final = {k: hist[-1][k] for k in KEYS}
        res = {"best": best, "val_selected": val_sel, "final": final,
               "n_params": runner.n_params(), "rotation_class": int(idx),
               "seconds": round(time.time() - t0, 1),
               "n_evals": len(hist), "val_peak_epoch": vi,
               "split_sha": split_fingerprint(split)}
        if torch.cuda.is_available():
            res["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        print(f"  rotation {idx}: best ROC {best['auroc_all']:.4f} PR {best['aupr_all']:.4f} "
              f"| valsel ROC {val_sel['auroc_all']:.4f} (epoch {vi}) | {res['seconds']:.0f}s")
        cli = argparse.Namespace(dataset=a.dataset, method="nsreg", seed=a.seed,
                                 train_seed=a.train_seed, tag=a.tag)
        p = write_rotation(cli, cfg, res)
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
