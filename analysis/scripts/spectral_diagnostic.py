"""Phase-0 diagnostic: is unseen-anomaly detectability a spectral/topological property?

Thesis under test
-----------------
In open-set GAD, whether an *unseen* anomaly class is detectable is governed by
its local spectral signature -- how its nodes sit in the graph relative to the
normal mass -- rather than by the novelty of the class per se:

  * a *homophilous* anomaly class (nodes clustered with each other) survives
    low-pass message passing: propagation concentrates its deviant features and
    any smoothing-based / GNN detector can separate it without labels;
  * a *scattered* anomaly class (nodes isolated inside normal neighborhoods,
    i.e. locally heterophilous) is averaged INTO normality by the same
    low-pass operations every current open-set GAD backbone uses -- and
    confidence-based self-training then certifies the error by pseudo-labeling
    those nodes normal.

Pre-registered predictions
--------------------------
P1 (between-class)  Detectability of an anomaly class under a training-free
    normal-prototype detector increases with the class's internal label
    homophily (same-class neighbor fraction).
P2 (within-class)   Within one anomaly class, per-node anomaly score under the
    prototype detector correlates positively with the node's same-class
    neighbor fraction.
P3 (smoothing gain) Feature propagation (k hops) improves detectability for
    homophilous anomaly classes and is flat/negative for scattered ones.
P4 (real vs synthetic)  Real fraud (Yelp) exhibits the scattered profile,
    i.e. it aligns with the *hard* synthetic classes; the *easy* synthetic
    classes owe their detectability to being homophilous minority clusters
    (a topological artifact of the class-relabeling benchmark construction).

Metrics (per node v, over the undirected 1-hop neighborhood N(v))
-----------------------------------------------------------------
  same_frac   |{u in N(v): y_u = y_v}| / |N(v)|        (label homophily)
  norm_frac   fraction of N(v) that is normal-class    (anomaly isolation)
  feat_dissim 1 - cos(x_v, mean_{u in N(v)} x_u)       (local feature contrast)
  dirichlet   local Dirichlet energy of the feature signal under the
              symmetric-normalized adjacency:
              sum_u a_hat_vu * ||x_v/sqrt(d_v) - x_u/sqrt(d_u)||^2 / ||x_v||^2
              (high = the node is a high-frequency component)

Detector (training-free, label-free at test time)
-------------------------------------------------
Spherical k-means prototypes (K=12) fit on the features of a 5% sample of
normal-class nodes (mirroring the harness's labeled-normal budget), at
propagation depth k in {0,1,2,3} under symmetric-normalized adjacency with
self-loops. Score(v) = 1 - max_j cos(z_v, mu_j). AUC is computed per anomaly
class against all normal nodes.

Outputs
-------
  analysis/tables/spectral_<dataset>.csv   per-class summary rows
  stdout                                    formatted tables + tests

Statistical reporting: median per group, separation AUC (probability of
superiority) between anomaly classes, two-sided Mann-Whitney U p-value,
and Spearman rho for within-class correlations.

Usage
-----
  python analysis/spectral_diagnostic.py --dataset photo
  python analysis/spectral_diagnostic.py --dataset all      # photo computers cs yelp

Reproducibility: seed fixed (42) for the normal-train sample and k-means init.
Runs on CPU in minutes for all four datasets.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from utils import npz_data_to_pyg_graph  # noqa: E402


SEED = 42
K_PROTO = 12
NORMAL_TRAIN_RATIO = 0.05
HOPS = (0, 1, 2, 3)


# ----------------------------------------------------------------------
# data loading (self-contained; avoids utils.load_data's yelp path quirk)
# ----------------------------------------------------------------------

def load_dataset(name):
    """Return (x [N,d] float tensor, edge_index [2,E], y [N] int array,
    anomaly_classes list[int], normal_classes list[int])."""
    if name in ("photo", "computers", "cs"):
        data = np.load(f"data/{name}/{name}.npz", allow_pickle=True)
        graph, labels, dset_info = npz_data_to_pyg_graph(data)
        y = np.asarray(labels)
        per = np.asarray(dset_info["class_per"])
        anomaly_classes = [int(c) for c in np.where((per <= 0.05) & (per >= 0.0))[0]]
        normal_classes = [int(c) for c in dset_info["class_idx"] if c not in anomaly_classes]
        return graph.x.float(), graph.edge_index, y, anomaly_classes, normal_classes
    if name in ("yelp", "amazon", "tfinance"):
        # amazon is the out-of-sample test of the law: its diagnostic is
        # computed and a prediction registered BEFORE any model is trained on
        # it, so the law's 7th graph is a genuine forecast rather than another
        # fitted point.
        g = torch.load(f"data/{name}/{name}.zip", weights_only=False)
        y = np.asarray(g.y)
        return g.x.float(), g.edge_index, y, [1], [0]
    if name in ("ogbn-arxiv", "ogbn-mag"):
        # The two OGB graphs contribute 19 further anomaly classes. Adding them
        # is what takes the between-class regression of the detectability law
        # from 16 points to 35, which is the sample-size limitation recorded in
        # the Phase-0 threats to validity.
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        from main import anomaly_classes as _acls
        from utils import load_data as _load
        graph, labels, info = _load(name)
        y = np.asarray(labels)
        anomaly = [int(c) for c in _acls(name, info)]
        normal = [int(c) for c in info["class_idx"] if int(c) not in anomaly]
        return graph.x.float(), graph.edge_index, y, anomaly, normal
    raise ValueError(name)


# ----------------------------------------------------------------------
# per-node structural / spectral metrics
# ----------------------------------------------------------------------

def node_metrics(x, edge_index, y):
    """Vectorized per-node metrics over the 1-hop neighborhood (no self-loops)."""
    N = x.size(0)
    src, dst = edge_index[0], edge_index[1]
    keep = src != dst
    src, dst = src[keep], dst[keep]
    deg = torch.bincount(dst, minlength=N).clamp(min=1).float()

    y_t = torch.from_numpy(y.astype(np.int64))
    same = (y_t[src] == y_t[dst]).float()
    same_frac = torch.zeros(N).scatter_add_(0, dst, same) / deg

    # neighbor mean features -> local feature dissimilarity
    nbr_sum = torch.zeros(N, x.size(1)).index_add_(0, dst, x[src])
    nbr_mean = nbr_sum / deg.unsqueeze(1)
    feat_dissim = 1.0 - F.cosine_similarity(x, nbr_mean, dim=1, eps=1e-8)

    # local Dirichlet energy under symmetric normalization
    xs = x / deg.sqrt().unsqueeze(1)
    diff2 = ((xs[dst] - xs[src]) ** 2).sum(1)
    dirichlet = torch.zeros(N).scatter_add_(0, dst, diff2) / (x.pow(2).sum(1) + 1e-8)

    has_nbrs = torch.bincount(dst, minlength=N) > 0
    return {"same_frac": same_frac.numpy(), "feat_dissim": feat_dissim.numpy(),
            "dirichlet": dirichlet.numpy(), "has_nbrs": has_nbrs.numpy(),
            "deg": deg.numpy()}


# ----------------------------------------------------------------------
# training-free prototype detector at several propagation depths
# ----------------------------------------------------------------------

def sym_norm_adj(edge_index, N):
    from torch_geometric.utils import add_self_loops, degree
    ei, _ = add_self_loops(edge_index, num_nodes=N)
    row, col = ei
    d = degree(row, N).clamp(min=1)
    w = (d[row] * d[col]).rsqrt()
    return torch.sparse_coo_tensor(torch.stack([row, col]), w, (N, N)).coalesce()

def spherical_kmeans(z, K, iters=50):
    mu = z[torch.randperm(z.size(0))[:K]].clone()
    for _ in range(iters):
        a = (z @ mu.t()).argmax(1)
        for j in range(K):
            m = z[a == j]
            if m.size(0) > 0:
                mu[j] = F.normalize(m.mean(0), dim=0)
    return mu

def prototype_scores_by_hop(x, edge_index, y, normal_classes):
    """Score(v) = 1 - max_j cos(z_v, mu_j), prototypes from 5% of normals."""
    N = x.size(0)
    A = sym_norm_adj(edge_index, N)
    normal_idx = np.where(np.isin(y, normal_classes))[0]
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(normal_idx), generator=g)
    tr_n = normal_idx[perm[: int(len(normal_idx) * NORMAL_TRAIN_RATIO)].numpy()]

    scores = {}
    Xp = x.clone()
    torch.manual_seed(SEED)
    for k in range(max(HOPS) + 1):
        if k > 0:
            Xp = torch.sparse.mm(A, Xp)
        if k in HOPS:
            Z = F.normalize(Xp, dim=-1)
            mu = spherical_kmeans(Z[tr_n], K_PROTO)
            scores[k] = (1.0 - (Z @ mu.t()).max(1).values).numpy()
    return scores


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------

def sep_auc(a, b):
    """Probability-of-superiority AUC separating sample a (pos) from b (neg)."""
    lab = np.r_[np.ones(len(a)), np.zeros(len(b))]
    return roc_auc_score(lab, np.r_[a, b])

def run(dataset):
    print(f"\n{'=' * 72}\n[{dataset}]")
    x, edge_index, y, anomaly_classes, normal_classes = load_dataset(dataset)
    N = x.size(0)
    m = node_metrics(x, edge_index, y)
    scores = prototype_scores_by_hop(x, edge_index, y, normal_classes)
    normal_mask = np.isin(y, normal_classes)

    rows = []
    print(f"N={N}  anomaly classes={anomaly_classes}  "
          f"(normals: {int(normal_mask.sum())} nodes)")
    hdr = (f"{'class':>6} {'n':>6} {'same_frac':>10} {'feat_dissim':>12} "
           f"{'dirichlet':>10} | AUC@hop " + " ".join(f"{k:>6}" for k in HOPS)
           + f" {'gain':>7}")
    print(hdr)

    cls_summary = {}
    for c in anomaly_classes:
        mask = (y == c) & m["has_nbrs"]
        med = {k: float(np.median(m[k][mask])) for k in ("same_frac", "feat_dissim", "dirichlet")}
        aucs = {}
        for k in HOPS:
            s = scores[k]
            aucs[k] = roc_auc_score(
                np.r_[np.ones(mask.sum()), np.zeros(normal_mask.sum())],
                np.r_[s[mask], s[normal_mask]])
        gain = aucs[max(HOPS)] - aucs[0]
        # P2: within-class Spearman(same_frac, score at best hop)
        best_hop = max(aucs, key=aucs.get)
        rho, rho_p = spearmanr(m["same_frac"][mask], scores[best_hop][mask])
        cls_summary[c] = {"mask": mask, "aucs": aucs, "med": med}
        print(f"{c:>6} {int(mask.sum()):>6} {med['same_frac']:>10.3f} "
              f"{med['feat_dissim']:>12.3f} {med['dirichlet']:>10.3f} |         "
              + " ".join(f"{aucs[k]:>6.3f}" for k in HOPS)
              + f" {gain:>+7.3f}   [P2 rho={rho:+.3f} p={rho_p:.1e}]")
        rows.append({"dataset": dataset, "class": c, "n": int(mask.sum()),
                     **{f"med_{k}": v for k, v in med.items()},
                     **{f"auc_hop{k}": aucs[k] for k in HOPS},
                     "smoothing_gain": gain, "p2_spearman": rho, "p2_p": rho_p})

    # normal-node reference row
    nm = normal_mask & m["has_nbrs"]
    print(f"{'norm':>6} {int(nm.sum()):>6} "
          f"{np.median(m['same_frac'][nm]):>10.3f} "
          f"{np.median(m['feat_dissim'][nm]):>12.3f} "
          f"{np.median(m['dirichlet'][nm]):>10.3f} |")

    # P1: between-anomaly-class separation tests on structural metrics
    if len(anomaly_classes) >= 2:
        a, b = anomaly_classes[0], anomaly_classes[1]
        ma, mb = cls_summary[a]["mask"], cls_summary[b]["mask"]
        print(f"\nP1 between-class tests ({a} vs {b}):")
        for k in ("same_frac", "feat_dissim", "dirichlet"):
            u = mannwhitneyu(m[k][ma], m[k][mb], alternative="two-sided")
            print(f"  {k:>12}: sepAUC={sep_auc(m[k][ma], m[k][mb]):.3f} "
                  f"MWU p={u.pvalue:.2e}")

    os.makedirs("analysis/tables", exist_ok=True)
    import csv
    out = f"analysis/tables/spectral_{dataset}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {out}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all",
                    choices=["photo", "computers", "cs", "yelp", "amazon", "tfinance",
                             "ogbn-arxiv", "ogbn-mag", "all", "all+ogb"])
    args = ap.parse_args()
    small = ["photo", "computers", "cs", "yelp"]
    targets = (small if args.dataset == "all" else
               small + ["ogbn-arxiv", "ogbn-mag"] if args.dataset == "all+ogb"
               else [args.dataset])
    # data/ lives at the repository root, three levels up from this file
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    for d in targets:
        run(d)
