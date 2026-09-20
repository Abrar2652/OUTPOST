"""Precompute the personalised-PageRank matrix DEMO's mixup needs.

DEMO's anomaly_mixup interpolates between anomalies weighted by PPR, and it is
the paper's headline contribution. The shipped configuration has `mixup: False`
and `main.py` passes `ppr_matrix=None`, so every DEMO arm we ran was in fact
DEMO *w/o Mix* - its own ablation. Comparing against that and calling it DEMO is
the kind of thing a referee rejects a paper for, correctly.

`utils.compute_ppr` cannot help here: for anything other than Photo/Computers it
routes through DGL's APPNPConv, and DGL is broken in this environment
(`libcudart.so.12` missing). This is the same computation in pure torch -
APPNP power iteration, K=20, alpha=0.2, exactly the parameters DEMO uses:

    H_0 = I ;  H_{k+1} = (1-a) * A_hat @ H_k + a * I ;  A_hat = D^-1/2 (A+I) D^-1/2

Stored float32 rather than float64. The values are identical to within float32
precision and it halves the footprint, which is what puts Yelp (8.4 GB rather
than 16.9) comfortably on one card.

    python analysis/scripts/make_ppr.py --dataset yelp
    python analysis/scripts/make_ppr.py --dataset photo --check
"""
import argparse, os, sys, time
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT); sys.path.insert(0, ROOT)


def appnp_ppr(edge_index, N, alpha=0.2, K=20, device=None, block=4096):
    """Dense PPR by APPNP power iteration, in column blocks to bound memory."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    src, dst = edge_index[0].long(), edge_index[1].long()
    loop = torch.arange(N)
    src = torch.cat([src, loop]); dst = torch.cat([dst, loop])   # A + I
    deg = torch.bincount(dst, minlength=N).float().clamp(min=1)
    w = deg[src].rsqrt() * deg[dst].rsqrt()                      # D^-1/2 (A+I) D^-1/2
    A = torch.sparse_coo_tensor(torch.stack([dst, src]), w, (N, N)).coalesce().to(dev)

    out = np.empty((N, N), dtype=np.float32)
    for lo in range(0, N, block):
        hi = min(lo + block, N)
        E = torch.zeros(N, hi - lo, device=dev)
        E[torch.arange(lo, hi, device=dev), torch.arange(hi - lo, device=dev)] = 1.0
        H = E.clone()
        for _ in range(K):
            H = (1 - alpha) * torch.sparse.mm(A, H) + alpha * E
        out[:, lo:hi] = H.cpu().numpy()
        del E, H
        if dev != "cpu":
            torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--check", action="store_true",
                    help="cross-check against utils.compute_ppr's exact inverse "
                         "(Photo/Computers only, where that path does not need DGL)")
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--force-appnp", action="store_true",
                    help="use APPNP even on photo/computers")
    a = ap.parse_args()

    from utils import load_data
    graph, _, _ = load_data(a.dataset)
    N = graph.num_nodes
    path = f"data/{a.dataset.replace('-', '_')}/ppr_matrix.npy"

    # Match the authors' own branching, not one method everywhere. Their
    # compute_ppr uses the EXACT closed form a(I - (1-a)A~)^-1 on photo and
    # computers, and APPNP(K=20) on everything larger. Using APPNP on photo
    # would be a 2.3e-04 deviation from the baseline as published - small, but
    # it is their baseline and there is no reason to perturb it.
    if a.dataset in ("photo", "computers") and not a.force_appnp:
        import numpy.linalg as npl
        print("  exact inverse (the authors' path for this dataset)")
        t0 = time.time()
        src, dst = graph.edge_index[0].numpy(), graph.edge_index[1].numpy()
        A = np.zeros((N, N), dtype=np.float64)
        A[src, dst] = 1.0
        A = A + np.eye(N)
        dinv = np.diag(1.0 / np.sqrt(A.sum(1)))
        At = dinv @ A @ dinv
        ppr = (0.2 * npl.inv(np.eye(N) - 0.8 * At)).astype(np.float32)
        print(f"  computed in {time.time()-t0:.0f}s; column sums "
              f"min {ppr.sum(0).min():.3f} max {ppr.sum(0).max():.3f}")
        np.save(path, ppr)
        print(f"  saved {os.path.getsize(path)/1e9:.1f} GB")
        return
    print(f"{a.dataset}: N={N}, dense float32 = {N*N*4/1e9:.1f} GB -> {path}")

    t0 = time.time()
    ppr = appnp_ppr(graph.edge_index, N, block=a.block)
    print(f"  computed in {time.time()-t0:.0f}s; row sums "
          f"min {ppr.sum(0).min():.3f} max {ppr.sum(0).max():.3f}")

    if a.check:
        from utils import compute_ppr
        exact = compute_ppr(graph, a.dataset)
        d = np.abs(ppr - exact.astype(np.float32))
        print(f"  vs exact inverse: max abs diff {d.max():.2e}, mean {d.mean():.2e}")
        print("  (APPNP is a K=20 truncation of the exact series, so a small "
              "difference is expected and is what DEMO itself uses)")

    np.save(path, ppr)
    print(f"  saved {os.path.getsize(path)/1e9:.1f} GB")


if __name__ == "__main__":
    main()
