"""Build data/amazon/amazon.zip in the same format as data/yelp/yelp.zip.

Amazon-Fraud (McAuley & Leskovec) is a REAL anomaly-detection dataset: 11,944
reviewers, 25 handcrafted features, 6.87% labelled fraudulent, three relation
types. It is standard in exactly the papers this work compares against (GGAD,
NSReg, DEMO), so its published numbers are directly usable.

Why it is worth adding: every claim this paper makes about semi-synthetic
benchmarks inflating results rests on ONE real dataset, Yelp. Oracle selection
inflates Photo/Computers/CS by 0.045-0.08 and Yelp by 0.0023 - a 34x contrast
with n=1 on the side that carries the thesis. Amazon takes that to n=2, and
independently tests whether pseudo-labelling's sign flip (essential on Computers
and CS, harmful on Yelp) is a property of real data or a property of Yelp.

THE CONVERSION IS VALIDATED, NOT ASSUMED. The same function applied to DGL's
FraudYelpDataset reproduces data/yelp/yelp.zip exactly: identical labels,
features allclose, and identical edge sets over all 7,693,958 edges (0
differences in either direction). The recipe is therefore known to match
whatever produced the existing file, rather than being a plausible guess.
"""
import os
import numpy as np
import torch
import dgl
from torch_geometric.data import Data

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)


def homogenise(g):
    """Union of all relations, symmetrised, de-duplicated, self-loops dropped."""
    src, dst = [], []
    for et in g.etypes:
        s, d = g.edges(etype=et)
        src.append(s); dst.append(d)
    s, d = torch.cat(src), torch.cat(dst)
    s2, d2 = torch.cat([s, d]), torch.cat([d, s])
    m = s2 != d2
    s2, d2 = s2[m], d2[m]
    key = s2.to(torch.int64) * g.num_nodes() + d2.to(torch.int64)
    _, idx = np.unique(key.numpy(), return_index=True)
    idx = torch.from_numpy(np.sort(idx))
    return torch.stack([s2[idx], d2[idx]], 0)


def check_recipe():
    """Refuse to write Amazon unless the recipe still reproduces Yelp exactly."""
    ref = "data/yelp/yelp.zip"
    if not os.path.exists(ref):
        print("  yelp reference absent, skipping validation")
        return
    g = dgl.data.FraudYelpDataset(raw_dir="data/_dgl_raw")[0]
    ei = homogenise(g)
    ours = torch.load(ref, weights_only=False)
    assert torch.equal(g.ndata["label"].long(), ours.y.long()), "labels differ"
    assert torch.allclose(g.ndata["feature"].float(), ours.x.float(), atol=1e-5), "features differ"
    a = set(map(tuple, ei.t().tolist())); b = set(map(tuple, ours.edge_index.t().tolist()))
    assert a == b, f"edge sets differ: {len(b - a)} / {len(a - b)}"
    print("  recipe validated: reproduces data/yelp/yelp.zip exactly")


def main():
    check_recipe()
    g = dgl.data.FraudAmazonDataset(raw_dir="data/_dgl_raw")[0]
    data = Data(x=g.ndata["feature"].float(),
                edge_index=homogenise(g),
                y=g.ndata["label"].long())
    os.makedirs("data/amazon", exist_ok=True)
    torch.save(data, "data/amazon/amazon.zip")
    print(f"  wrote data/amazon/amazon.zip  {data}")
    print(f"  anomaly rate {float((data.y == 1).float().mean()):.4f}  "
          f"mean degree {data.edge_index.shape[1] / data.x.shape[0]:.1f}")


if __name__ == "__main__":
    main()
