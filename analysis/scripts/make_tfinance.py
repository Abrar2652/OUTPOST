"""Build data/tfinance/tfinance.zip (PyG Data) from BWGNN's DGL archive.

T-Finance (Tang et al., ICML 2022): 39,357 accounts, 10 features, 1,804 (4.58%)
labelled anomalous, 42.4M edges, one edge type, already symmetric. The third
real anomaly-detection graph, and the eighth graph overall - the one the
two-regime law owes its out-of-sample test to (METHODOLOGY 6.2-6.3).

Same homogenisation as make_amazon.py, which reproduces data/yelp/yelp.zip
exactly (all 7,693,958 edges, zero difference); on an already-symmetric graph it
is an identity plus de-duplication. Labels are one-hot -> argmax, as in
BWGNN's and ConsisGAD's loaders. Features are cast to float32.
"""
import os, numpy as np, torch
from dgl.data.utils import load_graphs
from torch_geometric.data import Data
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
SRC = "data/_tfinance_raw/ICML22_Rethinking_Anomaly_Detection/dataset/tfinance"


def homogenise_edges(src, dst, n):
    s2, d2 = torch.cat([src, dst]), torch.cat([dst, src])
    m = s2 != d2
    s2, d2 = s2[m], d2[m]
    key = s2.to(torch.int64) * n + d2.to(torch.int64)
    _, idx = np.unique(key.numpy(), return_index=True)
    idx = torch.from_numpy(np.sort(idx))
    return torch.stack([s2[idx], d2[idx]], 0)


g, _ = load_graphs(SRC); g = g[0]
y = g.ndata["label"]
y = (y.argmax(1) if y.dim() == 2 else y).long()
x = g.ndata["feature"].float()
src, dst = g.edges()
ei = homogenise_edges(src, dst, g.num_nodes())
data = Data(x=x, edge_index=ei, y=y)
os.makedirs("data/tfinance", exist_ok=True)
torch.save(data, "data/tfinance/tfinance.zip")
print(f"wrote data/tfinance/tfinance.zip  {data}  anomaly rate {float((y==1).float().mean()):.4f}  "
      f"mean degree {ei.shape[1]/x.shape[0]:.1f}  (raw edges {g.num_edges()})")
