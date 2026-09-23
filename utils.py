"""Data loading, splits, augmentation and the neighbour sampler."""

import datetime
import os
import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim
import yaml
from torch_geometric.data import Data
from torch_geometric.utils import (dense_to_sparse, from_scipy_sparse_matrix,
                                   is_undirected, to_undirected)





def load_npz(fp):
    data = np.load(fp, allow_pickle=True)

    return data



def make_pyg_graph(x, adj, undirected=True):
    features = torch.from_numpy(x.todense()).float()
    edge_index, _ = from_scipy_sparse_matrix(adj)

    if undirected:
        if not is_undirected(edge_index):
            edge_index = to_undirected(edge_index)

    data = Data(x=features, edge_index=edge_index)

    if undirected:
        assert data.is_undirected()

    return data


def npz_data_to_pyg_graph(data):
    # Load adj matrix
    adj_matrix = sp.csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']), shape=data['adj_shape'])

    # Load feature matrix
    if 'attr_data' in data:
        # Attributes are stored as a sparse CSR matrix
        attr_matrix = sp.csr_matrix((data['attr_data'], data['attr_indices'], data['attr_indptr']),
                                    shape=data['attr_shape'])
    elif 'attr_matrix' in data:
        # Attributes are stored as a (dense) np.ndarray
        attr_matrix = data['attr_matrix']
    else:
        attr_matrix = None

    # Load label matrix
    if 'labels_data' in data:
        # Labels are stored as a CSR matrix
        labels = sp.csr_matrix((data['labels_data'], data['labels_indices'], data['labels_indptr']),
                               shape=data['labels_shape'])
    elif 'labels' in data:
        # Labels are stored as a numpy array
        labels = data['labels']
    else:
        labels = None

    class_idx, class_size = np.unique(labels, return_counts=True)
    class_per = class_size/labels.shape[0]

    graph = make_pyg_graph(attr_matrix, adj_matrix, undirected=True)
    dset_info = {
        'node_names': data.get('node_names'),
        'attr_names': data.get('attr_names'),
        'class_names': data.get('class_names'),
        'metadata': data.get('metadata'),
        'class_idx': class_idx,
        'class_size': class_size,
        'class_per': class_per,
    }

    return graph, labels, dset_info

def load_yaml(fn):
    with open(fn) as fp:
        config = yaml.safe_load(fp)
    return config





class EdgeIndexTuple(tuple):
    """(edge_index, e_id, size) that also allows .to(device) like PyG's Adj."""
    def to(self, device):
        ei, eid, size = self
        return EdgeIndexTuple((ei.to(device), eid, size))


class NeighborSamplerShim:
    def __init__(self, edge_index, node_idx=None, sizes=(25, 10), batch_size=512,
                 shuffle=False, drop_last=False, sim_x=None, sim_topk_frac=0.0,
                 sim_shuffle=False, **kwargs):
        self.sizes = list(sizes)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sim_topk_frac = float(sim_topk_frac) if sim_x is not None else 0.0
        num_nodes = int(edge_index.max()) + 1
        # CSR over incoming edges: for target t, neighbors are sources
        src, dst = edge_index[0], edge_index[1]
        order = torch.argsort(dst)
        src_s, dst_s = src[order], dst[order]
        if sim_x is not None and self.sim_topk_frac > 0:
            # cosine similarity per edge, then stable dst-major / sim-descending
            xn = torch.nn.functional.normalize(sim_x.float(), dim=1)
            sim = torch.empty(src_s.numel(), dtype=xn.dtype)
            step = max(1, 200_000_000 // max(1, xn.size(1)))
            for lo in range(0, src_s.numel(), step):
                hi = min(lo + step, src_s.numel())
                sim[lo:hi] = (xn[src_s[lo:hi]] * xn[dst_s[lo:hi]]).sum(1)
            if sim_shuffle:
                sim = torch.rand_like(sim)
            i = torch.argsort(-sim, stable=True)
            i = i[torch.argsort(dst_s[i], stable=True)]
            src_s = src_s[i]
        self.nbr = src_s.contiguous()
        order = torch.argsort(dst)  # keep downstream ptr/deg computation intact
        counts = torch.bincount(dst, minlength=num_nodes)
        self.ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
        self.deg = counts
        if node_idx is None:
            node_idx = torch.arange(num_nodes)
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)
        self.node_idx = node_idx.long()

    def _sample_hop(self, targets, size):
        """Return (src_global, dst_local) sampled edges for `targets`."""
        deg = self.deg[targets]
        has = deg > 0
        t_loc = torch.arange(targets.size(0))[has]
        t_glob = targets[has]
        d = deg[has]
        k = torch.clamp(d, max=size)
        reps = k.sum()
        dst_local = torch.repeat_interleave(t_loc, k)
        base = torch.repeat_interleave(self.ptr[t_glob], k)
        dmax = torch.repeat_interleave(d, k)
        if self.sim_topk_frac > 0:
            # neighbours are stored sim-descending, so rank r < k_top means
            # "one of the most similar neighbours"; the rest stay uniform.
            rank = (torch.arange(reps) -
                    torch.repeat_interleave(torch.cumsum(k, 0) - k, k))
            k_top = torch.repeat_interleave(
                (k.float() * self.sim_topk_frac).long(), k)
            off_top = torch.minimum(rank, dmax - 1)
            off_rand = (torch.rand(reps) * dmax).long()
            off = torch.where(rank < k_top, off_top, off_rand)
        else:
            off = (torch.rand(reps) * dmax).long()
        src_global = self.nbr[base + off]
        # de-duplicate (src, dst) pairs: sort keys, keep first of each run
        key = src_global * targets.size(0) + dst_local
        key_sorted, perm = torch.sort(key)
        mask = torch.ones_like(key_sorted, dtype=torch.bool)
        mask[1:] = key_sorted[1:] != key_sorted[:-1]
        sel = perm[mask]
        return src_global[sel], dst_local[sel]

    def __iter__(self):
        idx = self.node_idx
        if self.shuffle:
            idx = idx[torch.randperm(idx.size(0))]
        for s in range(0, idx.size(0), self.batch_size):
            seeds = idx[s:s + self.batch_size]
            n_id = seeds.clone()
            adjs = []
            for size in self.sizes:
                targets = n_id
                src_g, dst_l = self._sample_hop(targets, size)
                # order-preserving unique over [n_id, sampled sources]:
                # n_id keeps local ids 0..len-1, new sources get fresh ids.
                all_nodes = torch.cat([n_id, src_g])
                pos = torch.arange(all_nodes.size(0))
                uniq, inv = torch.unique(all_nodes, return_inverse=True)
                first_pos = torch.full((uniq.size(0),), all_nodes.size(0), dtype=torch.long)
                first_pos.scatter_reduce_(0, inv, pos, reduce='amin')
                order = torch.argsort(first_pos)
                rank = torch.empty_like(order)
                rank[order] = torch.arange(order.size(0))
                local = rank[inv]
                src_l = local[n_id.size(0):]
                n_id = all_nodes[first_pos[order]]
                ei = torch.stack([src_l, dst_l])
                adjs.append(EdgeIndexTuple((ei, None, (n_id.size(0), targets.size(0)))))
            adjs.reverse()
            if len(self.sizes) == 1:
                yield seeds.size(0), n_id, adjs[0:1]
            else:
                yield seeds.size(0), n_id, adjs

    def __len__(self):
        return (self.node_idx.size(0) + self.batch_size - 1) // self.batch_size


def get_neighbor_sampler(edge_index, **kwargs):
    try:
        from torch_geometric.loader import NeighborSampler
        return NeighborSampler(edge_index, **kwargs)
    except Exception:
        return NeighborSamplerShim(edge_index, **kwargs)




# ogb / Planetoid are needed for ogbn-* loading; guard so a minimal env still imports.
try:
    from ogb.nodeproppred import PygNodePropPredDataset
except Exception:
    PygNodePropPredDataset = None
try:
    from torch_geometric.datasets import Planetoid
except Exception:
    Planetoid = None

def split_fingerprint(split):
    import hashlib
    h = hashlib.sha256()
    for part in (split["idx_train"], split["idx_val"], split["idx_test"]["all"],
                 split["idx_test"]["unknown"]):
        a = np.asarray(part.cpu() if torch.is_tensor(part) else part).astype(np.int64)
        h.update(np.sort(a).tobytes())
    return h.hexdigest()[:16]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)
    return seed

def merge_configs(cmd_args, yaml_args):
    for key, value in cmd_args.items():
        if value is not None:
            keys = key.split('.')
            temp_yaml = yaml_args
            for k in keys[:-1]:
                temp_yaml = temp_yaml.setdefault(k, {})
            temp_yaml[keys[-1]] = value
    return yaml_args

def aggregate_rotations(rotation_results, which='best'):
    keys = ['auroc_all', 'aupr_all', 'auroc_unknown', 'aupr_unknown']
    out = {}
    for k in keys:
        vals = [r[which][k] for r in rotation_results if which in r and k in r[which]]
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out

def get_optimiser(name, param, lr, weight_decay):
    if name.lower() == 'adam':
        optimiser = optim.Adam(param, lr=lr, weight_decay=weight_decay)
    elif name.lower() == 'adamw':
        optimiser = optim.AdamW(param, lr=lr, weight_decay=weight_decay)
    elif name.lower() == 'sgd':
        optimiser = optim.SGD(param, lr=lr)
    elif name.lower() == 'rmsprop':
        optimiser = optim.RMSprop(param, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError("Optimiser function not supported!")
    return optimiser

def log(message, data_name=None, level="INFO", log_dir="logs"):

    os.makedirs(log_dir, exist_ok=True)

    if not hasattr(log, "filename"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{data_name}_{timestamp}.log"
        log.filename = os.path.join(log_dir, filename)

    log_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_entry = f"[{log_time}] [{level:<7}] {message}\n"

    with open(log.filename, "a", encoding="utf-8") as f:
        f.write(log_entry)

    return log.filename

def _allow_pyg_globals():
    try:
        import torch.serialization as _ts
        safe = []
        try:
            from torch_geometric.data.data import (Data, DataEdgeAttr,
                                                   DataTensorAttr)
            safe += [Data, DataEdgeAttr, DataTensorAttr]
        except Exception:
            pass
        try:
            from torch_geometric.data.storage import (BaseStorage, EdgeStorage,
                                                      GlobalStorage, NodeStorage)
            safe += [BaseStorage, EdgeStorage, GlobalStorage, NodeStorage]
        except Exception:
            pass
        if safe and hasattr(_ts, "add_safe_globals"):
            _ts.add_safe_globals(safe)
    except Exception:
        pass


def load_data(data_name):
    if data_name in ['photo', 'computers', 'cs']:
        data = np.load(f'data/{data_name}/{data_name}.npz', allow_pickle=True)
        graph, labels, dset_info = npz_data_to_pyg_graph(data)
    elif data_name in ['ogbn-arxiv']:
        if PygNodePropPredDataset is None:
            raise ImportError("ogbn-* needs the 'ogb' package: pip install ogb")
        _allow_pyg_globals()
        data = PygNodePropPredDataset(name=data_name, root='data')
        graph = data[0]
        labels = graph.y.long().squeeze(-1)
        class_idx, class_size = torch.unique(labels, return_counts=True)
        class_per = class_size.float() / labels.shape[0]
        dset_info = {
            'node_names': None,
            'attr_names': None,
            'class_names': None,
            'metadata': None,
            'class_idx': np.array(class_idx),
            'class_size': np.array(class_size),
            'class_per': np.array(class_per),
        }
    elif data_name in ['ogbn-mag']:
        if PygNodePropPredDataset is None:
            raise ImportError("ogbn-* needs the 'ogb' package: pip install ogb")
        _allow_pyg_globals()
        data = PygNodePropPredDataset(name=data_name, root='data')
        paper_x = data[0]['x_dict']['paper']
        paper_y = data[0]['y_dict']['paper']
        edge_index = data[0]['edge_index_dict']['paper', 'cites', 'paper']
        
        graph = Data(x=paper_x, y=paper_y, edge_index=edge_index)
        labels = graph.y.long().squeeze(-1)
        class_idx, class_size = torch.unique(labels, return_counts=True)
        class_per = class_size.float() / labels.shape[0]
        dset_info = {
            'node_names': None,
            'attr_names': None,
            'class_names': None,
            'metadata': None,
            'class_idx': np.array(class_idx),
            'class_size': np.array(class_size),
            'class_per': np.array(class_per),
        }
    elif data_name in ['tfinance', 'yelp', 'amazon']:
        _p = f'data/{data_name}/{data_name}'
        if not os.path.exists(_p):
            _p = _p + '.zip'
        graph = torch.load(_p, weights_only=False)
        labels = graph.y.long().squeeze(-1)
        class_idx, class_size = torch.unique(labels, return_counts=True)
        class_per = class_size.float() / labels.shape[0]
        dset_info = {
            'node_names': None,
            'attr_names': None,
            'class_names': None,
            'metadata': None,
            'class_idx': np.array(class_idx),
            'class_size': np.array(class_size),
            'class_per': np.array(class_per),
        }
    elif data_name in ['pubmed', 'cora', 'cityseer']:
        data_name = 'cora'
        data = Planetoid(root=f'data/{data_name}', name=data_name)
        graph = data[0]
        labels = graph.y.long().squeeze(-1)
        class_idx, class_size = torch.unique(labels, return_counts=True)
        class_per = class_size.float() / labels.shape[0]
        dset_info = {
            'node_names': None,
            'attr_names': None,
            'class_names': None,
            'metadata': None,
            'class_idx': np.array(class_idx),
            'class_size': np.array(class_size),
            'class_per': np.array(class_per),
        }
    else:
        raise ValueError(f"Unsupported dataset: {data_name}")
    return graph, labels, dset_info

def ad_split_num(labels, args, class_info):
    known_anomaly = class_info['known_anomaly']
    unknown_anomaly_classes = class_info['unknown_anomaly']
    normal_classes = class_info['normal']
    print(f'Normal classes: {normal_classes} in {args.dataname}')

    known_anomaly_idx = np.where(labels==known_anomaly)[0].flatten()
    normal_idx = select_class_idx_in_list(labels, normal_classes) # all normal nodes idx
    # binary datasets (yelp/tfinance/amazon) have no unseen anomaly classes
    unknown_anomaly_idx = (select_class_idx_in_list(labels, unknown_anomaly_classes)
                           if len(unknown_anomaly_classes) > 0
                           else np.array([], dtype=np.int64))

    normal_train, normal_val, normal_test = random_split(normal_idx, args.train_normal_ratio, args.val_normal_ratio)
    known_anomaly_train, known_anomaly_val, known_anomaly_test = num_split(known_anomaly_idx, args.train_anormaly_num, args.val_anormaly_num)
    unknown_anomaly_test = unknown_anomaly_idx

    train_idx = np.hstack((normal_train, known_anomaly_train))
    val_idx = np.hstack((normal_val, known_anomaly_val))
    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)


    test_idx = {
        'all': np.hstack((normal_test, known_anomaly_test, unknown_anomaly_test)),
        'known': np.hstack((normal_test, known_anomaly_test)),
        'unknown': np.hstack((normal_test, unknown_anomaly_test)),
        'normal': normal_test,
        'known_only': known_anomaly_test,
        'unknown_only': unknown_anomaly_test,
    }

    split_info = {
        'idx_train': train_idx,
        'idx_normal_train': normal_train,
        'idx_anomaly_train': known_anomaly_train,
        'idx_val': val_idx,
        'idx_test': test_idx
    }

    return split_info

def select_class_idx_in_list(labels, classes):
    node_idx = None
    for i in classes:
        cur_idx = np.where(labels == i)[0].flatten()
        node_idx = cur_idx if node_idx is None else np.hstack((node_idx, cur_idx))
    return node_idx

def random_split(idx, train_ratio, valid_ratio):
    n_train = int(idx.shape[0] * train_ratio)
    n_valid = int(idx.shape[0] * valid_ratio)
    randperm = torch.randperm(idx.shape[0])
    return idx[randperm[:n_train]], idx[randperm[n_train:n_train+n_valid]], idx[randperm[n_train+n_valid:]]


def num_split(idx, n_train, n_val):
    randperm = torch.randperm(idx.shape[0])
    return idx[randperm[:n_train]], idx[randperm[n_train:n_train+n_val]], idx[randperm[n_train+n_val:]]

class NodeFeatureAugmentor:
    def __init__(self, augmentation_config):
        self.config = augmentation_config

    def _standardize(self, x):
        self.orig_mean = x.mean(dim=0, keepdim=True)
        self.orig_std = x.std(dim=0, keepdim=True) + 1e-8

        if torch.allclose(self.orig_mean, torch.zeros_like(self.orig_mean), atol=1e-3) and \
                torch.allclose(self.orig_std, torch.ones_like(self.orig_std), atol=1e-2):
            return x.clone()
        return (x - self.orig_mean) / self.orig_std

    def _restore(self, x_normalized):
        return x_normalized * self.orig_std + self.orig_mean

    def _gaussian_noise(self, x, sigma=0.05):
        noise = torch.randn_like(x) * sigma
        return x + noise

    def _feature_mask(self, x, mask_prob=0.3):
        mask = torch.rand_like(x) > mask_prob
        return x * mask.float()

    def _feature_mixup(self, x, alpha=0.2):
        idx = torch.randperm(x.size(0))
        lam = np.random.beta(alpha, alpha)
        return lam * x + (1 - lam) * x[idx]

    def _scaling_jitter(self, x, gamma=0.1):
        scale = torch.FloatTensor(x.size(1)).uniform_(1 - gamma, 1 + gamma).to(x.device)
        return x * scale

    def augment(self, x):
        x_norm = self._standardize(x)

        if 'noise' in self.config:
            x_norm = self._gaussian_noise(x_norm, **self.config['noise'])

        if 'mask' in self.config:
            x_norm = self._feature_mask(x_norm, **self.config['mask'])

        if 'mixup' in self.config:
            x_norm = self._feature_mixup(x_norm, **self.config['mixup'])

        if 'scaling' in self.config:
            x_norm = self._scaling_jitter(x_norm, **self.config['scaling'])

        return self._restore(x_norm)







class DeviceFeatures:
    def __init__(self, x, device):
        object.__setattr__(self, "_x", x.to(device))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            if idx.device != self._x.device:
                idx = idx.to(self._x.device, non_blocking=True)
        elif isinstance(idx, (list, tuple, np.ndarray)):
            idx = torch.as_tensor(np.asarray(idx), device=self._x.device)
        return self._x[idx]

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_x"), name)

    def __len__(self):
        return len(self._x)


class _GraphView:
    def __init__(self, graph, x):
        object.__setattr__(self, "_graph", graph)
        object.__setattr__(self, "x", x)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_graph"), name)


def features_to_device(graph, device, max_gb=4.0, enabled=None):
    if device is None or str(device) == "cpu" or not torch.cuda.is_available():
        return graph
    x = getattr(graph, "x", None)
    if x is None or isinstance(x, DeviceFeatures):
        return graph
    gb = x.numel() * x.element_size() / 1e9
    if enabled is False or (enabled is None and gb > max_gb):
        return graph
    try:
        return _GraphView(graph, DeviceFeatures(x, device))
    except RuntimeError:
        return graph
