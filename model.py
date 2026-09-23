"""OUTPOST model definitions."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, inits
from torch_geometric.utils import dropout_adj




class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        return self.layers(x)
    
    def get_last_layer(self, x):
        return self.layers[1](self.layers[0](x))


class CLF(nn.Module):
    def __init__(self, int_dim, out_dim, bias=False):
        super(CLF, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(int_dim, int_dim),
            nn.ReLU(),
            nn.Linear(int_dim, out_dim),
        )

    def forward(self, x):
        return self.layers(x)


class Proj(nn.Module):
    def __init__(self, in_dim, out_dim, norm, scaler, bias=True):
        super(Proj, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim, bias=bias)
        self.norm, self.scaler = norm, scaler
        if norm and scaler is not None:
            self.scale = torch.nn.Parameter(torch.Tensor([scaler]), requires_grad=True)

    def forward(self, x):
        x = self.layer(x)
        if self.norm and self.scaler is None:
            x = torch.nn.functional.normalize(x) * self.scale

        return x


class MSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, output_type="logit", adj_dropout=0.0):
        super(GraphSAGE, self).__init__()

        self.num_layers = num_layers
        self.drop_out = dropout
        self.output_type = output_type

        self.convs = torch.nn.ModuleList()

        self.normalise = False

        self.ebd_dim = out_channels

        self.adj_dropout = adj_dropout

        if self.num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=self.normalise))

            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=self.normalise))

            self.convs.append(SAGEConv(hidden_channels, out_channels, normalize=self.normalise))

    def reset_parameters(self):
        for conv in self.convs:
            inits.kaiming_uniform(conv.lin_l.weight, conv.lin_l.in_channels, a=math.sqrt(5))
            inits.kaiming_uniform(conv.lin_r.weight, conv.lin_r.in_channels, a=math.sqrt(5))
            inits.zeros(conv.lin_l.in_channels)
            inits.zeros(conv.lin_r.in_channels)

    def forward(self, x, adjs, pp_matrix=None):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]

            if self.adj_dropout > 0:
                edge_index = dropout_adj(edge_index, p=self.adj_dropout, force_undirected=True, training=self.training)[0]

            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.drop_out, training=self.training)

        if self.output_type == "ebd":
            return x

        return x.log_softmax(dim=-1).float()




def geodesic_dist(z, mu):
    """Pairwise geodesic (arc) distance between z [B,d] and mu [Kp,d] -> [B,Kp]."""
    cos = z @ mu.t()
    cos = cos.clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.arccos(cos)


class PrototypeAtlas(nn.Module):
    def __init__(self, d, K_p, tau_mu=0.05, alpha_r=0.1, kmeans_iters=20):
        super().__init__()
        self.K_p = K_p
        self.tau_mu = tau_mu
        self.alpha_r = alpha_r
        self.kmeans_iters = kmeans_iters
        self.register_buffer('mu', F.normalize(torch.randn(K_p, d), dim=-1))
        self.register_buffer('r', torch.full((K_p,), 0.3))
        self.register_buffer('initialized', torch.zeros(1))

    @torch.no_grad()
    def init_kmeans(self, z_normal):
        """Spherical k-means on labeled-normal embeddings."""
        z = F.normalize(z_normal, dim=-1)
        n = z.size(0)
        if n < self.K_p:
            self.mu.copy_(F.normalize(
                z[torch.randint(0, n, (self.K_p,))], dim=-1))
        else:
            idx = torch.randperm(n)[:self.K_p]
            mu = F.normalize(z[idx], dim=-1)
            for _ in range(self.kmeans_iters):
                assign = (z @ mu.t()).argmax(dim=1)
                new = mu.clone()
                for j in range(self.K_p):
                    m = z[assign == j]
                    if m.size(0) > 0:
                        new[j] = F.normalize(m.mean(0), dim=-1)
                if torch.allclose(new, mu, atol=1e-5):
                    mu = new
                    break
                mu = new
            self.mu.copy_(mu)
        assign = (z @ self.mu.t()).argmax(dim=1)
        dg = geodesic_dist(z, self.mu)
        for j in range(self.K_p):
            dj = dg[assign == j, j]
            if dj.numel() > 0:
                self.r[j] = torch.quantile(dj, 1 - self.alpha_r).clamp(0.05, 1.5)
        self.initialized.fill_(1.0)

    @torch.no_grad()
    def ema_update(self, z_normal):
        z = F.normalize(z_normal, dim=-1)
        assign = (z @ self.mu.t()).argmax(dim=1)
        for j in range(self.K_p):
            m = z[assign == j]
            if m.size(0) > 0:
                upd = (1 - self.tau_mu) * self.mu[j] + self.tau_mu * m.mean(0)
                self.mu[j] = F.normalize(upd, dim=0)
        dg = geodesic_dist(z, self.mu)
        for j in range(self.K_p):
            dj = dg[assign == j, j]
            if dj.numel() > 0:
                target = torch.quantile(dj, 1 - self.alpha_r).clamp(0.05, 1.5)
                self.r[j] = (1 - self.tau_mu) * self.r[j] + self.tau_mu * target

    def d_atlas(self, z):
        """min_j [ d_g(z, mu_j) - r_j ]_+ ; differentiable in z, stop-grad on mu,r."""
        dg = geodesic_dist(z, self.mu.detach())
        d = (dg - self.r.detach().unsqueeze(0)).clamp(min=0.0)
        return d.min(dim=1).values

    def assign(self, z):
        return (F.normalize(z, dim=-1) @ self.mu.t()).argmax(dim=1)




def _get(args, key, default):
    v = args.get(key) if hasattr(args, 'get') else getattr(args, key, None)
    if v is None or (hasattr(v, '__len__') and len(v) == 0 and not isinstance(v, (str,))):
        return default
    return v


class OUTPOST_V4(nn.Module):
    """OUTPOST v4."""

    def __init__(self, d0, h, K_p, n_layers=2, dropout=0.5, tau_mu=0.05, alpha_r=0.1):
        super().__init__()
        self.encoder = GraphSAGE(d0, h, h, n_layers, dropout, output_type="ebds")
        self.proj = MLP(h, h, h)
        self.clf = MLP(h, 32, 1)
        self.atlas = PrototypeAtlas(h, K_p, tau_mu=tau_mu, alpha_r=alpha_r)

    def embed(self, x, adjs):
        return self.proj(self.encoder(x, adjs))

    def forward(self, x, adjs):
        return self.clf(self.embed(x, adjs)).squeeze(-1)

    def pdh_logit(self, z):
        return self.clf(z).squeeze(-1)

    def d_atlas(self, z):
        return self.atlas.d_atlas(F.normalize(z, dim=-1, eps=1e-8))

    def main_parameters(self):
        return (list(self.encoder.parameters()) + list(self.proj.parameters())
                + list(self.clf.parameters()))


class OUTPOST_V4HM(OUTPOST_V4):
    """v4 + HopMix: per-node adaptive fusion of a propagation-free view."""

    def __init__(self, d0, h, K_p, n_layers=2, dropout=0.5, tau_mu=0.05,
                 alpha_r=0.1, n_ctx=2, fview_hidden=64):
        super().__init__(d0, h, K_p, n_layers=n_layers, dropout=dropout,
                         tau_mu=tau_mu, alpha_r=alpha_r)
        self.fview = MLP(d0, fview_hidden, 1)
        self.mixer = nn.Sequential(nn.Linear(n_ctx, 8), nn.ReLU(), nn.Linear(8, 1))
        nn.init.constant_(self.mixer[2].bias, -1.0)

    def mix_weight(self, ctx):
        """w in (0,1): how much to trust the propagation-free view."""
        return torch.sigmoid(self.mixer(ctx)).squeeze(-1)

    def fused_logit(self, x_raw, adjs, ctx):
        """x_raw: [n_id, d0] batch features (targets first). ctx: [bs, n_ctx]."""
        z = self.embed(x_raw, adjs)
        s_gnn = self.clf(z).squeeze(-1)
        bs = s_gnn.size(0)
        s_mlp = self.fview(x_raw[:bs]).squeeze(-1)
        w = self.mix_weight(ctx)
        return (1.0 - w) * s_gnn + w * s_mlp, w, z

    def main_parameters(self):
        return (super().main_parameters() + list(self.fview.parameters())
                + list(self.mixer.parameters()))


class OUTPOST_V4SG(OUTPOST_V4):
    """v4 + Spectral Gate: adds a propagation-free feature view (F-view)."""

    def __init__(self, d0, h, K_p, n_layers=2, dropout=0.5, tau_mu=0.05, alpha_r=0.1):
        super().__init__(d0, h, K_p, n_layers=n_layers, dropout=dropout,
                         tau_mu=tau_mu, alpha_r=alpha_r)
        self.fview = MLP(d0, 128, 1)

    def fview_logit(self, x):
        return self.fview(x).squeeze(-1)

    def main_parameters(self):
        return super().main_parameters() + list(self.fview.parameters())


