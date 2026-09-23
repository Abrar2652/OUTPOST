"""Training / evaluation loops."""

import os

import torch
import numpy as np

from utils import get_optimiser
from collections import Counter
from copy import deepcopy
from sklearn.metrics import roc_auc_score, average_precision_score


def node_context(graph, dataname, cache=True):
    """Label-free per-node scatter context for HopMix."""
    import torch.nn.functional as F
    path = f"data/{dataname.replace('-', '_')}/node_ctx.pt"
    if cache and os.path.exists(path):
        return torch.load(path)
    x = graph.x.float()
    N = x.size(0)
    src, dst = graph.edge_index[0], graph.edge_index[1]
    keep = src != dst
    src, dst = src[keep], dst[keep]
    deg = torch.bincount(dst, minlength=N).clamp(min=1).float()
    nbr_sum = torch.zeros(N, x.size(1)).index_add_(0, dst, x[src])
    nbr_mean = nbr_sum / deg.unsqueeze(1)
    feat_dissim = 1.0 - F.cosine_similarity(x, nbr_mean, dim=1, eps=1e-8)
    ctx = torch.stack([feat_dissim, torch.log1p(deg)], dim=1)
    ctx = (ctx - ctx.mean(0, keepdim=True)) / (ctx.std(0, keepdim=True) + 1e-8)
    if cache:
        try:
            torch.save(ctx, path)
        except Exception:
            pass
    return ctx


def train_outpost_v4(split_info, labels, graph, args, anomaly_info, logger):
    import torch.nn.functional as F
    from model import OUTPOST_V4
    from utils import get_neighbor_sampler
    from utils import NodeFeatureAugmentor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    rest_labels = np.zeros(labels_np.shape[0])
    for i in anomaly_info['all_anomaly']:
        rest_labels[np.where(labels_np == i)[0]] = 1
    rest_labels_t = torch.from_numpy(rest_labels).float().to(device)
    y_np = rest_labels

    idx_train = torch.as_tensor(np.asarray(split_info['idx_train'])).long()
    idx_val = np.asarray(split_info['idx_val'])
    idx_normal_train = torch.as_tensor(np.asarray(split_info['idx_normal_train'])).long()
    idx_anomaly_train = torch.as_tensor(np.asarray(split_info['idx_anomaly_train'])).long()
    pool = torch.as_tensor(np.asarray(split_info['idx_test']['all'])).long()
    cal_n = idx_val[y_np[idx_val] == 0]

    g = lambda k, d: (args.get(k) if args.get(k) is not None else d)
    d0 = int(args.input_dim)
    h = int(g('hidden_dim', 64))
    K_p = int(g('K_p', 8))
    n_layers = int(g('n_layers', 2))
    dropout = float(g('drop_out', 0.5))
    num_epochs = int(args.num_epochs)
    warmup = int(g('warmup_epochs', 5))
    sizes = list(g('sampling_sizes', [25, 10]))
    bs = int(g('batch_size', 512))
    margin = float(g('margin', 0.1))

    use_pl = bool(g('use_pl', True))
    use_conformal = bool(g('use_conformal', True))
    use_mixup = bool(g('use_mixup', True))
    use_halo = bool(g('use_halo', True))
    use_atlas_loss = bool(g('use_atlas_loss', False))
    use_atlas_gate = bool(g('use_atlas_gate', True))
    use_fview_gate = bool(g('use_fview_gate', False))
    alpha_f = float(g('alpha_f', 0.2))
    lam_f = float(g('lambda_f', 1.0))

    lam_un = float(g('lambda_un', 0.5))
    lam_mix = float(g('lambda_mixup', 0.1))
    lam_halo = float(g('lambda_halo', 0.1))
    lam_atlas = float(g('lambda_atlas', 0.5))
    tau_plus_fix = float(g('tau_plus', 0.95))
    tau_minus = float(g('tau_minus', 0.05))
    alpha_plus = float(g('alpha_plus', 0.05))
    atlas_gate_q = float(g('atlas_gate_q', 0.9))
    eta_min = float(g('eta_min', 1.2))
    eta_max = float(g('eta_max', 2.0))

    use_hybrid = bool(g('use_hybrid', False))
    if use_hybrid and use_fview_gate:
        raise ValueError(
            "use_hybrid and use_fview_gate are mutually exclusive (both are "
            "separately-tested view ablations; V4HM provides fused_logit, not "
            "fview_logit). Set exactly one to true.")
    ctx_all = None
    if use_hybrid:
        from model import OUTPOST_V4HM
        ctx_all = node_context(graph, args.dataname).to(device)
        model = OUTPOST_V4HM(d0, h, K_p, n_layers=n_layers, dropout=dropout,
                             tau_mu=float(g('tau_mu', 0.05)),
                             alpha_r=float(g('alpha_r', 0.1)),
                             n_ctx=ctx_all.size(1),
                             fview_hidden=int(g('fview_hidden', 64))).to(device)
    elif use_fview_gate:
        from model import OUTPOST_V4SG
        model = OUTPOST_V4SG(d0, h, K_p, n_layers=n_layers, dropout=dropout,
                             tau_mu=float(g('tau_mu', 0.05)),
                             alpha_r=float(g('alpha_r', 0.1))).to(device)
    else:
        model = OUTPOST_V4(d0, h, K_p, n_layers=n_layers, dropout=dropout,
                           tau_mu=float(g('tau_mu', 0.05)),
                           alpha_r=float(g('alpha_r', 0.1))).to(device)
    opt = get_optimiser(args.optimiser, model.main_parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)
    bce = torch.nn.BCEWithLogitsLoss()

    sim_frac = float(g('sim_topk_frac', 0.0))
    sim_kw = ({'sim_x': graph.x, 'sim_topk_frac': sim_frac,
               'sim_shuffle': bool(g('sim_shuffle', False))}
              if sim_frac > 0 else {})
    loader_tr = get_neighbor_sampler(graph.edge_index, node_idx=idx_train,
                                     sizes=sizes, batch_size=bs, shuffle=False,
                                     **sim_kw)
    loader_un = get_neighbor_sampler(graph.edge_index, node_idx=pool,
                                     sizes=sizes, batch_size=bs, shuffle=False,
                                     **sim_kw)
    loader_eval = get_neighbor_sampler(graph.edge_index, node_idx=None,
                                       sizes=sizes,
                                       batch_size=bs * int(g('eval_batch_mult', 4)),
                                       shuffle=False, **sim_kw)
    from utils import features_to_device
    graph = features_to_device(graph, device, enabled=g('features_on_gpu', None))

    weak_aug = NodeFeatureAugmentor({'noise': {'sigma': 0.02}, 'mask': {'mask_prob': 0.1}})
    strong_aug = NodeFeatureAugmentor({'noise': {'sigma': 0.02}, 'mask': {'mask_prob': 0.1},
                                       'mixup': {'alpha': 0.1}, 'scaling': {'gamma': 0.1}})

    selected_label = torch.full((graph.num_nodes,), -1, dtype=torch.long, device=device)
    max_counter = {0: 0, 1: 0}

    n_tr = idx_train.size(0)
    is_anom_tr = torch.zeros(n_tr, dtype=torch.bool)
    is_anom_tr[torch.isin(idx_train, idx_anomaly_train)] = True
    is_norm_tr = ~is_anom_tr

    best = {'auroc_all': 0.0, 'aupr_all': 0.0, 'auroc_unknown': 0.0, 'aupr_unknown': 0.0}
    val_sel = {'val_auc': -1.0}
    final = {}
    tau_plus_cur = tau_plus_fix
    record_scores = bool(g('record_scores', False))
    score_log = [] if record_scores else None

    for epoch in range(num_epochs):
        model.train()

        zs, lg = [], []
        w_log = []
        for bsz, n_id, adjs in loader_tr:
            x = graph.x[n_id].to(device).float()
            adjs = [a.to(device) for a in adjs]
            if use_hybrid:
                tgt = torch.as_tensor(n_id[:bsz]).to(device)
                fl, w, z = model.fused_logit(x, adjs, ctx_all[tgt])
                zs.append(z); lg.append(fl); w_log.append(w.detach())
            else:
                zs.append(model.embed(x, adjs))
        z_tr = torch.cat(zs, 0)
        logits_tr = torch.cat(lg, 0) if use_hybrid else model.pdh_logit(z_tr)
        loss = bce(logits_tr, rest_labels_t[idx_train])

        n_veto = 0
        if use_fview_gate:
            x_tr = graph.x[idx_train].to(device).float()
            loss = loss + lam_f * bce(model.fview_logit(x_tr), rest_labels_t[idx_train])

        if epoch == warmup - 1 and (use_atlas_gate or use_atlas_loss):
            with torch.no_grad():
                model.atlas.init_kmeans(F.normalize(z_tr[is_norm_tr].detach(), dim=-1))

        if epoch >= warmup:
            z_n = z_tr[is_norm_tr]
            z_a = z_tr[is_anom_tr]
            if float(model.atlas.initialized) > 0:
                model.atlas.ema_update(F.normalize(z_n.detach(), dim=-1))

            if use_pl:
                cnt = {c: int((selected_label == c).sum()) for c in (0, 1)}
                acc = {}
                for c in (0, 1):
                    max_counter[c] = max(max_counter[c], cnt[c])
                    acc[c] = cnt[c] / max_counter[c] if max_counter[c] > 0 else 0.0
                thr_a = tau_plus_cur * (acc[1] / (2 - acc[1]))
                thr_n = 2 * tau_minus - tau_minus * (acc[0] / (2 - acc[0]))

                tau_f = None
                if use_fview_gate and len(cal_n) > 0:
                    with torch.no_grad():
                        pf_cal = torch.sigmoid(model.fview_logit(
                            graph.x[torch.as_tensor(cal_n)].to(device).float()))
                        k_f = min(int(np.ceil((pf_cal.numel() + 1) * (1 - alpha_f))) - 1,
                                  pf_cal.numel() - 1)
                        tau_f = pf_cal.sort().values[max(k_f, 0)]

                l_un_sum, n_un = 0.0, 0
                for bsz, n_id, adjs in loader_un:
                    xw = weak_aug.augment(graph.x[n_id].to(device).float())
                    xs = strong_aug.augment(graph.x[n_id].to(device).float())
                    adjs = [a.to(device) for a in adjs]
                    tgt_b = torch.as_tensor(n_id[:bsz]).to(device)
                    with torch.no_grad():
                        if use_hybrid:
                            fl_w, _, z_w = model.fused_logit(xw, adjs, ctx_all[tgt_b])
                            p_w = torch.sigmoid(fl_w)
                        else:
                            z_w = model.embed(xw, adjs)
                            p_w = torch.sigmoid(model.pdh_logit(z_w))
                    pred = (p_w >= 0.5).float()
                    mask = torch.where(pred == 1, (p_w >= thr_a).float(), (p_w <= thr_n).float())
                    if use_atlas_gate and float(model.atlas.initialized) > 0:
                        d_at = model.d_atlas(z_w)
                        gate_ref = model.d_atlas(z_n.detach())
                        gate = torch.quantile(gate_ref, atlas_gate_q) + 1e-6
                        mask = torch.where(pred == 0, mask * (d_at <= gate).float(), mask)
                    if tau_f is not None:
                        with torch.no_grad():
                            p_f = torch.sigmoid(model.fview_logit(
                                graph.x[torch.as_tensor(n_id[:bsz])].to(device).float()))
                        veto = (pred == 0) & (mask > 0) & (p_f > tau_f)
                        n_veto += int(veto.sum())
                        mask = mask * (~veto).float()
                    if use_hybrid:
                        logits_s = model.fused_logit(xs, adjs, ctx_all[tgt_b])[0]
                    else:
                        logits_s = model.pdh_logit(model.embed(xs, adjs))
                    l_un_sum = l_un_sum + (F.binary_cross_entropy_with_logits(
                        logits_s, pred, reduction='none') * mask).sum()
                    n_un += bsz
                    with torch.no_grad():
                        a_probs = p_w[p_w >= 0.5]; np_ = p_w[p_w < 0.5]
                        a_cut = a_probs.quantile(0.95) if a_probs.numel() > 0 else torch.tensor(1.0, device=device)
                        n_cut = np_.quantile(0.05) if np_.numel() > 0 else torch.tensor(0.0, device=device)
                        sel_mem = (p_w >= a_cut) | (p_w <= n_cut)
                        b_nid = torch.as_tensor(n_id[:bsz]).to(device)
                        selected_label[b_nid[sel_mem]] = pred[sel_mem].long()
                loss = loss + lam_un * (l_un_sum / max(n_un, 1))

            z_mix = None
            if use_mixup and z_a.size(0) > 1:
                with torch.no_grad():
                    zna = F.normalize(z_a, dim=-1)
                    sim = zna @ zna.t()
                    sim.fill_diagonal_(-float('inf'))
                    alpha_w = F.softmax(sim, dim=1)
                z_mix = alpha_w @ z_a
                loss = loss + lam_mix * F.binary_cross_entropy_with_logits(
                    model.pdh_logit(z_mix), torch.ones(z_mix.size(0), device=device))
            if use_halo and z_a.size(0) > 0:
                src = z_mix if z_mix is not None else z_a
                zbar = z_n.mean(0, keepdim=True).detach()
                eta = torch.empty(src.size(0), 1, device=device).uniform_(eta_min, eta_max)
                halo = zbar + eta * (src - zbar)
                loss = loss + lam_halo * F.binary_cross_entropy_with_logits(
                    model.pdh_logit(halo), torch.ones(halo.size(0), device=device))
            if use_atlas_loss and float(model.atlas.initialized) > 0:
                d_n = model.d_atlas(z_n)
                d_a = model.d_atlas(z_a)
                loss = loss + lam_atlas * (d_n.mean() + F.relu(margin - d_a).mean())

        opt.zero_grad()
        loss.backward()
        opt.step()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        m = eval_outpost_v4(model, split_info, y_np, graph, args, device, sizes,
                            idx_val, ctx_all=ctx_all if use_hybrid else None,
                            sim_kw=sim_kw, loader=loader_eval)
        if use_conformal and len(cal_n) > 0:
            pc = np.sort(m['scores_sig'][cal_n])
            k_idx = min(int(np.ceil((len(pc) + 1) * (1 - alpha_plus))) - 1, len(pc) - 1)
            tau_plus_cur = float(np.clip(pc[max(k_idx, 0)], 0.5, 0.995))
        else:
            tau_plus_cur = tau_plus_fix

        if record_scores:
            score_log.append(m['scores_sig'].astype(np.float32))
        for k in best:
            if m[k] > best[k]:
                best[k] = m[k]
        if m['val_auc'] > val_sel['val_auc']:
            val_sel = {'val_auc': m['val_auc'], **{k: m[k] for k in best}}
        final = {k: m[k] for k in best}
        print(f"[OUTPOST-v4] epoch {epoch}: test-all ROC {m['auroc_all']:.4f} "
              f"PR {m['aupr_all']:.4f} | unseen ROC {m['auroc_unknown']:.4f} "
              f"PR {m['aupr_unknown']:.4f} | valAUC {m['val_auc']:.4f} | tau+ {tau_plus_cur:.3f}"
              + (f" | veto {n_veto}" if use_fview_gate else "")
              + (f" | w {m['mix_w_mean']:.3f} (a {m['mix_w_anom']:.3f} / "
                 f"n {m['mix_w_norm']:.3f})" if 'mix_w_mean' in m else ""))
        logger.info(f"[OUTPOST-v4] epoch {epoch} cur="
                    f"{ {k: m[k] for k in ('auroc_all','aupr_all','auroc_unknown','aupr_unknown','val_auc')} } best={best}")

    out = {'best': best, 'val_selected': val_sel, 'final': final,
           'n_params': sum(p.numel() for p in model.parameters())}
    if record_scores:
        out['scores'] = np.stack(score_log)
    return out


def eval_outpost_v4(model, split_info, y_np, graph, args, device, sizes, idx_val,
                    ctx_all=None, sim_kw=None, loader=None):
    from utils import get_neighbor_sampler
    model.eval()
    bs = int(args.batch_size) * int(args.get('eval_batch_mult') or 4)
    if loader is None:
        loader = get_neighbor_sampler(graph.edge_index, node_idx=None,
                                      sizes=sizes, batch_size=bs, shuffle=False,
                                      **(sim_kw or {}))
    outs, ws = [], []
    with torch.no_grad():
        for bsz, n_id, adjs in loader:
            x = graph.x[n_id].to(device).float()
            adjs = [a.to(device) for a in adjs]
            if ctx_all is not None:
                tgt = torch.as_tensor(n_id[:bsz]).to(device)
                fl, w, _ = model.fused_logit(x, adjs, ctx_all[tgt])
                outs.append(torch.sigmoid(fl).cpu()); ws.append(w.cpu())
            else:
                outs.append(torch.sigmoid(model(x, adjs)).cpu())
    s = torch.cat(outs).numpy()
    w_all = torch.cat(ws).numpy() if ws else None

    def auc(idx):
        idx = np.asarray(idx)
        yt, yp = y_np[idx], s[idx]
        if len(np.unique(yt)) < 2:
            return 0.5, float(np.mean(yt))
        return roc_auc_score(yt, yp), average_precision_score(yt, yp)

    roc_all, pr_all = auc(split_info['idx_test']['all'])
    roc_unk, pr_unk = auc(split_info['idx_test']['unknown'])
    val_auc = auc(idx_val)[0]
    out = {'auroc_all': roc_all, 'aupr_all': pr_all,
           'auroc_unknown': roc_unk, 'aupr_unknown': pr_unk,
           'val_auc': val_auc, 'scores_sig': s}
    if w_all is not None:
        out['mix_w_mean'] = float(w_all.mean())
        out['mix_w_anom'] = float(w_all[y_np == 1].mean())
        out['mix_w_norm'] = float(w_all[y_np == 0].mean())
    return out


