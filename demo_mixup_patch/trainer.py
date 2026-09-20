"""Training / evaluation loops.

Two paths only:
  train()             the DEMO baseline (reference implementation, unchanged)
  train_outpost_v4()  OUTPOST - the method reported in the paper

The v1-v3 OUTPOST research iterations were removed for clarity; they are in git
history and none of the reported numbers depend on them.

OUTPOST ablation flags (all read from the dataset config, see config.json):
  use_pl          conformal pseudo-labelling            (default on)
  use_conformal   conformal vs fixed 0.95 threshold     (default on)
  use_mixup/halo  anomaly synthesis                     (per-dataset)
  use_atlas_gate  geometric veto on pseudo-normals      (default on)
  use_fview_gate  spectral gate     - tested, rejected  (default off)
  use_hybrid      HopMix fusion     - tested, rejected  (default off)
  sim_topk_frac   SimSample similarity-ordered sampling (0 = off)
"""

import os

import torch
import numpy as np

from utils import get_optimiser
from collections import Counter
from copy import deepcopy
from sklearn.metrics import roc_auc_score, average_precision_score

# DEMO-only imports: kept lazy so the OUTPOST path runs even in a modern PyG
# environment where the deprecated NeighborSampler was removed.
try:
    from model import train_model
    from losses import DeviationLoss, compute_beta, consistency_loss, anomaly_mixup
    from utils import get_neighbor_sampler as NeighborSampler
    from utils import NodeFeatureAugmentor
except Exception as _demo_import_err:  # pragma: no cover
    train_model = DeviationLoss = compute_beta = None
    consistency_loss = anomaly_mixup = NeighborSampler = NodeFeatureAugmentor = None
    print(f"[trainer] DEMO path unavailable ({_demo_import_err}); OUTPOST path still usable.")

def train(split_info, labels, graph, args, anomaly_info, logger, ppr_matrix=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_cutoff = torch.tensor(args.p_cutoff).to(device)
    idx_train = split_info['idx_train']
    idx_val = split_info['idx_val']
    idx_train_anomaly = torch.tensor(split_info['idx_anomaly_train'])
    unlabeled_idx = torch.tensor(split_info['idx_test']['all']).to(device) # all the unlabeled nodes

    # create the encoder and classifier
    if args.loss == 'bce':
        print("using binary cross entropy loss...")
        clf_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    elif args.loss == 'dev':
        print("using deviation loss...")
        clf_criterion = DeviationLoss()
    else:
        raise NotImplementedError("Loss is not supported!")

    model = train_model(args)

    model.to(device)
    clf_criterion.to(device)

    # create the optimisers
    optimizer = get_optimiser(args.optimiser, model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Dataloaders
    print("building the dataloader ...")
    dataloader = NeighborSampler(graph.edge_index,
                                      node_idx=idx_train,
                                      sizes=args.sampling_sizes,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      pin_memory=True)
    unlabeled_dataloader = NeighborSampler(graph.edge_index,
                                      node_idx=unlabeled_idx,
                                      sizes=args.sampling_sizes,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=0,
                                      pin_memory=True)
    weak_augmentor = NodeFeatureAugmentor({
                        'noise': {'sigma': 0.02},
                        'mask': {'mask_prob': 0.1}})
    strong_augmentor = NodeFeatureAugmentor({
                        'noise': {'sigma': 0.02},
                        'mask': {'mask_prob': 0.1},
                        'mixup': {'alpha': 0.1},
                        'scaling': {'gamma': 0.1}})


    
    # create the labels
    rest_labels = np.zeros(shape=labels.shape[0])
    for i in anomaly_info['all_anomaly']:
        rest_labels[np.where(labels == i)[0]] = 1
    rest_labels = torch.from_numpy(rest_labels).long().to(device)
    classwise_acc = torch.zeros((args.num_classes,)).to(device) # just judge the normal and anomaly class

    selected_label = torch.ones((graph.num_nodes), dtype=torch.long) * -1
    selected_label = selected_label.to(device)

    auroc_test_all_best = 0.0
    aupr_test_all_best = 0.0
    auroc_test_unknown_best = 0.0
    aupr_test_unknown_best = 0.0
    embedds_best = None
    max_counter = {i: 0 for i in [0, 1]}

    for epoch in range(args.num_epochs):
        model.train()
        logits_labeled = None

        ###### train the model ######
        for i, (batch_size_labeled, n_id_labeled, adjs_labeled) in enumerate(dataloader):
            # labeled nodes forward pass
            x_labeled = graph.x[n_id_labeled].to(device)
            adjs_labeled = [adj.to(device) for adj in adjs_labeled]
            if i == 0:
                logits_labeled = model(x_labeled, adjs_labeled)
            else:
                logits_labeled = torch.cat((logits_labeled, model(x_labeled, adjs_labeled)), dim=0)
        labeled_loss = clf_criterion(logits_labeled, rest_labels[idx_train].unsqueeze(1).float())
        loss = labeled_loss

        # mixup for the anomaly samples
        mixup_loss = 0.0
        if args.mixup:
            mixup_loss = anomaly_mixup(args, model, graph, idx_train_anomaly, ppr_matrix)
            loss = labeled_loss + args.mixup_loss * mixup_loss

        unnlabeled_loss = 0.0
        if args.used_unlabeled_data:

            pseudo_counter = Counter(selected_label.tolist())
            print(f"pseudo_counter: {pseudo_counter}")
            if max(pseudo_counter.values()) < graph.num_nodes:
                if args.thresh_warmup:
                    for i in range(args.num_classes):
                        max_counter[i] = max(max_counter[i], pseudo_counter[i])
                        if max_counter[i] == 0:
                            classwise_acc[i] = 0.0
                        else:
                            classwise_acc[i] = pseudo_counter[i] / max_counter[i]
                else:
                    wo_negative_one = deepcopy(pseudo_counter)
                    if -1 in wo_negative_one.keys():
                        wo_negative_one.pop(-1)
                    for i in range(args.num_classes):
                        max_counter[i] = max(max_counter[i], wo_negative_one[i])
                        if max_counter[i] == 0:
                            classwise_acc[i] = 0.0
                        else:
                            classwise_acc[i] = wo_negative_one[i] / max_counter[i]
            logits_unlabeled_weak, logits_unlabeled_strong, logits_unlabeled = [], [], []
            for i, (batch_size_unlabeled, n_id_unlabeled, adjs_unlabeled) in enumerate(unlabeled_dataloader):
                # unlabeled nodes forward pass
                if args.used_augment_for_anormaly:
                    x_unlabeled_weak = weak_augmentor.augment(graph.x[n_id_unlabeled].to(device))
                    x_unlabeled_strong = strong_augmentor.augment(graph.x[n_id_unlabeled].to(device))
                    adjs_unlabeled = [adj.to(device) for adj in adjs_unlabeled]
                    logits_w = model(x_unlabeled_weak, adjs_unlabeled)
                    logits_s = model(x_unlabeled_strong, adjs_unlabeled)
                    logits_unlabeled_weak.append(logits_w)
                    logits_unlabeled_strong.append(logits_s)
                else:
                    x_unlabeled = graph.x[n_id_unlabeled].to(device)
                    adjs_unlabeled = [adj.to(device) for adj in adjs_unlabeled]
                    logits_unlabeled.append(model(x_unlabeled, adjs_unlabeled))

            if args.used_augment_for_anormaly:
                logits_unlabeled_weak = torch.cat(logits_unlabeled_weak, dim=0)
                logits_unlabeled_strong = torch.cat(logits_unlabeled_strong, dim=0)
                logits = tuple([logits_unlabeled_weak, logits_unlabeled_strong])
            else:
                logits_unlabeled = torch.cat(logits_unlabeled, dim=0)
                logits = tuple([logits_unlabeled])
            if epoch > 5: # warm up the threshold
                unnlabeled_loss, mask, select, pseudo_labels = consistency_loss(logits, classwise_acc, p_cutoff=p_cutoff)

                if unlabeled_idx[select == 1].nelement() != 0: # select中选择的有正常跟异常的节点
                    selected_label[unlabeled_idx[select == 1]] = pseudo_labels[select == 1]
            loss = labeled_loss + args.unlabeled_loss * unnlabeled_loss + args.mixup_loss * mixup_loss

        # computer energy loss
        energy_loss = 0.0
        if args.energy_loss:
            beta = compute_beta(model, idx_train, idx_val, graph, rest_labels, dataloader, device, args)
            energy = model.energy(logits_labeled)
            energy_loss = torch.mean(beta * energy)
            loss = labeled_loss + args.energy_loss_weight * energy_loss + args.unlabeled_loss * unnlabeled_loss + args.mixup_loss * mixup_loss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

        if epoch % 1 == 0:

            auroc_test_all_best, aupr_test_all_best, auroc_test_unknown_best, aupr_test_unknown_best, embedds_best = (
                eval(model, split_info, rest_labels, graph, args, device, auroc_test_all_best, aupr_test_all_best, auroc_test_unknown_best, aupr_test_unknown_best, embedds_best))
            print(f"Epoch: {epoch}, Best AUROC test all: {auroc_test_all_best}, AUPR test all: {aupr_test_all_best}, AUROC test unknown: {auroc_test_unknown_best}, AUPR test unknown: {aupr_test_unknown_best}")
            logger.info(f"Epoch: {epoch}, loss: {loss}, labeled_loss: {labeled_loss}, unnlabeled_loss: {unnlabeled_loss}, mixup_loss: {mixup_loss}, energy_loss: {energy_loss}")
            logger.info(f"Best AUROC test all: {auroc_test_all_best}, AUPR test all: {aupr_test_all_best}, AUROC test unknown: {auroc_test_unknown_best}, AUPR test unknown: {aupr_test_unknown_best}")

    # Return the per-rotation metrics in the same shape train_outpost_v4 uses, so
    # DEMO rows reach results/results.csv instead of only the log. DEMO tracks no
    # validation-selected checkpoint, so only 'best' (max over epochs) exists; the
    # valsel_* columns are written as 0.0 for DEMO rows and are not comparable.
    return {"best": {"auroc_all": float(auroc_test_all_best),
                     "aupr_all": float(aupr_test_all_best),
                     "auroc_unknown": float(auroc_test_unknown_best),
                     "aupr_unknown": float(aupr_test_unknown_best)}}


def eval(model, split_info, labels, graph, args, device, auroc_test_all_best, aupr_test_all_best, auroc_test_unknown_best, aupr_test_unknown_best, embedds_best):
    model.eval()
    y_true = labels.cpu().numpy()
    idx_train = split_info['idx_train']
    idx_test_all = split_info['idx_test']['all'] # normal_test + known_abnormal_test + unknown_abnormal_test
    idx_test_unknown = split_info['idx_test']['unknown'] # normal_test + unknown_abnormal_test
    idx_test_known = split_info['idx_test']['known'] # normal_test + known_abnormal_test
    
    test_dataloader = NeighborSampler(graph.edge_index,
                                      node_idx=None,
                                      sizes=args.sampling_sizes,
                                      batch_size=args.batch_size * 4, #args.batch_size * 4
                                      shuffle=False,
                                      drop_last=False,
                                      pin_memory=True)

    with torch.no_grad():
        logits_test = None
        embeds = None
        for i, (batch_size_test, n_id_test, adjs_test) in enumerate(test_dataloader):
            x_test = graph.x[n_id_test].to(device)
            adjs_test = [adj.to(device) for adj in adjs_test]
            if i == 0:
                results = model(x_test, adjs_test, return_ebds=True)
                logits_test = results[0]
                embeds = results[1]
            else:
                results = model(x_test, adjs_test, return_ebds=True)
                logits_test = torch.cat((logits_test, results[0]), dim=0)
                embeds = torch.cat((embeds, results[1]), dim=0)
        y_pred = torch.sigmoid(logits_test).squeeze(1).cpu().numpy()

    ################# compute the metrics #####################
    # all nodes
    auroc_all, aupr_all = roc_auc_score(y_true, y_pred), average_precision_score(y_true, y_pred)
    auroc_train, aupr_train = roc_auc_score(y_true[idx_train], y_pred[idx_train]), average_precision_score(y_true[idx_train], y_pred[idx_train])
    auroc_test_all, aupr_test_all = roc_auc_score(y_true[idx_test_all], y_pred[idx_test_all]), average_precision_score(y_true[idx_test_all], y_pred[idx_test_all])
    auroc_test_unknown, aupr_test_unknown = roc_auc_score(y_true[idx_test_unknown], y_pred[idx_test_unknown]), average_precision_score(y_true[idx_test_unknown], y_pred[idx_test_unknown])
    auroc_test_known, aupr_test_known = roc_auc_score(y_true[idx_test_known], y_pred[idx_test_known]), average_precision_score(y_true[idx_test_known], y_pred[idx_test_known])
    # print(f"AUROC all: {auroc_all}, AUPR all: {aupr_all}, AUROC train: {auroc_train}, AUPR train: {aupr_train}, AUROC test all: {auroc_test_all}, AUPR test all: {aupr_test_all}, AUROC test unknown: {auroc_test_unknown}, AUPR test unknown: {aupr_test_unknown}, AUROC test known: {auroc_test_known}, AUPR test known: {aupr_test_known}")

    print(f"AUROC test all: {auroc_test_all}, AUPR test all: {aupr_test_all}, AUROC test unknown: {auroc_test_unknown}, AUPR test unknown: {aupr_test_unknown}")
    if auroc_test_all > auroc_test_all_best:
        auroc_test_all_best = auroc_test_all
        embedds_best = embeds.detach()
    if aupr_test_all > aupr_test_all_best:
        aupr_test_all_best = aupr_test_all
    if auroc_test_unknown > auroc_test_unknown_best:
        auroc_test_unknown_best = auroc_test_unknown
    if aupr_test_unknown > aupr_test_unknown_best:
        aupr_test_unknown_best = aupr_test_unknown
    return auroc_test_all_best, aupr_test_all_best, auroc_test_unknown_best, aupr_test_unknown_best, embedds_best


# ======================================================================
# OUTPOST v4 — Conformal Outpost in the stochastic sampled-subgraph regime.
# Calibration (2026-07-15, original DEMO run locally, no EG/no Mix):
# photo agg 0.8642/0.5573 — the neighbor-sampling stochasticity is the
# proven driver of the unseen-anomaly bootstrap (unseen-7 hits 0.81 there
# vs 0.69 max under full-graph training).  v4 keeps that regime and adds
# OUTPOST's components: conformal anomaly threshold, atlas gate on
# pseudo-normals, multi-sample mixup + hull-escape halo positives.
# ======================================================================

def node_context(graph, dataname, cache=True):
    """Label-free per-node scatter context for HopMix: [feat_dissim, log1p(deg)].

    feat_dissim = 1 - cos(x_v, mean of neighbour features) measures how much a
    node stands out from its neighbourhood; together with degree it is the
    deployable stand-in for the local-homophily term of the Phase-0 law (see
    analysis/proxy_validation.py, where these were validated against
    ground-truth same-class fraction). Uses no labels. Standardized to zero
    mean / unit variance and cached per dataset.
    """
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
    # Spectral Gate (Phase-0/Step-1, see analysis/): propagation-free F-view
    # vetoes pseudo-normal labels for nodes it finds suspicious.
    use_fview_gate = bool(g('use_fview_gate', False))
    alpha_f = float(g('alpha_f', 0.2))       # conformal veto budget on cal normals
    lam_f = float(g('lambda_f', 1.0))        # F-view labeled BCE weight

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

    use_hybrid = bool(g('use_hybrid', False))   # HopMix per-node view fusion
    # The two rejected view-ablations are mutually exclusive: OUTPOST_V4HM has no
    # fview_logit, so enabling both would crash deep inside the labelled pass.
    # Fail loudly here instead.
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

    # similarity-ordered neighbour sampling (zero-parameter; see
    # tools/neighbor_sampler.py and analysis/PHASE0_FINDINGS.md)
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
    # built once, reused every epoch (see eval_outpost_v4)
    loader_eval = get_neighbor_sampler(graph.edge_index, node_idx=None,
                                       sizes=sizes, batch_size=bs * 4,
                                       shuffle=False, **sim_kw)
    weak_aug = NodeFeatureAugmentor({'noise': {'sigma': 0.02}, 'mask': {'mask_prob': 0.1}})
    strong_aug = NodeFeatureAugmentor({'noise': {'sigma': 0.02}, 'mask': {'mask_prob': 0.1},
                                       'mixup': {'alpha': 0.1}, 'scaling': {'gamma': 0.1}})

    # class-progress memory (FlexMatch warmup, as in DEMO)
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

        # ---- labeled pass (accumulate over batches like DEMO) ----
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

        # F-view trains on the LABELED loss only (independent witness — no
        # pseudo-label feedback, so self-training cannot contaminate it).
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

            # ---- FixMatch-CR on the unlabeled pool (sampled subgraphs) ----
            if use_pl:
                cnt = {c: int((selected_label == c).sum()) for c in (0, 1)}
                acc = {}
                for c in (0, 1):
                    max_counter[c] = max(max_counter[c], cnt[c])
                    acc[c] = cnt[c] / max_counter[c] if max_counter[c] > 0 else 0.0
                thr_a = tau_plus_cur * (acc[1] / (2 - acc[1]))
                thr_n = 2 * tau_minus - tau_minus * (acc[0] / (2 - acc[0]))

                # Spectral-gate veto threshold: conformal (1-alpha_f) quantile
                # of F-view scores over calibration normals, this epoch.
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
                    # atlas gate: pseudo-normal only if inside normality
                    if use_atlas_gate and float(model.atlas.initialized) > 0:
                        d_at = model.d_atlas(z_w)
                        gate_ref = model.d_atlas(z_n.detach())
                        gate = torch.quantile(gate_ref, atlas_gate_q) + 1e-6
                        mask = torch.where(pred == 0, mask * (d_at <= gate).float(), mask)
                    # spectral gate: the propagation-free view must AGREE the
                    # node is normal before a pseudo-normal label is allowed
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
                    # memory update from quantile extremes (class-progress signal)
                    with torch.no_grad():
                        a_probs = p_w[p_w >= 0.5]; np_ = p_w[p_w < 0.5]
                        a_cut = a_probs.quantile(0.95) if a_probs.numel() > 0 else torch.tensor(1.0, device=device)
                        n_cut = np_.quantile(0.05) if np_.numel() > 0 else torch.tensor(0.0, device=device)
                        sel_mem = (p_w >= a_cut) | (p_w <= n_cut)
                        b_nid = torch.as_tensor(n_id[:bsz]).to(device)
                        selected_label[b_nid[sel_mem]] = pred[sel_mem].long()
                loss = loss + lam_un * (l_un_sum / max(n_un, 1))

            # ---- multi-sample mixup + hull-escape halo positives ----
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

        # ---- stochastic sampled eval (protocol-parity with DEMO) ----
        m = eval_outpost_v4(model, split_info, y_np, graph, args, device, sizes,
                            idx_val, ctx_all=ctx_all if use_hybrid else None,
                            sim_kw=sim_kw, loader=loader_eval)
        # conformal anomaly threshold from calibration-normal scores (this epoch)
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

    out = {'best': best, 'val_selected': val_sel, 'final': final}
    if record_scores:
        out['scores'] = np.stack(score_log)   # [T, N] per-epoch sigmoid scores
    return out


def eval_outpost_v4(model, split_info, y_np, graph, args, device, sizes, idx_val,
                    ctx_all=None, sim_kw=None, loader=None):
    from utils import get_neighbor_sampler
    model.eval()
    bs = int(args.batch_size) * 4
    # eval must use the SAME neighbourhood construction as training.
    # `loader` is built ONCE by the caller and reused: the sampler's CSR (and,
    # for SimSample, the similarity ordering over every edge) is deterministic
    # and epoch-independent, while the stochasticity lives in __iter__. Building
    # it per epoch cost 0.74 s plain / 3.47 s with SimSample on Yelp -> 5-23 min
    # of pure waste per 400-epoch run.
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


