"""Regenerate every figure in figures/ from the committed result tables.

No training required - each figure is a view of a CSV under analysis/tables/ or
of numbers recorded in analysis/YELP_ABLATIONS.md. Writes PDF (for the paper)
and PNG (for slides/README) side by side.

    python analysis/scripts/make_figures.py
"""

import os, json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
FIG = "figures"
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})
C = {"ours": "#1b6ca8", "demo": "#c0392b", "other": "#7f8c8d",
     "pos": "#27ae60", "neg": "#c0392b", "neutral": "#95a5a6"}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}/{name}.{ext}")
    plt.close(fig)
    print(f"  figures/{name}.pdf + .png")


# --------------------------------------------------------------------------
# 1. the detectability law: topology predicts achievable detectability
# --------------------------------------------------------------------------
def fig_detectability_law():
    rows = []
    # the OGB graphs contribute 19 further anomaly classes and are what closes
    # the "16 classes is a small sample" threat to validity, so the figure has
    # to show them
    for ds in ("photo", "computers", "cs", "yelp", "amazon", "tfinance", "ogbn-arxiv", "ogbn-mag"):
        p = f"analysis/tables/spectral_{ds}.csv"
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d["dataset"] = ds
        rows.append(d)
    if not rows:
        print("  [skip] detectability_law (no spectral_*.csv)")
        return
    df = pd.concat(rows, ignore_index=True)
    hop = [c for c in df.columns if c.startswith("auc_hop")]
    df["best_auc"] = df[hop].max(axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.5), constrained_layout=True)
    # semi-synthetic graphs in grey-blues, the one real-fraud graph in the
    # accent colour: the argument of section 6.1 is where yelp sits, so it has
    # to be findable at a glance
    style = {
        "photo":      dict(marker="o", c="#8fb8de"),
        "computers":  dict(marker="s", c="#5a8fc7"),
        "cs":         dict(marker="^", c="#2e6da4"),
        "ogbn-arxiv": dict(marker="v", c="#9b8ec4"),
        "ogbn-mag":   dict(marker="P", c="#6b5b95"),
        "yelp":       dict(marker="D", c="#e07b39"),
        "amazon":     dict(marker="X", c="#c1442e"),   # the second real graph
        "tfinance":   dict(marker="*", c="#8a2d1e"),   # the third
    }
    REAL = ("yelp", "amazon", "tfinance")
    order = [d for d in style if d in set(df.dataset)]

    def draw(a, ycol):
        # Two regimes (METHODOLOGY 6.2): classes that propagation HELPS (gain>0)
        # are filled, classes it hurts (gain<=0, "feature-visible") are hollow.
        # The same-class fraction predicts only the filled ones; Amazon, the
        # out-of-sample test that forced the split, is hollow and at the top.
        for ds in order:
            g = df[df.dataset == ds]
            st = style[ds]
            real = ds in REAL
            for hollow, gg in ((False, g[g.smoothing_gain > 0]),
                               (True, g[g.smoothing_gain <= 0])):
                if gg.empty:
                    continue
                a.scatter(gg.med_same_frac, gg[ycol], s=64 if real else 42,
                          marker=st["marker"],
                          facecolor="none" if hollow else st["c"],
                          edgecolor=st["c"] if hollow else "k",
                          label=ds if not hollow or g[g.smoothing_gain > 0].empty else None,
                          alpha=.95, linewidth=1.4 if (hollow or real) else .4,
                          zorder=4 if real else 3)

    from scipy.stats import spearmanr
    rho = spearmanr(df.med_same_frac, df.best_auc).statistic
    # the permutation p and the cluster-bootstrap interval, not the asymptotic
    # p: 35 classes drawn from 6 graphs are not 35 independent observations, so
    # the asymptotic figure overstates the evidence (see detectability_law.py)
    sub = "permutation $p$=5e-5, cluster-bootstrap 95% CI [0.61, 0.92]"
    law = None
    p = "analysis/tables/detectability_law.json"
    if os.path.exists(p):
        import json
        law = json.load(open(p))
        if law.get("n_classes") == len(df):
            ci = law["cluster_bootstrap_ci95"]
            sub = (f"permutation $p$={law['p_permutation_within_dataset']:.0e}, "
                   f"cluster-bootstrap 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")

    draw(ax[0], "best_auc")
    # the propagation-regime fit, drawn only over the regime it applies to
    if law and "two_regime" in law and "propagation_fit" in law["two_regime"]:
        pf = law["two_regime"]["propagation_fit"]
        prop = df[df.smoothing_gain > 0]
        xs = np.linspace(prop.med_same_frac.min(), prop.med_same_frac.max(), 50)
        ax[0].plot(xs, pf["intercept"] + pf["slope"] * xs, color="#2e6da4", lw=1.2,
                   ls="--", zorder=2,
                   label=f"propagation regime: $\\rho$={pf['spearman']:.2f}, n={law['two_regime']['n_propagation_regime']}")
        ax[0].text(0.02, 0.97, "hollow = propagation hurts (feature-visible)",
                   transform=ax[0].transAxes, fontsize=7, va="top", color="#444")
    ax[0].set_xlabel("anomaly same-class neighbour fraction\n"
                     "(how clustered the class is — no training, no test labels)")
    ax[0].set_ylabel("best prototype AUC-ROC (training-free ceiling)")
    ax[0].set_title(f"Detectability is topological\n"
                    f"Spearman $\\rho$={rho:.3f}, n={len(df)} classes / "
                    f"{df.dataset.nunique()} graphs\n{sub}", fontsize=8.5)
    # one shared legend under both panels: inside either one it sits on data

    ax[1].axhline(0, color="k", lw=.8, zorder=1)
    draw(ax[1], "smoothing_gain")
    ax[1].set_xlabel("anomaly same-class neighbour fraction")
    ax[1].set_ylabel("AUC gain from propagation (hop 3 − hop 0)")
    ax[1].set_title("Propagation amplifies clustered anomalies\n"
                    "and erases scattered ones", fontsize=8.5)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc="outside lower center",
               ncol=len(order))
    save(fig, "detectability_law")


# --------------------------------------------------------------------------
# run-record readers: every figure below draws from results.csv at draw time.
# The previous versions carried three-seed numbers typed in by hand from the
# original laptop runs, and kept drawing them for a month after ten-seed
# re-runs on this machine had replaced every one of them.
# --------------------------------------------------------------------------
def _runs():
    d = pd.read_csv("results/results.csv").drop_duplicates(
        subset=["dataset", "method", "seed", "train_seed", "tag"], keep="last")
    return d[d.num_epochs >= 100]


def _cell(d, dataset, tag, method="outpost", metric="best"):
    g = d[(d.dataset == dataset) & (d.method == method) & (d.tag.astype(str) == tag)]
    if g.empty:
        return None
    return (float(g[f"{metric}_auroc"].mean()), float(g[f"{metric}_aupr"].mean()), int(len(g)))


def _params(dataset, model, hidden, gate=False):
    """Parameter count for a configuration.

    `gate` is the SPECTRAL fview gate, which is False in every arm the paper
    reports; the atlas gate that gets ablated is non-parametric and does not
    change this number (METHODOLOGY 5.5). The column was renamed gate ->
    spectral_gate on 2026-09-07 to stop the two being confused, so accept either.
    """
    p = "analysis/tables/table_complexity.csv"
    if not os.path.exists(p):
        return None
    t = pd.read_csv(p)
    col = "spectral_gate" if "spectral_gate" in t.columns else "gate"
    _ = col
    q = t[(t.dataset == dataset) & (t.model == model) & (t.hidden == hidden)]
    if model != "DEMO":
        q = q[q[col].astype(str) == str(gate)]
    return int(q.params.iloc[0]) if len(q) else None


def _params_of_run(dataset, tag):
    """Parameters of the model that arm ACTUALLY trained, from its shard.

    Preferred over _params(): the shard stamps n_params from the instantiated
    model, so it cannot disagree with what ran. table_complexity.csv is built by
    instantiating modules by hand and its rows include a spectral-gate variant
    that no reported arm uses - reading it by (hidden, gate) once put the main
    Yelp arm on this figure at 11,714 parameters when the model it trained had
    7,361 (METHODOLOGY 5.5).
    """
    import glob as _g
    for f in sorted(_g.glob(f"results/rotations/{dataset}_outpost_s*_{tag}_rot*.json")):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        for r in j.get("rotations", []):
            if r.get("n_params"):
                return int(r["n_params"])
    return None


# --------------------------------------------------------------------------
# 2. SimSample: dose-response + placebo control (the causal argument)
# --------------------------------------------------------------------------
def fig_simsample():
    """Dose-response with a placebo, on Yelp, from the ten-seed re-runs."""
    d = _runs()
    arms = [("uniform\n(off)", "C_nosim"), ("placebo\n(shuffled)", "C_simplacebo"),
            ("frac 0.25", "D_sim0.25"), ("frac 0.5", "D_sim0.5"),
            ("frac 0.75", "D_sim0.75"), ("frac 1.0\n(main)", "A_main")]
    cells = [(lab, _cell(d, "yelp", tag)) for lab, tag in arms]
    cells = [(lab, c) for lab, c in cells if c]
    if len(cells) < 4:
        print("  [skip] simsample (yelp arms missing)"); return
    labs = [f"{lab}\nn={c[2]}" for lab, c in cells]
    pr = [c[1] for _, c in cells]; roc = [c[0] for _, c in cells]
    col = [C["neutral"] if ("uniform" in lab or "placebo" in lab) else C["pos"] for lab, _ in cells]
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4), constrained_layout=True)
    for a, y, name, pub in ((ax[0], pr, "AUC-PR", 0.3029), (ax[1], roc, "AUC-ROC", 0.7097)):
        a.bar(range(len(y)), y, color=col, edgecolor="k", linewidth=.5)
        a.axhline(pub, ls="--", c=C["demo"], lw=1, label=f"best published ({pub:.4f})")
        a.set_xticks(range(len(y))); a.set_xticklabels(labs, fontsize=6.5)
        a.set_ylabel(f"Yelp {name}")
        lo, hi = min(y + [pub]), max(y + [pub]); a.set_ylim(lo - .03, hi + .03)
        for i, v in enumerate(y):
            a.text(i, v + .003, f"{v:.4f}", ha="center", fontsize=6.5)
        a.legend(fontsize=7, frameon=False, loc="upper left")
    ax[0].set_title("Placebo isolates the cause: shuffling the\nsimilarity order removes the whole gain", fontsize=8.5)
    ax[1].set_title("Dose-response over the similarity fraction", fontsize=8.5)
    save(fig, "simsample_mechanism")


# --------------------------------------------------------------------------
# 3. Yelp comparison against the published field
# --------------------------------------------------------------------------
def fig_yelp_comparison():
    """Published field plus every method WE re-ran on Yelp, from results.csv."""
    p = "analysis/tables/published_baselines.csv"
    meth, roc, pr = [], [], []
    if os.path.exists(p):
        b = pd.read_csv(p).dropna(subset=["Yelp_AUC-PR"])
        b = b[~b["Method"].str.upper().str.startswith("OUTPOST")]
        for _, r in b.iterrows():
            lab = "DEMO (published)" if r["Method"] == "DEMO" else r["Method"]
            meth.append(lab); roc.append(r["Yelp_AUC-ROC"]); pr.append(r["Yelp_AUC-PR"])
    d = _runs()
    for lab, tag, method in (("DEMO (re-run, ours)", "B_demo_mix", "demo"),
                             ("NSReg (re-run, ours)", "E400_nsreg", "nsreg"),
                             ("OUTPOST (ours)", "A_main", "outpost")):
        c = _cell(d, "yelp", tag, method)
        if c:
            meth.append(f"{lab}\nn={c[2]}"); roc.append(c[0]); pr.append(c[1])
    o = np.argsort(pr)
    meth = [meth[i] for i in o]; pr = [pr[i] for i in o]; roc = [roc[i] for i in o]
    def colour(m):
        if "OUTPOST" in m: return C["ours"]
        if "re-run" in m: return "#7a4f9a"
        if "DEMO" in m: return C["demo"]
        return C["other"]
    col = [colour(m) for m in meth]
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.6))
    ax[0].barh(range(len(meth)), pr, color=col, edgecolor="k", linewidth=.5)
    ax[0].set_yticks(range(len(meth))); ax[0].set_yticklabels(meth, fontsize=6.5)
    ax[0].set_xlabel("Yelp AUC-PR"); ax[0].set_title("Real fraud: AUC-PR", fontsize=9)
    for i, v in enumerate(pr):
        ax[0].text(v + .005, i, f"{v:.4f}", va="center", fontsize=6.5)
    ax[1].barh(range(len(meth)), roc, color=col, edgecolor="k", linewidth=.5)
    ax[1].set_yticks(range(len(meth))); ax[1].set_yticklabels([])
    ax[1].set_xlim(0.5, 0.80)
    ax[1].set_xlabel("Yelp AUC-ROC"); ax[1].set_title("Real fraud: AUC-ROC", fontsize=9)
    for i, v in enumerate(roc):
        ax[1].text(v + .003, i, f"{v:.4f}", va="center", fontsize=6.5)
    fig.text(0.5, -0.02, "purple = baselines re-run under this protocol at the same seeds and splits as OUTPOST; grey = as published",
             ha="center", fontsize=6.5, color="#444")
    save(fig, "yelp_comparison")


# --------------------------------------------------------------------------
# 4. efficiency: performance vs parameter budget
# --------------------------------------------------------------------------
def fig_efficiency():
    """Yelp AUC-PR against parameter count, ten-seed arms where they exist."""
    d = _runs()
    pts = []
    # The atlas gate is non-parametric, so "main" and "gate off" sit at the SAME
    # parameter count and differ only in accuracy; labelling them as a size
    # trade-off was the error corrected in METHODOLOGY 5.5.
    for lab, tag, hidden, gate in (("main (h32)", "A_main", 32, False),
                                   ("atlas gate off (same size)", "C_nogate", 32, False),
                                   ("gate+PL off", "L_lean", 32, False),
                                   ("h16", "T_hidden16", 16, False),
                                   ("h64", "T_hidden64", 64, False)):
        c = _cell(d, "yelp", tag)
        n_par = _params_of_run("yelp", tag) or _params("yelp", "OUTPOST", hidden, gate)
        if c and n_par:
            pts.append((lab, n_par, c[1], c[2]))
    demo = _cell(d, "yelp", "B_demo_mix", "demo"); demo_par = _params("yelp", "DEMO", 64, None)
    if not pts:
        print("  [skip] efficiency (no yelp arms)"); return
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.axhline(0.3029, ls="--", c=C["demo"], lw=1, label="best published (NSReg 0.3029)")
    if demo and demo_par:
        ax.scatter([demo_par], [demo[1]], s=80, marker="s", c=C["demo"], edgecolor="k",
                   linewidth=.5, zorder=3, label=f"DEMO re-run (n={demo[2]})")
        ax.axvline(demo_par, ls=":", c=C["other"], lw=1)
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
    ax.scatter(xs, ys, s=[70 if p[3] >= 10 else 40 for p in pts], c=C["ours"],
               edgecolor="k", linewidth=.5, zorder=4, label="OUTPOST arms (large = n>=10)")
    for lab, x, y, n in pts:
        dy = 5 if lab.startswith("gate+PL") else (-9 if lab.startswith("gate off") else -3)
        ax.annotate(f"{lab}", (x, y), textcoords="offset points", xytext=(7, dy), fontsize=6.5)
    ax.set_xscale("log"); ax.set_xlabel("trainable parameters (log)"); ax.set_ylabel("Yelp AUC-PR")
    ax.set_ylim(0.28, max(ys + ([demo[1]] if demo else [])) + 0.012)   # room for the legend under the line
    main = [p for p in pts if p[0].startswith("main")]
    ratio = f"{main[0][1]/demo_par:.2f}" if (main and demo_par) else "?"
    ax.set_title(f"Above the published field at {ratio}$\\times$ DEMO's parameters", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    save(fig, "efficiency")


# --------------------------------------------------------------------------
# 5. why SimSample is dataset-conditional (the honest mechanism decomposition)
# --------------------------------------------------------------------------
def fig_regime():
    """SimSample's effect against degree and budget, on every graph measured.

    The earlier version drew three graphs and captioned the panel 'SimSample
    applies only where degree exceeds the sampling budget'. That mechanism was
    pre-registered as P14/P17 and falsified: CS and ogbn-arxiv sit far below
    the budget and show nothing; Yelp and Amazon sit far above it with nearly
    identical above-budget fractions and OPPOSITE signs; Photo and Computers
    have identical degree profiles and opposite signs. This is the figure of
    that falsification.
    """
    p = "analysis/tables/simsample_by_dataset.json"
    if not os.path.exists(p):
        print("  [skip] regime (no simsample_by_dataset.json)"); return
    import json
    J = json.load(open(p))
    order = sorted(J, key=lambda k: J[k]["median_degree"])
    deg = [J[k]["median_degree"] for k in order]
    above = [100 * J[k]["frac_above_budget25"] for k in order]
    eff = [J[k]["sim_delta_roc"]["delta"] if J[k]["sim_delta_roc"] else np.nan for k in order]
    nn = [J[k]["sim_delta_roc"]["n"] if J[k]["sim_delta_roc"] else 0 for k in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.6), constrained_layout=True)
    ax[0].bar(x, [max(v, .5) for v in deg], color=[C["pos"] if v > 25 else C["neg"] for v in deg],
              edgecolor="k", linewidth=.5)
    ax[0].axhline(25, ls="--", c="k", lw=1, label="hop-1 sampling budget (25)")
    ax[0].set_yscale("log"); ax[0].set_ylabel("median degree (log)")
    ax[0].set_title("Degree vs the sampling budget", fontsize=9); ax[0].legend(fontsize=6.5, frameon=False)
    ax[1].bar(x, above, color=[C["pos"] if v > 50 else C["neg"] for v in above], edgecolor="k", linewidth=.5)
    ax[1].set_ylabel("% nodes above budget"); ax[1].set_title("Where selection can act", fontsize=9)
    ax[2].axhline(0, color="k", lw=.8)
    ax[2].bar(x, eff, color=[C["pos"] if v > 0 else C["neg"] for v in eff], edgecolor="k", linewidth=.5)
    lo, hi = min(eff), max(eff); span = hi - lo
    ax[2].set_ylim(lo - .28 * span, hi + .30 * span)
    for i, (v, n) in enumerate(zip(eff, nn)):
        # label above positive bars, below negative ones, always inside the axes
        ax[2].text(i, v + (.02 * span if v >= 0 else -.02 * span), f"{v:+.4f}\nn={n}",
                   ha="center", va="bottom" if v >= 0 else "top", fontsize=6)
    ax[2].set_ylabel("$\\Delta$ AUC-ROC (on $-$ off)"); ax[2].set_title("What it did", fontsize=9)
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(order, fontsize=7, rotation=20)
    fig.suptitle("SimSample's effect tracks neither degree nor budget: the subsampling-removal "
                 "mechanism is falsified (P14, P17)", fontsize=9, y=1.03)
    save(fig, "regime_decomposition")


if __name__ == "__main__":
    # All five figures this file used to draw have been ported to
    # analysis/scripts/make_figures_paper.py, which draws them in the paper style
    # (no chart titles, warm palette, measured label placement) and is checked by
    # check_figure_text.py. Both scripts wrote the SAME five filenames for a while
    # and only the running order decided which survived -- the exact trap that has
    # bitten this repo before. The drawing code is kept below for history; the data
    # helpers (_runs, _cell, _params, _params_of_run) are still imported and used.
    print("make_figures.py no longer draws: the paper figures are built by")
    print("  python analysis/scripts/make_figures_paper.py")
    raise SystemExit(0)
