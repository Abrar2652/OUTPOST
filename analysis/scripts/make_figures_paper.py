#!/usr/bin/env python3
"""The paper's figure set. One function per figure; every number read at runtime.

Nothing here is typed in. Each function names the artifact it reads, and a value
that cannot be found is an error rather than a default, because the alternative -
a plausible-looking figure built from a stale constant - is the failure this
project has already had once (the ranking headline moved 3-of-5 -> 2-of-7 after
figures were first drawn).

    python analysis/scripts/make_figures_paper.py [fig1 fig2 ...]
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, "analysis")
import figstyle as F                                          # noqa: E402
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox                               # noqa: E402


def load(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing artifact: {path} (run refresh_all.sh)")
    return json.load(open(path))


# --------------------------------------------------------------------------
def fig_selection_noise():
    """The tie band sits below the noise of the comparison it decides.

    Source: analysis/tables/selection_noise.json   Claim: METHODOLOGY 4.9 (P35, P40)
    """
    d = load("analysis/tables/selection_noise.json")
    band, mseeds = d["tie_band"], d["min_shared_seeds"]
    rows = sorted(d["datasets"], key=lambda r: r["paired_diff_sd"])
    n = np.arange(1, max(51, d["summary"]["max_seeds_needed"] + 11))
    fig, ax = plt.subplots(figsize=(3.9, 2.8))
    cols = [F.CRIMSON, F.AMBER, F.PLUM, F.TEAL, F.SAND, F.GREY]
    ends = []
    for r, c in zip(rows, cols):
        se = r["paired_diff_sd"] / np.sqrt(n)
        ax.plot(n, se, color=c, lw=1.6, solid_capstyle="round", zorder=3)
        k = r["seeds_needed_for_band"]          # where the curve clears the band
        ax.plot([k], [band], marker="o", ms=5, color=c, zorder=5, **F.MEDGE)
        ends.append([np.log10(se[-1]), F.nice(r["dataset"]), c])
    # labels sit at the right end of each curve; curves for graphs with similar
    # spread finish within a few pixels, so they are separated after drawing, by
    # their measured boxes rather than by a guessed gap in log space
    ends.sort()
    lab = [ax.annotate(name, (n[-1], 10 ** ly), xytext=(4, 0),
                       textcoords="offset points", va="center", fontsize=7.5, color=c)
           for ly, name, c in ends]
    # the band line stops where the curves stop: drawn across the full axes it ran
    # straight through the dataset labels parked past the last data point
    ax.plot([1, n[-1]], [band, band], color=F.INK, lw=1.2, ls=(0, (4, 2)), zorder=4)
    # every curve is below the band once it has crossed, so the region above the
    # line's right end is the only place this label does not sit on a curve
    ax.annotate(f"tie band {band}", (n[-1] * 0.9, band), xytext=(0, 26),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=7.5, color=F.INK,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=F.INK,
                                shrinkA=0.5, shrinkB=1.5))
    ax.axvspan(1, mseeds, color=F.CRIMSON, alpha=0.10, zorder=1)
    ax.annotate(f"{mseeds} seeds:\nthe rule's budget", (mseeds, ax.get_ylim()[1]),
                xytext=(5, -3),
                textcoords="offset points", va="top", fontsize=7.5, color=F.CRIMSON)
    ax.set_yscale("log")
    ax.set_xlim(1, n[-1] + 12)
    ax.set_xlabel("selection seeds")
    ax.set_ylabel("SE of the paired arm difference")
    F.declutter(fig, lab, axis="y")
    F.save(fig, "selection_noise")


# --------------------------------------------------------------------------
def fig_law_crossval():
    """Out of sample the law loses to a constant, and loses where the classes are.

    Source: analysis/tables/law_crossval.json      Claim: METHODOLOGY 6.5
    """
    d = load("analysis/tables/law_crossval.json")
    folds = d["folds"]
    x = [f["mae_intercept_only"] for f in folds]
    y = [f["mae_same_frac_line"] for f in folds]
    s = [f["n_classes"] for f in folds]
    px, py = d["pooled_mae_intercept_only"], d["pooled_mae_same_frac_line"]
    fig, ax = plt.subplots(figsize=(3.9, 3.0))
    hi = max(x + y + [px, py]) * 1.18
    ax.plot([0, hi], [0, hi], color=F.INK, lw=1.0, ls=(0, (4, 2)), zorder=2)
    ax.fill_between([0, hi], [0, hi], [hi, hi], color=F.CRIMSON, alpha=0.07, zorder=1)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    for xi, yi, si in zip(x, y, s):
        ax.scatter([xi], [yi], s=26 + 22 * si, color=F.AMBER if yi < xi else F.CRIMSON,
                   zorder=4, **F.EDGE)
    ax.scatter([px], [py], marker="D", s=70, color=F.PLUM, zorder=5, **F.EDGE)

    # Every label is drawn first, then measured, then placed. Sizing a label by its
    # character count is what put "Computers (5)" across "ogbn-arxiv (4)": the guess
    # was narrower than the text, so a direction that scored clear was not.
    anchors = [(xi, yi) for xi, yi in zip(x, y)] + [(px, py)]
    labs = [f"{F.nice(f['held_out'])} ({si})" for f, si in zip(folds, s)] + ["pooled"]
    cols = [F.INK] * len(folds) + [F.PLUM]
    region = ax.annotate("law worse than a constant", (hi * 0.03, hi * 0.95),
                         xytext=(0, 0), textcoords="offset points", ha="left",
                         va="top", fontsize=7.5, color=F.CRIMSON, zorder=6)
    anns = [ax.annotate(l, xy, xytext=(0, 14), textcoords="offset points",
                        ha="center", va="center", fontsize=7, color=c, zorder=6,
                        arrowprops=dict(arrowstyle="-", lw=0.6, color=F.GREY,
                                        shrinkA=0.5, shrinkB=2.5))
            for l, xy, c in zip(labs, anchors, cols)]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ab = ax.bbox
    dirs = [(np.cos(t), np.sin(t)) for t in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
    # markers, plus a dense sample of the parity line: both are drawn data that a
    # label must not sit on, and the line is what "Computers (5)" was landing on
    diag = np.linspace(0, hi, 400)
    avoid = np.vstack([np.array([ax.transData.transform(p) for p in anchors]),
                       ax.transData.transform(np.column_stack([diag, diag]))])
    pts = [ax.transData.transform(p) for p in anchors]
    keepout = [F.textbox(region, r)]        # placed labels plus the region note

    def clash(bx):
        """How badly a candidate box sits: off-axes area, then overlap with what is there."""
        off = ((max(ab.x0 - bx.x0, 0) + max(bx.x1 - ab.x1, 0)) * bx.height +
               (max(ab.y0 - bx.y0, 0) + max(bx.y1 - ab.y1, 0)) * bx.width)
        over = sum(max(0, min(bx.x1, k.x1) - max(bx.x0, k.x0)) *
                   max(0, min(bx.y1, k.y1) - max(bx.y0, k.y0)) for k in keepout)
        hit = ((avoid[:, 0] > bx.x0 - 3) & (avoid[:, 0] < bx.x1 + 3) &
               (avoid[:, 1] > bx.y0 - 3) & (avoid[:, 1] < bx.y1 + 3))
        near = float(hit.any())
        # the page is saved with a tight bbox, so a label just outside the axes is
        # still drawn and merely looks detached; a label on top of other text is
        # unreadable. Overlap is weighted well above going over the edge.
        return off * 0.5 + over * 5 + near * 400

    for a_, (ax0, ay0), si in zip(anns, pts, s + [max(s)]):
        bb = F.textbox(a_, r)
        w, h = bb.width, bb.height
        best, score = (0, 14), None
        for rad in (13 + 0.7 * si, 20 + 0.7 * si, 30 + 0.7 * si, 42, 56, 72):
            for v in dirs:
                cx, cy = ax0 + v[0] * rad * fig.dpi / 72, ay0 + v[1] * rad * fig.dpi / 72
                x0 = cx - w / 2 if abs(v[0]) < 0.6 else (cx if v[0] > 0 else cx - w)
                cand = Bbox.from_bounds(x0, cy - h / 2, w, h)
                pen = clash(cand)
                if score is None or pen < score:
                    best, score, bestbox, bestv = (v[0] * rad, v[1] * rad), pen, cand, v
                if pen == 0:
                    break
            if score == 0:
                break
        a_.set_ha("center" if abs(bestv[0]) < 0.6 else ("left" if bestv[0] > 0 else "right"))
        a_.xyann = best
        keepout.append(bestbox)
    ax.set_xlabel("held-out MAE, predicting the training mean")
    ax.set_ylabel("held-out MAE, the law")
    F.save(fig, "law_crossval")


# --------------------------------------------------------------------------
def fig_mechanism():
    """Every component is null on graphs whose anomalies were observed, not injected.

    Source: analysis/appendix.json (ablations)     Claim: METHODOLOGY 5.4 (P28, P39)
    """
    A = load("analysis/appendix.json")
    # arm names carry a per-graph suffix (T-Finance's PL arm is C_nopl_sim0.0);
    # match on the stem so a renamed arm cannot silently drop a point
    STEM = {"gate": "C_nogate", "conformal": "C_noconformal", "pseudo-label": "C_nopl"}
    graphs = ["Yelp", "Amazon", "T-Finance"]
    synth_graphs = ["Computers", "CS", "Photo"]
    comps = list(STEM)
    by = {}
    for r in A["ablations"]:
        for comp, stem in STEM.items():
            if r["arm"] == stem or r["arm"].startswith(stem + "_"):
                by[(r["dataset"], comp)] = r
    missing = [(g, c) for g in graphs for c in comps if (g, c) not in by]
    if missing:
        raise SystemExit(f"mechanism figure: no ablation row for {missing}")
    fig, ax = plt.subplots(figsize=(3.8, 2.5))
    # the semi-synthetic pseudo-label effects, as the band the real graphs are measured against
    sv = [by[(g, "pseudo-label")]["delta_roc"]["delta"] for g in synth_graphs
          if (g, "pseudo-label") in by]
    if sv:
        ax.axvspan(min(sv), max(sv), color=F.AMBER, alpha=0.16, zorder=1)
        # the label goes in the empty column between the band and the zero line, at
        # mid height. Across the top of the axes it read as a chart title, which is
        # the one thing these figures never have.
        ax.annotate("pseudo-labelling\non the three\ninjected-anomaly\ngraphs",
                    (max(sv), 1.0), xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=7, color="#7A4A10")
    ax.axvline(0, color=F.INK, lw=1.0, zorder=2)
    marks = {"gate": "o", "conformal": "s", "pseudo-label": "^"}
    for gi, g in enumerate(graphs):
        for ci, comp in enumerate(comps):
            r = by[(g, comp)]
            dd, p = r["delta_roc"]["delta"], r["delta_roc"]["p_wilcoxon"]
            yy = gi + (ci - 1) * 0.22
            sig = p < 0.05
            ax.scatter([dd], [yy], marker=marks[comp], s=54 if sig else 40,
                       color=F.CRIMSON if sig else "white", zorder=4, **F.EDGE)
    ax.set_yticks(range(len(graphs)))
    ax.set_yticklabels(graphs)
    ax.set_ylim(-0.55, 2.95)
    ax.set_xlabel("change in AUC-ROC when the component is removed")
    h = [plt.Line2D([], [], marker=marks[c], ls="none", mfc="white", mec=F.INK,
                    ms=6, label=c) for c in comps]
    h.append(plt.Line2D([], [], marker="o", ls="none", mfc=F.CRIMSON, mec=F.INK,
                        ms=6, label="p < 0.05"))
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=4,
              handletextpad=0.3, columnspacing=0.9, borderpad=0.4)
    F.save(fig, "mechanism_real_vs_synthetic")


# --------------------------------------------------------------------------
def fig_efficiency():
    """OUTPOST clears the published field at a fraction of DEMO's parameters.

    Source: results/results.csv via analysis/scripts/make_figures.py helpers
    Claim: METHODOLOGY 5.5 (parameter efficiency; the atlas gate is non-parametric)
    """
    import make_figures as G          # data helpers only
    F.use()                           # it sets its own rcParams at import; undo that
    d = G._runs()
    # the atlas gate is non-parametric, so main and gate-off sit at the SAME
    # parameter count and differ only in accuracy (METHODOLOGY 5.5)
    ARMS = (("main", "A_main", 32), ("gate off", "C_nogate", 32),
            ("gate+PL off", "L_lean", 32), ("h16", "T_hidden16", 16),
            ("h64", "T_hidden64", 64))
    pts = []
    for lab, tag, hidden in ARMS:
        c = G._cell(d, "yelp", tag)
        npar = G._params_of_run("yelp", tag) or G._params("yelp", "OUTPOST", hidden, False)
        if c and npar:
            pts.append((lab, npar, c[1], c[2]))
    demo = G._cell(d, "yelp", "B_demo_mix", "demo")
    dpar = G._params("yelp", "DEMO", 64, None)
    if not pts or not demo or not dpar:
        raise SystemExit("efficiency figure: missing yelp arms or DEMO parameters")
    # the published ceiling is read from the baseline table, never typed: it moved
    # once already, and a figure quoting a stale one is a figure that lies
    import csv as _csv
    cand = [(r["Method"], float(r["Yelp_AUC-PR"]))
            for r in _csv.DictReader(open("analysis/tables/published_baselines.csv"))
            if (r.get("Yelp_AUC-PR") or "").strip() not in ("", "--", "nan")
            and r["Method"] != "DEMO"]
    if not cand:
        raise SystemExit("efficiency figure: no published Yelp AUC-PR to compare against")
    pub_name, pub = max(cand, key=lambda t: t[1])

    fig, ax = plt.subplots(figsize=(3.8, 2.7))
    ax.set_xscale("log")
    ax.axhline(pub, ls=(0, (4, 2)), color=F.PLUM, lw=1.1, zorder=2)
    ax.axvline(dpar, ls=(0, (1, 2)), color=F.GREY, lw=1.0, zorder=1)
    ax.scatter([dpar], [demo[1]], s=70, marker="s", color=F.AMBER, zorder=4, **F.EDGE)
    for lab, xp, yp, n in pts:
        ax.scatter([xp], [yp], s=68 if n >= 10 else 34, color=F.CRIMSON,
                   zorder=4, **F.EDGE)
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel("Yelp AUC-PR")
    ys = [p[2] for p in pts] + [demo[1], pub]
    ax.set_ylim(min(ys) - 0.012, max(ys) + 0.016)

    # three arms share a parameter count, so their labels are placed by measurement
    anns = [ax.annotate(f"{lab} (n={n})", (xp, yp), xytext=(0, 12),
                        textcoords="offset points", ha="center", va="center",
                        fontsize=6.8, color=F.INK, zorder=6,
                        arrowprops=dict(arrowstyle="-", lw=0.6, color=F.GREY,
                                        shrinkA=0.5, shrinkB=2.5))
            for lab, xp, yp, n in pts]
    anns.append(ax.annotate(f"DEMO (n={demo[2]})", (dpar, demo[1]), xytext=(0, 12),
                            textcoords="offset points", ha="center", va="center",
                            fontsize=6.8, color=F.INK, zorder=6,
                            arrowprops=dict(arrowstyle="-", lw=0.6, color=F.GREY,
                                            shrinkA=0.5, shrinkB=2.5)))
    F.place_labels(fig, ax, anns)
    h = [plt.Line2D([], [], marker="o", ls="none", mfc=F.CRIMSON, mec=F.INK, ms=6,
                    label="OUTPOST arm"),
         plt.Line2D([], [], marker="s", ls="none", mfc=F.AMBER, mec=F.INK, ms=6,
                    label="DEMO re-run"),
         plt.Line2D([], [], color=F.PLUM, lw=1.1, ls=(0, (4, 2)),
                    label=f"best published ({pub_name} {pub:.4f})")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2,
              fontsize=7, handletextpad=0.5, columnspacing=1.0)
    F.save(fig, "efficiency")


# --------------------------------------------------------------------------
def fig_yelp_comparison():
    """On the one graph with real fraud labels, OUTPOST leads the published field.

    Source: analysis/tables/published_baselines.csv + results/results.csv
    Claim: METHODOLOGY 5.5 (Yelp is the only observed-anomaly graph with a field)
    """
    import csv as _csv
    import make_figures as G
    F.use()
    rows = []
    for r in _csv.DictReader(open("analysis/tables/published_baselines.csv")):
        v = (r.get("Yelp_AUC-PR") or "").strip()
        if v in ("", "--", "nan") or r["Method"].upper().startswith("OUTPOST"):
            continue
        lab = "DEMO (published)" if r["Method"] == "DEMO" else r["Method"]
        rows.append([lab, float(r["Yelp_AUC-ROC"]), float(v), "published"])
    d = G._runs()
    for lab, tag, meth in (("DEMO", "B_demo_mix", "demo"),
                           ("NSReg", "E400_nsreg", "nsreg"),
                           ("OUTPOST", "A_main", "outpost")):
        c = G._cell(d, "yelp", tag, meth)
        if c:
            kind = "ours" if meth == "outpost" else "rerun"
            rows.append([f"{lab} (ours, n={c[2]})", c[0], c[1], kind])
    if not rows:
        raise SystemExit("yelp comparison: no rows")
    rows.sort(key=lambda r: r[2])
    col = {"published": F.GREY, "rerun": F.PLUM, "ours": F.CRIMSON}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    for ax, j, name, lo in ((axes[0], 2, "Yelp AUC-PR", 0.0),
                            (axes[1], 1, "Yelp AUC-ROC", 0.5)):
        v = [r[j] for r in rows]
        ax.barh(range(len(rows)), v, color=[col[r[3]] for r in rows],
                height=0.72, zorder=3, **F.EDGE)
        ax.set_xlim(lo, max(v) * 1.20 if lo == 0 else 0.82)
        ax.set_xlabel(name)
        for i, x in enumerate(v):
            ax.annotate(f"{x:.4f}", (x, i), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=6.4, color=F.INK)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=6.8)
    axes[0].set_ylim(-0.7, len(rows) - 0.3)
    h = [plt.Line2D([], [], marker="s", ls="none", mfc=col[k], mec=F.INK, ms=6, label=l)
         for k, l in (("ours", "OUTPOST"), ("rerun", "baseline re-run by us"),
                      ("published", "as published"))]
    fig.subplots_adjust(bottom=0.26)
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=3,
               fontsize=7, handletextpad=0.5, columnspacing=1.2)
    F.save(fig, "yelp_comparison")


# --------------------------------------------------------------------------
def _published_yelp():
    """(name, AUC-PR, AUC-ROC) of the best published Yelp method, read not typed."""
    import csv as _csv
    best = None
    for r in _csv.DictReader(open("analysis/tables/published_baselines.csv")):
        v = (r.get("Yelp_AUC-PR") or "").strip()
        if v in ("", "--", "nan") or r["Method"].upper().startswith("OUTPOST"):
            continue
        if best is None or float(v) > best[1]:
            best = (r["Method"], float(v), float(r["Yelp_AUC-ROC"]))
    if best is None:
        raise SystemExit("no published Yelp baseline to compare against")
    return best


def fig_simsample():
    """A placebo removes the whole SimSample gain, so the similarity order is the cause.

    Source: results/results.csv (yelp arms)   Claim: METHODOLOGY 5.5 (P14, P17)
    """
    import make_figures as G
    F.use()
    d = G._runs()
    # short tick labels: six long ones do not fit across a single-column panel,
    # and the axis label carries what they mean
    ARMS = [("off", "C_nosim"), ("placebo", "C_simplacebo"),
            ("0.25", "D_sim0.25"), ("0.5", "D_sim0.5"),
            ("0.75", "D_sim0.75"), ("1.0", "A_main")]
    cells = [(l, G._cell(d, "yelp", t)) for l, t in ARMS]
    cells = [(l, c) for l, c in cells if c]
    if len(cells) < 4:
        raise SystemExit("simsample figure: fewer than four yelp arms present")
    pub_name, pub_pr, pub_roc = _published_yelp()
    labs = [f"{l}\nn={c[2]}" for l, c in cells]
    ctrl = [(l in ("off", "placebo")) for l, _ in cells]
    col = [F.GREY if k else F.CRIMSON for k in ctrl]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for ax, j, name, pub in ((axes[0], 1, "AUC-PR", pub_pr),
                             (axes[1], 0, "AUC-ROC", pub_roc)):
        v = [c[j] for _, c in cells]
        ax.bar(range(len(v)), v, color=col, width=0.7, zorder=3, **F.EDGE)
        ax.axhline(pub, ls=(0, (4, 2)), color=F.PLUM, lw=1.1, zorder=2)
        ax.set_xticks(range(len(v)))
        ax.set_xticklabels(labs, fontsize=6.4)
        ax.set_xlabel("similarity fraction")
        ax.set_ylabel(f"Yelp {name}")
        lo, hi = min(v + [pub]), max(v + [pub])
        ax.set_ylim(lo - 0.035, hi + 0.045)
        for i, x in enumerate(v):
            ax.annotate(f"{x:.4f}", (i, x), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=6.2, color=F.INK)
    h = [plt.Line2D([], [], marker="s", ls="none", mfc=F.CRIMSON, mec=F.INK, ms=6,
                    label="similarity-ordered sampling"),
         plt.Line2D([], [], marker="s", ls="none", mfc=F.GREY, mec=F.INK, ms=6,
                    label="control (off, or shuffled)"),
         plt.Line2D([], [], color=F.PLUM, lw=1.1, ls=(0, (4, 2)),
                    label=f"best published ({pub_name})")]
    fig.subplots_adjust(bottom=0.30, wspace=0.32)
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3,
               fontsize=7, handletextpad=0.5, columnspacing=1.2)
    F.save(fig, "simsample_mechanism")


# --------------------------------------------------------------------------
def fig_detectability_law():
    """Detectability tracks how clustered a class is, and propagation is why.

    Source: analysis/tables/spectral_*.csv + detectability_law.json
    Claim: METHODOLOGY 6.1-6.2 (see 6.5: descriptive, not predictive)
    """
    import glob as _glob
    import pandas as pd
    rows = []
    for f in sorted(_glob.glob("analysis/tables/spectral_*.csv")):
        ds = os.path.basename(f)[len("spectral_"):-len(".csv")]
        t = pd.read_csv(f); t["dataset"] = ds
        rows.append(t)
    if not rows:
        raise SystemExit("detectability figure: no spectral_*.csv")
    df = pd.concat(rows, ignore_index=True)
    hops = [c for c in df.columns if c.startswith("auc_hop")]
    df["best_auc"] = df[hops].max(axis=1)
    # colour carries the distinction the paper turns on -- how the anomalies came
    # to exist -- and the marker carries the graph, so neither has to do both
    KIND = {"yelp": "real", "amazon": "real", "tfinance": "real",
            "ogbn-arxiv": "OGB", "ogbn-mag": "OGB",
            "photo": "semi-synthetic", "computers": "semi-synthetic",
            "cs": "semi-synthetic"}
    MARK = {"photo": "o", "computers": "s", "cs": "^", "ogbn-arxiv": "v",
            "ogbn-mag": "P", "yelp": "D", "amazon": "X", "tfinance": "*"}
    order = [k for k in MARK if k in set(df.dataset)]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3))

    def draw(ax, ycol):
        for ds in order:
            g = df[df.dataset == ds]
            c = F.KIND[KIND[ds]]
            for hollow, gg in ((False, g[g.smoothing_gain > 0]),
                               (True, g[g.smoothing_gain <= 0])):
                if gg.empty:
                    continue
                ax.scatter(gg.med_same_frac, gg[ycol], marker=MARK[ds],
                           s=58 if ds in ("yelp", "amazon", "tfinance") else 40,
                           facecolor="none" if hollow else c,
                           edgecolor=c if hollow else F.INK,
                           linewidth=1.3 if hollow else 0.9, zorder=4)

    draw(axes[0], "best_auc")
    law = load("analysis/tables/detectability_law.json")
    tr = law.get("two_regime", {})
    if "propagation_fit" in tr:
        pf = tr["propagation_fit"]
        prop = df[df.smoothing_gain > 0]
        xs = np.linspace(prop.med_same_frac.min(), prop.med_same_frac.max(), 50)
        axes[0].plot(xs, pf["intercept"] + pf["slope"] * xs, color=F.INK, lw=1.1,
                     ls=(0, (4, 2)), zorder=3)
    axes[0].set_xlabel("same-class neighbour fraction")
    axes[0].set_ylabel("best prototype AUC-ROC")
    axes[1].axhline(0, color=F.INK, lw=0.9, zorder=2)
    draw(axes[1], "smoothing_gain")
    axes[1].set_xlabel("same-class neighbour fraction")
    axes[1].set_ylabel("propagation AUC gain")
    h = [plt.Line2D([], [], marker=MARK[k], ls="none", mfc=F.KIND[KIND[k]],
                    mec=F.INK, ms=6, label=F.nice(k)) for k in order]
    h.append(plt.Line2D([], [], marker="o", ls="none", mfc="none", mec=F.INK, ms=6,
                        label="hollow: propagation hurts"))
    fig.subplots_adjust(bottom=0.30, wspace=0.28)
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=5,
               fontsize=6.6, handletextpad=0.4, columnspacing=0.9)
    F.save(fig, "detectability_law")


# --------------------------------------------------------------------------
def fig_regime():
    """SimSample's effect tracks neither degree nor sampling budget.

    Source: analysis/tables/simsample_by_dataset.json   Claim: METHODOLOGY 5.5 (P14, P17)
    """
    J = load("analysis/tables/simsample_by_dataset.json")
    order = sorted(J, key=lambda k: J[k]["median_degree"])
    deg = [J[k]["median_degree"] for k in order]
    above = [100 * J[k]["frac_above_budget25"] for k in order]
    eff = [J[k]["sim_delta_roc"]["delta"] if J[k]["sim_delta_roc"] else np.nan
           for k in order]
    nn = [J[k]["sim_delta_roc"]["n"] if J[k]["sim_delta_roc"] else 0 for k in order]
    BUDGET = 25          # hop-1 fan-out; the mechanism under test
    x = np.arange(len(order))
    # stacked, sharing one x axis: eight dataset names will not fit three times
    # across a page, and stacking lines the three quantities up per graph
    fig, axes = plt.subplots(3, 1, figsize=(3.9, 5.2), sharex=True)
    col = [F.CRIMSON if v > BUDGET else F.TEAL for v in deg]
    axes[0].bar(x, [max(v, 0.5) for v in deg], width=0.72, color=col,
                zorder=3, **F.EDGE)
    axes[0].axhline(BUDGET, ls=(0, (4, 2)), color=F.INK, lw=1.0, zorder=4)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("median degree")
    axes[0].annotate(f"budget {BUDGET}", (-0.5, BUDGET), xytext=(2, 3),
                     textcoords="offset points", ha="left", va="bottom",
                     fontsize=6.4, color=F.INK)
    # colour means ONE thing in all three panels: whether the graph's median degree
    # is above the sampling budget. Colouring panel 3 by the sign of the effect made
    # a graph change colour between panels and hid the very mismatch this figure is
    # about -- Amazon is above budget and its effect is negative.
    axes[1].bar(x, above, width=0.72, color=col, zorder=3, **F.EDGE)
    axes[1].set_ylabel("% above budget")
    axes[2].axhline(0, color=F.INK, lw=0.9, zorder=2)
    axes[2].bar(x, eff, width=0.72, color=col, zorder=3, **F.EDGE)
    lo, hi = min(eff), max(eff)
    span = hi - lo
    axes[2].set_ylim(lo - 0.34 * span, hi + 0.34 * span)
    axes[2].set_ylabel("$\\Delta$ AUC-ROC (on $-$ off)")
    # each value sits centred on its own bar, outside it. They are NOT run through
    # place_labels: with one bar per slot there is room, and letting labels drift to
    # avoid each other left them hovering over the wrong bar with nothing to say so.
    for i, (v, n) in enumerate(zip(eff, nn)):
        axes[2].annotate(f"{v:+.4f}\nn={n}", (i, v), xytext=(0, 4 if v >= 0 else -4),
                         textcoords="offset points", ha="center",
                         va="bottom" if v >= 0 else "top", fontsize=5.6,
                         color=F.INK, zorder=6)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([F.nice(k) for k in order], fontsize=6.4, rotation=45,
                             ha="right", rotation_mode="anchor")
    # "Computers" is the longest name and at 45 degrees it reaches back into
    # "Photo"; 55 is the angle at which all eight clear each other at this width
    for t_ in axes[-1].get_xticklabels():
        t_.set_rotation(55)
    h = [plt.Line2D([], [], marker="s", ls="none", mfc=F.CRIMSON, mec=F.INK, ms=6,
                    label=f"median degree above {BUDGET}"),
         plt.Line2D([], [], marker="s", ls="none", mfc=F.TEAL, mec=F.INK, ms=6,
                    label=f"median degree below {BUDGET}")]
    fig.subplots_adjust(bottom=0.19, hspace=0.26)
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2,
               fontsize=7, handletextpad=0.5, columnspacing=1.2)
    F.save(fig, "regime_decomposition")


# --------------------------------------------------------------------------
def fig_inflation():
    """Oracle selection taxes benchmarks by how they were built, with no overlap.

    Source: analysis/tables/inflation_ordering.json  Claim: METHODOLOGY 3.2
    """
    d = load("analysis/tables/inflation_ordering.json")
    order = ["semi-synthetic", "OGB", "real"]
    rows = sorted(d["datasets"], key=lambda r: (order.index(r["kind"]), -r["inflation"]))
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    # the artifact stores only mean, sd and n per dataset -- no per-seed values --
    # so this draws mean +/- sd, never a simulated cloud of individual runs
    for i, r in enumerate(rows):
        c = F.KIND[r["kind"]]
        ax.errorbar([r["inflation"]], [i], xerr=[r["sd"] or 0.0], fmt="D", ms=5.5,
                    color=c, ecolor=F.INK, elinewidth=0.9, capsize=2.4,
                    zorder=4, **F.MEDGE)
    # the separation is between dataset means: shade the empty gap each kind
    # boundary leaves and label how wide it is
    seps = [x for x in d["separation"] if x["complete_separation"]]
    for sep in seps:
        lo, hi = sep["max_lower"], sep["min_higher"]
        ax.axvspan(lo, hi, color=F.GREY, alpha=0.16, zorder=1)
        # The gap widths are NOT annotated in the figure. The narrow real|OGB band is
        # 0.003 wide and sits exactly on the two OGB markers, so a label inside it
        # covers data and a label below it needs a leader that crosses the other
        # band's label. Both numbers are in the caption, generated from this file.
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([F.nice(r["dataset"]) for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.35)
    ax.set_xlabel("oracle inflation in AUC-ROC")
    h = [plt.Line2D([], [], marker="D", ls="none", mfc=F.KIND[k], mec=F.INK, ms=6,
                    label=k) for k in order]
    h.append(plt.Line2D([], [], color=F.GREY, lw=6, alpha=0.4, label="gap between kinds"))
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
              handletextpad=0.5, columnspacing=1.0)
    F.save(fig, "oracle_inflation_separation")


# --------------------------------------------------------------------------
def fig_p8_diagnostic_vs_trained():
    """A class's training-free advantage does not survive training; its stability does.

    Source: analysis/tables/p8_per_class.json + analysis/tables/spectral_ogbn-mag.csv
    Claim: METHODOLOGY 6.6 (P8)
    """
    import csv
    d = load("analysis/tables/p8_per_class.json")
    diag = {}
    with open("analysis/tables/spectral_ogbn-mag.csv") as fh:
        for row in csv.DictReader(fh):
            diag[int(row["class"])] = float(row["auc_hop0"])
    rows = [r for r in d["per_class"] if r["class"] in diag]
    if not rows:
        raise SystemExit("no overlap between p8 classes and the mag diagnostic")
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    hero_lab = "P8 class"
    for r in rows:
        c = r["class"]
        hero = (c == (d.get("class_263") or {}).get("class", 263))
        ax.errorbar(diag[c], r["mean_auc"], yerr=r["sd"] or 0, fmt="none",
                    ecolor=F.INK if hero else F.GREY, elinewidth=1.1 if hero else 0.7,
                    capsize=2, zorder=3)
        ax.scatter([diag[c]], [r["mean_auc"]], s=64 if hero else 34,
                   marker="D" if hero else "o",
                   color=F.CRIMSON if hero else "white", zorder=4, **F.EDGE)
        if hero:
            hero_lab = f"class {c}: rank {r['rank']} of {len(rows)}"
    # the reference is not parity -- the two axes are different quantities -- but
    # the constant the diagnostic has to beat to carry any information at all
    mbar = float(np.mean([r["mean_auc"] for r in rows]))
    xs = [diag[r["class"]] for r in rows]
    ax.axhline(mbar, color=F.INK, lw=1.0, ls=(0, (4, 2)), zorder=2)
    # park the line's label in the widest gap along x so it never lands on a point
    # both reference labels go in the corner: the cloud is dense enough that any
    # inline placement lands on a point
    leg = [plt.Line2D([], [], color=F.INK, lw=1.0, ls=(0, (4, 2)),
                      label="mean over classes"),
           plt.Line2D([], [], marker="D", ls="none", mfc=F.CRIMSON, mec=F.INK,
                      ms=6, label=hero_lab)]
    c = d.get("diagnostic_vs_trained")
    if c:
        h = c["against"]["auc_hop0"] if "against" in c else c
        leg.insert(0, plt.Line2D([], [], ls="none", marker="", label=(
            f"Spearman $\\rho$ = {h['spearman_rho']:+.3f} (p = {h['p_value']:.2f})")))
    # below the axes, not in a corner: in the corner its box covered the lower half
    # of two error bars
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1,
              fontsize=7, handletextpad=0.5, borderpad=0.4, labelspacing=0.3)
    ax.set_xlabel("training-free detectability (hop-0 AUC)")
    ax.set_ylabel("trained AUC-ROC, unseen")
    F.save(fig, "p8_diagnostic_vs_trained")


FIGS = {"selection_noise": fig_selection_noise,
        "law_crossval": fig_law_crossval,
        "mechanism": fig_mechanism,
        "inflation": fig_inflation,
        "p8": fig_p8_diagnostic_vs_trained,
        "efficiency": fig_efficiency,
        "yelp": fig_yelp_comparison,
        "simsample": fig_simsample,
        "law": fig_detectability_law,
        "regime": fig_regime}

F.use()          # at import: anything that imports this module gets the paper style

if __name__ == "__main__":
    want = sys.argv[1:] or list(FIGS)
    for k in want:
        if k not in FIGS:
            raise SystemExit(f"unknown figure {k}; have {list(FIGS)}")
        FIGS[k]()
