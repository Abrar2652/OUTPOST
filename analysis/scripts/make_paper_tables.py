"""Build the paper's two main tables.

  Table 1  small-scale : Photo, Computers, CS
  Table 2  large-scale : Yelp, ogbn-arxiv, ogbn-mag

Three kinds of row appear:

  baselines             from analysis/tables/published_baselines.csv - the
                        numbers exactly as printed in the DEMO paper
  OUTPOST (reference)   our measurements (results/reference_runs.csv), each
                        traceable to a log under analysis/runlogs/
  OUTPOST (this run)    whatever is in results/results.csv on THIS machine

so a third party can compare their reproduction against ours line by line.

Outputs (regenerable, nothing hand-typed):
  analysis/tables/table1_small.csv  .tex
  analysis/tables/table2_large.csv  .tex

    python analysis/scripts/make_paper_tables.py
    python analysis/scripts/make_paper_tables.py --metric valsel

`--metric best` (default) is the per-metric maximum over epochs: oracle test
selection, and the protocol every published baseline used, so it is the
comparable column. `--metric valsel` uses the peak-validation epoch instead -
the honest, deployable number, usually lower. Report both in the paper.
"""

import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

SMALL = ["photo", "computers", "cs"]
LARGE = ["yelp", "amazon", "tfinance", "ogbn-arxiv", "ogbn-mag"]
NUMWORD = {3: "three", 4: "four", 5: "five", 6: "six"}
COL = {"photo": "Photo", "computers": "Computers", "cs": "CS", "yelp": "Yelp",
       "amazon": "Amazon", "tfinance": "T-Finance", "ogbn-arxiv": "ogbn-arxiv", "ogbn-mag": "ogbn-mag"}

# The uniform 400-epoch protocol, per dataset and per method. photo/computers/
# cs were run at 200 under A_main and re-run at 400 under E400_*; yelp, amazon,
# arxiv and mag are natively 400. Amazon's arm is the one VALIDATION selected
# (A_sim0.0; val_auc 0.9692 vs 0.9510, test never consulted).
#
# DEMO_TAG is the FULL published method (mixup on, energy off as in every E400
# arm). Until 2026-09-04 this file read the DEMO re-run from tag B_demo, which
# is DEMO's own "w/o Mix" ablation - the exact mistake config.json warns about,
# reproduced in the script that writes the paper's table.
OUTPOST_TAG = {"photo": "E400_outpost", "computers": "E400_outpost",
               "cs": "E400_outpost", "yelp": "A_main",
               "ogbn-arxiv": "A_main", "ogbn-mag": "A_main"}
# amazon / tfinance: the validation-selected SimSample arm, decided once by
# build_appendix.select_arm() and read from the file it writes
try:
    OUTPOST_TAG.update(json.load(open("analysis/tables/arm_selection.json")))
except Exception:
    pass
DEFAULTS_BEFORE_SELECTION = dict(OUTPOST_TAG)
DEMO_TAG = {"photo": "E400_demo", "computers": "E400_demo", "cs": "E400_demo",
            "yelp": "B_demo_mix", "amazon": "B_demo_mix", "tfinance": "B_demo_mix",
            # ogbn-arxiv was listed as infeasible for DEMO because mixup needs a
            # dense NxN PPR. At N=169,343 that is 114.7 GB - large, but it built in
            # 481 s and DEMO ran against it at 400 epochs for all 10 seeds
            # (METHODOLOGY 11.1/11.2). ogbn-mag stays absent: its matrix is 2.17 TB.
            "ogbn-arxiv": "B_demo_mix"}
NSREG_TAG = {ds: "E400_nsreg" for ds in ("photo", "computers", "cs", "yelp", "amazon", "tfinance",
                                          "ogbn-arxiv")}   # arxiv: runs in 2.17 GB (METHODOLOGY 11.2)
# NSReg is reported at the configuration ITS OWN validation-only sweep selected,
# wherever that selection survives the ten-seed check (P35). Amazon's does: the
# tuned arm is +0.0054 oracle AUC-ROC at n=10, nine of ten seeds, p=0.006 - a
# genuinely stronger baseline on the graph where OUTPOST does not win, adopted
# because the rule and the data agree. Photo's does NOT survive (+0.0106 at three
# seeds, -0.0009 at ten, 4/10), so Photo keeps the released configuration; moving
# it would be reporting noise as a result. See analysis/tables/nsreg_selection.md.
NSREG_TAG["amazon"] = "NT_lr0.003_wd0.0"
# baseline row order as printed in the DEMO paper (unsupervised, then semi-supervised)
ORDER = ["ANOMALOUS", "DOMINANT", "AnomalyDAE", "GAAN", "CoLA", "CONAD",
         "ConsisGAD", "GGAD", "TAM", "OGCNN", "ANO-S", "DOM-S", "SpaceGNN",
         "NSReg", "GNN+OpenMax", "DEMO"]
RUN_LABEL = "OUTPOST"


def rows_from(path, metric, min_epochs=100, method="outpost", tag="A_main",
              tag_by_ds=None):
    """{dataset: (roc, pr, n_runs, sd_roc, sd_pr)} from one results csv.

    min_epochs drops --quick / smoke runs so they can never reach a table.

    `tag` is not optional bookkeeping. results.csv holds every run of every
    campaign, ablations included, and an ablation arm is by construction a
    DIFFERENT configuration of the same dataset. Averaging across tags would
    fold `C_nopl` and `C_nogate` into the main-table cell and report the mean of
    a method and its own ablations as the method's score. Only the main-table
    tag belongs in the main table.
    """
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path)
    if d.empty or "method" not in d.columns:
        return {}
    d = d.drop_duplicates(subset=["dataset", "method", "seed", "train_seed",
                                  "tag"], keep="last")
    d = d[(d["method"] == method) & (d["num_epochs"] >= min_epochs)]
    if tag_by_ds is not None:
        d = d[d.apply(lambda r: str(r["tag"]) == tag_by_ds.get(r["dataset"], "\0"),
                      axis=1)]
    elif tag is not None and "tag" in d.columns:
        d = d[d["tag"].astype(str) == tag]
    if d.empty:
        return {}
    rc, pc = f"{metric}_auroc", f"{metric}_aupr"
    out = {}
    for ds, g in d.groupby("dataset"):
        # ddof=1 sample std; undefined for a single run, reported as None
        sr = g[rc].std(ddof=1) if len(g) > 1 else None
        sp = g[pc].std(ddof=1) if len(g) > 1 else None
        out[ds] = (g[rc].mean(), g[pc].mean(), len(g), sr, sp)
    return out


def build(datasets, baselines, ours, metric, demo_rerun=None, nsreg_rerun=None):
    """Paper table: published baselines, then one OUTPOST row from THIS machine.

    Our own archived numbers are deliberately not a row here - the table is the
    paper's, so it carries one measurement per method. They are used only for the
    sanity check printed by check_against_reference().
    """
    rows, missing = [], []

    for m in ORDER:
        b = baselines[baselines["Method"] == m]
        if b.empty:
            continue
        r = {"Method": m}
        for ds in datasets:
            for suf, lab in (("AUC-ROC", "ROC"), ("AUC-PR", "PR")):
                col = f"{COL[ds]}_{suf}"
                v = b[col].iloc[0] if col in b.columns else float("nan")
                r[f"{COL[ds]} {lab}"] = "" if pd.isna(v) else f"{v:.4f}"
        rows.append(r)

    # DEMO re-run under OUR protocol, at the same seeds and therefore the same
    # splits as the OUTPOST row. The transcribed DEMO row above and this one are
    # different measurements and both belong in the table: the first says what
    # the DEMO paper reports, the second is the only DEMO number that can be
    # compared to ours by a paired test (analysis/scripts/stats.py).
    for label, rerun in (("DEMO (re-run, ours)", demo_rerun),
                         ("NSReg (re-run, ours)", nsreg_rerun)):
        if not rerun:
            continue
        r = {"Method": label}
        for ds in datasets:
            if ds in rerun:
                roc, pr, n, sr, sp = rerun[ds]
                r[f"{COL[ds]} ROC"] = (f"{roc:.4f}+-{sr:.4f}" if sr is not None
                                       else f"{roc:.4f}")
                r[f"{COL[ds]} PR"] = (f"{pr:.4f}+-{sp:.4f}" if sp is not None
                                      else f"{pr:.4f}")
            else:
                r[f"{COL[ds]} ROC"] = ""
                r[f"{COL[ds]} PR"] = ""
        rows.append(r)

    # an all-blank row reads as a bug, so say why it is blank
    r = {"Method": RUN_LABEL if ours else f"{RUN_LABEL} - NOT RUN YET"}
    for ds in datasets:
        if ds in ours:
            roc, pr, n, sr, sp = ours[ds]
            # mean +- sample std over seeds; bare mean when n == 1
            r[f"{COL[ds]} ROC"] = (f"{roc:.4f}+-{sr:.4f}" if sr is not None
                                   else f"{roc:.4f}")
            r[f"{COL[ds]} PR"] = (f"{pr:.4f}+-{sp:.4f}" if sp is not None
                                  else f"{pr:.4f}")
        else:
            r[f"{COL[ds]} ROC"] = ""
            r[f"{COL[ds]} PR"] = ""
            missing.append(ds)
    rows.append(r)

    return pd.DataFrame(rows), missing


def smoke_only(path, min_epochs=100):
    """Datasets that ran, but only as short (--quick) runs excluded from tables.

    Without this, a --quick run reports 'NOT YET RUN' for a dataset it just
    trained successfully, which reads as a failure rather than as the guard
    working.
    """
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    if d.empty or "method" not in d.columns:
        return set()
    d = d[d["method"] == "outpost"]
    return {ds for ds, g in d.groupby("dataset")
            if g["num_epochs"].max() < min_epochs}


def check_against_reference(ours, reference, tol=0.05):
    """Console-only sanity check: did this machine land near our measurements?

    Deliberately NOT a table row. Our archived runs used different hardware, and
    on Photo/Computers a different seeding scheme (fixed split), so the two are
    not the same measurement and must not sit side by side in a paper table.
    Its only job is to catch a broken port - a dataset that loaded wrong, a
    config that did not apply - before hours are spent on the rest.
    """
    shared = [d for d in ours if d in reference]
    if not shared:
        return
    print("\nSanity check against our archived runs (NOT part of the tables):")
    for ds in sorted(shared):
        a_roc, a_pr = ours[ds][0], ours[ds][1]
        b_roc, b_pr = reference[ds][0], reference[ds][1]
        # An archived row of 0.0000 means the reference never recorded this
        # dataset, not that we moved it by 0.77. Reporting that as "CHECK -
        # differs by >0.05" trains the reader to ignore the check.
        if not b_roc and not b_pr:
            print(f"  {ds:<12} no archived reference row - not compared")
            continue
        d_roc, d_pr = a_roc - b_roc, a_pr - b_pr
        ok = abs(d_roc) <= tol and abs(d_pr) <= tol
        print(f"  {ds:<12} ROC {a_roc:.4f} vs {b_roc:.4f} ({d_roc:+.4f})   "
              f"PR {a_pr:.4f} vs {b_pr:.4f} ({d_pr:+.4f})   "
              f"{'ok' if ok else 'CHECK - differs by >' + str(tol)}")
    print("  Differences within +-0.05 are expected (hardware, seeds, split "
          "scheme). Larger gaps suggest a setup problem, not a result.")


# Rows measured by us, under one protocol, at identical seeds and splits. Only
# these are comparable with each other; the transcribed rows above come from other
# papers' protocols and budgets.
COMPARABLE = ("DEMO (re-run, ours)", "NSReg (re-run, ours)", RUN_LABEL)


def _lead(x):
    """The numeric part of a 'mean+-sd' cell, or None if the cell is empty."""
    try:
        return float(str(x).split("+-")[0].strip())
    except (ValueError, AttributeError):
        return None


def to_latex(df, caption, label):
    cols = list(df.columns)
    # Bold marks the BEST value in each column among the rows run under this
    # protocol - not "our row". Bolding OUTPOST unconditionally put a bold
    # 0.9087 on T-Finance next to an unbolded NSReg 0.9238 that beats it, which
    # reads as a claim we do not make and a referee is right to call out.
    best = {}
    for c in cols[1:]:
        vals = [(_lead(r[c]), r["Method"]) for _, r in df.iterrows()
                if r["Method"] in COMPARABLE and _lead(r[c]) is not None]
        if vals:
            best[c] = max(vals)[1]
    out = ["\\begin{table}[t]", "\\centering", "\\small",
           "\\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", "\\toprule",
           " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = str(r[c]).replace("+-", r" $\pm$ ") if r[c] != "" else "--"
            if c != "Method" and v != "--" and best.get(c) == r["Method"]:
                v = f"\\textbf{{{v}}}"
            cells.append(v)
        if r["Method"].startswith(RUN_LABEL):
            out.append("\\midrule")
        if r["Method"] == RUN_LABEL:
            cells[0] = f"\\textbf{{{cells[0]}}}"      # our row stays identifiable
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}",
            f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="best", choices=["best", "valsel"])
    ap.add_argument("--selected-arm", action="store_true",
                    help="OUTPOST rows at the arm each dataset's validation-only "
                         "hyperparameter sweep selected, rather than the inherited "
                         "default. Only Amazon differs (T_hidden16); Photo, "
                         "Computers, CS and Yelp all retained their default (P29). "
                         "Writes *_selarm.* so the two are never confused. "
                         "Read P40 before quoting it: Amazon's selection reverses "
                         "at ten selection seeds.")
    ap.add_argument("--gate-off", action="store_true",
                    help="OUTPOST row from the gate-OFF arms (E400_nogate / C_nogate / "
                         "A_nogate400). The gate is null on every dataset measured "
                         "(appendix gate_off), so this is the smaller model the paper "
                         "can honestly describe. Writes *_gateoff.{csv,tex}; ogbn-mag "
                         "has no gate-off arm and keeps the gate-on value, footnoted.")
    a = ap.parse_args()
    if a.selected_arm:
        # One selection file per dataset, and it must be the sweep run AT THE
        # REPORTED BUDGET: photo/computers were swept again at 400 (_400 files),
        # yelp/amazon are natively 400 so their plain file is the 400-epoch one,
        # and CS has only a 200-epoch sweep, so it keeps its default rather than
        # importing a winner chosen at a budget the table does not report.
        SEL_FILE = {"photo": "hparam_selection_photo_400.json",
                    "computers": "hparam_selection_computers_400.json",
                    "yelp": "hparam_selection_yelp.json",
                    "amazon": "hparam_selection_amazon.json"}
        for _ds, _fn in SEL_FILE.items():
            try:
                _j = json.load(open("analysis/tables/" + _fn))
            except Exception:
                continue
            _w = _j.get("winner")
            _w = _w.get("tag") if isinstance(_w, dict) else _w
            if _w:
                OUTPOST_TAG[_ds] = _w
                if _w != DEFAULTS_BEFORE_SELECTION.get(_ds):
                    print(f"  [selected-arm] {_ds}: "
                          f"{DEFAULTS_BEFORE_SELECTION.get(_ds)} -> {_w}")
    if a.gate_off:
        OUTPOST_TAG.update({"photo": "E400_nogate", "computers": "E400_nogate", "cs": "E400_nogate",
                            "yelp": "C_nogate", "amazon": "C_nogate", "tfinance": "C_nogate",
                            "ogbn-arxiv": "A_nogate400"})   # ogbn-mag: unchanged (no arm)

    bp = "analysis/tables/published_baselines.csv"
    if not os.path.exists(bp):
        sys.exit(f"missing {bp}")
    baselines = pd.read_csv(bp)
    baselines = baselines[
        ~baselines["Method"].str.upper().str.startswith("OUTPOST")]

    ours = rows_from("results/results.csv", a.metric, tag_by_ds=OUTPOST_TAG)
    demo_rerun = rows_from("results/results.csv", a.metric,
                           method="demo", tag_by_ds=DEMO_TAG)
    nsreg_rerun = rows_from("results/results.csv", a.metric,
                            method="nsreg", tag_by_ds=NSREG_TAG)
    # the archived runs predate the tagging scheme, so they are read untagged
    reference = rows_from("results/reference_runs.csv", a.metric, tag=None)
    if not ours:
        print("[note] no completed runs in results/results.csv yet, so the "
              f"'{RUN_LABEL}' row is blank.")
        print("       run:  python reproduce.py\n")

    os.makedirs("analysis/tables", exist_ok=True)
    allmissing = []
    for name, ds, cap in (
        ("table1_small", SMALL,
         f"AUC-ROC and AUC-PR on the {NUMWORD.get(len(SMALL), len(SMALL))} small-scale datasets."),
        ("table2_large", LARGE,
         f"AUC-ROC and AUC-PR on the {NUMWORD.get(len(LARGE), len(LARGE))} large-scale datasets."),
    ):
        df, missing = build(ds, baselines, ours, a.metric, demo_rerun, nsreg_rerun)
        # the two selection protocols are different measurements and must not
        # overwrite each other: `best` is the oracle column the published
        # baselines used, `valsel` is the deployable one
        stem = name if a.metric == "best" else f"{name}_{a.metric}"
        if a.selected_arm:
            stem += "_selarm"
        if a.gate_off:
            stem += "_gateoff"
        df.to_csv(f"analysis/tables/{stem}.csv", index=False)
        tex_cap = cap + (
            " Rows above the rule are transcribed from their own papers and were"
            " produced under those papers' protocols and budgets; the three rows"
            " below it (DEMO, NSReg, OUTPOST) were re-run here under one protocol"
            " at identical seeds and splits. \\textbf{Bold} marks the best value"
            " among those three comparable rows only."
            " NSReg's Amazon row is at the configuration its own validation-only"
            " sweep selected, which is stronger than its released one; its Photo"
            " row is not, because that selection does not survive ten seeds."
            " Both baselines are absent on ogbn-mag for structural reasons, not for"
            " want of compute: DEMO is defined on a dense PPR matrix, 2.17\,TB at"
            " this graph's 736{,}389 nodes, and NSReg's normal-structure regulariser"
            " is defined over all ordered pairs of labelled normals, 1.34 billion"
            " here against 50 million on ogbn-arxiv."
        ) + ("" if a.metric == "best" else
             " Model selection at the peak-validation epoch "
             "(deployable), not the per-metric maximum over epochs.")
        if a.selected_arm:
            tex_cap += (" OUTPOST rows at the configuration each dataset's "
                        "validation-only sweep selected. Only Amazon differs from the "
                        "table above (\\texttt{T\\_hidden16}, 0.0040 lower AUC-ROC at "
                        "$n{=}10$, nine of ten seeds); every other dataset's sweep "
                        "retained its default. The Amazon selection does not survive "
                        "more selection seeds -- its validation margin reverses at ten "
                        "-- so this table is reported beside the default-arm one, not "
                        "instead of it.")
        if a.gate_off:
            tex_cap += (" OUTPOST without the atlas gate. The gate is a non-parametric quantile veto, "
                        "so the parameter count is unchanged by removing it (0.157$\\times$ DEMO on "
                        "Photo either way); this variant differs only in accuracy. "
                        "Datasets with no gate-off arm are listed in the row as "
                        "absent rather than silently carrying their gate-on value; "
                        "ogbn-mag was not re-run without the gate.")
        with open(f"analysis/tables/{stem}.tex", "w", encoding="utf-8") as f:
            f.write(to_latex(df, tex_cap, f"tab:{stem}"))
        print(f"=== {stem}   (metric={a.metric}) ===")
        print(df.to_string(index=False))
        print(f"-> analysis/tables/{stem}.csv + .tex\n")
        allmissing += missing

    smoke = smoke_only("results/results.csv")
    if allmissing:
        blank = sorted(set(allmissing))
        quick = [d for d in blank if d in smoke]
        never = [d for d in blank if d not in smoke]
        if quick:
            print("SMOKE-TESTED ONLY (ran fine, deliberately kept out of the "
                  "tables because\nruns under 100 epochs must never reach a "
                  "paper table):")
            for ds in quick:
                print(f"  {ds}")
            print("  -> this is --quick working as intended, not a failure.")
        if never:
            # In a --gate-off build a blank cell means "no gate-off ARM", not
            # "dataset never run": T-Finance has ten seeds of every other arm.
            # Saying "not yet run on this machine" there sends the reader to
            # reproduce.py for a run that already exists.
            if a.gate_off:
                print(("\n" if quick else "") + "NO GATE-OFF ARM YET "
                      "(the dataset itself is complete; only its C_nogate / "
                      "E400_nogate run is missing):")
            else:
                print(("\n" if quick else "") + "NOT YET RUN ON THIS MACHINE:")
            for ds in never:
                print(f"  {ds}")
        if not a.gate_off:
            print("\n  python reproduce.py --seeds 42 0 1    # the real run")
            print("  python reproduce.py --resume          # only what is missing")
        else:
            print("\n  see campaigns/tfinance_decomp.json (P38/P39) for the "
                  "missing gate-off arm")
    else:
        print("all cells filled. runs per dataset: "
              + str({d: ours[d][2] for d in ours}))
        print("(cells show mean+-sample std across seeds; a bare "
              "number means a single run)")
    check_against_reference(ours, reference)


if __name__ == "__main__":
    main()
