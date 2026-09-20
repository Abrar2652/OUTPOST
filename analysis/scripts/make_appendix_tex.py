"""Every appendix table as LaTeX, from analysis/appendix.json - nothing typed.

    python analysis/scripts/make_appendix_tex.py   # -> analysis/tables/tex/*.tex
"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
A = json.load(open("analysis/appendix.json"))
OUT = "analysis/tables/tex"; os.makedirs(OUT, exist_ok=True)


def esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def table(name, caption, label, header, rows, align=None, note=None):
    align = align or "l" + "c" * (len(header) - 1)
    body = " \\\\\n".join(" & ".join(r) for r in rows) + " \\\\"
    tex = (f"\\begin{{table}}[t]\n\\centering\n\\small\n\\begin{{tabular}}{{{align}}}\n\\toprule\n"
           f"{' & '.join(header)} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
           f"\\caption{{{caption}{(' ' + note) if note else ''}}}\n\\label{{tab:{label}}}\n\\end{{table}}\n")
    open(f"{OUT}/{name}.tex", "w", encoding="utf-8").write(tex)
    return name


def p_fmt(r):
    if r is None:
        return "--"
    p = r["p_wilcoxon"]
    star = "$^{*}$" if p < 0.05 else ""
    return f"{p:.3f}{star}"


def d_fmt(r):
    return "--" if r is None else f"{r['delta']:+.4f} ({r['W']}/{r['T']}/{r['L']})"


made = []
# 1. three methods, two rules (all-anomaly ROC)
rows = []
for r in A["three_method_ranking"]:
    o, v = r["oracle"], r["valsel"]
    ro = " $>$ ".join(f"{esc(m)} {o['means'][m]:.4f}" for m in o["ranking"])
    rv = " $>$ ".join(f"{esc(m)} {v['means'][m]:.4f}" for m in v["ranking"])
    rows.append([esc(r["dataset"]), ro, rv, "\\textbf{flips}" if r["ranking_changes"] else "same"])
made.append(table("three_method_ranking",
    "Three methods under one protocol, ranked two ways (AUC-ROC over all anomalies, mean over 10 seeds). "
    "Oracle: per-metric maximum over epochs, the published protocol. Val-selected: the peak-validation epoch.",
    "three_method", ["Dataset", "Oracle ranking", "Val-selected ranking", "Change"], rows, "llll"))

# 2. paired OUTPOST - DEMO and OUTPOST - NSReg
for key, who in (("paired_vs_demo", "DEMO"), ("paired_vs_nsreg", "NSReg")):
    rows = []
    for r in A[key]:
        if "status" in r:
            rows.append([esc(r["dataset"]), "\\multicolumn{7}{l}{\\emph{" + esc(r["status"]) + "}}"]); continue  # 1 + 7 = 8 columns, full text
        rows.append([esc(r["dataset"]), str(r["oracle ROC"]["n"]),
                     d_fmt(r["oracle ROC"]), p_fmt(r["oracle ROC"]), d_fmt(r["oracle PR"]), p_fmt(r["oracle PR"]),
                     d_fmt(r["valsel ROC"]), p_fmt(r["valsel ROC"])])
    made.append(table(f"paired_vs_{who.lower()}",
        f"OUTPOST minus {who}, paired on seed at identical splits; W/T/L counts seeds, $p$ is the two-sided Wilcoxon signed-rank test ($^{{*}}$: $p<0.05$).",
        f"paired_{who.lower()}", ["Dataset", "$n$", "$\\Delta$ ROC (oracle)", "$p$", "$\\Delta$ PR (oracle)", "$p$", "$\\Delta$ ROC (val-sel.)", "$p$"], rows, "lccccccc"))

# 3. oracle inflation, all anomalies and unseen classes
rows = []
by = {}
for r in A["oracle_inflation"]:
    by.setdefault(r["dataset"], {})[r["metric"]] = r
for ds, m in by.items():
    a = m.get("ROC"); u = m.get("ROC (unseen)")
    rows.append([esc(ds), str(a["n"]) if a else "--", f"{a['oracle']:.4f}" if a else "--", f"{a['valsel']:.4f}" if a else "--",
                 f"{a['inflation']:+.4f}" if a else "--", f"{u['inflation']:+.4f}" if u else "--"])
rows.sort(key=lambda r: -float(r[4]) if r[4] != "--" else 0)
made.append(table("oracle_inflation",
    "Oracle inflation: best-over-epochs minus peak-validation AUC-ROC for OUTPOST, all anomalies and unseen classes only. "
    "Binary graphs (Yelp, Amazon) have no unseen class.",
    "inflation", ["Dataset", "$n$", "Oracle", "Val-selected", "Inflation", "Inflation (unseen)"], rows))

# 4. component decomposition
want = ["C_nogate", "C_noconformal", "C_nopl", "L_lean"]
lab = {"C_nogate": "gate off", "C_noconformal": "conformal off", "C_nopl": "PL off", "L_lean": "gate+PL off"}
rows = []
for ds in ["Computers", "CS", "Photo", "Yelp", "Amazon"]:
    cells = [esc(ds)]
    for arm in want:
        r = next((x for x in A["ablations"] if x["dataset"] == ds and x["arm"] == arm), None)
        cells.append(d_fmt(r["delta_roc"]) if r else "--")
    rows.append(cells)
made.append(table("decomposition",
    "Component decomposition: arm minus full model, AUC-ROC, paired on seed (W/T/L). Negative means the component earns its place.",
    "decomposition", ["Dataset"] + [lab[a] for a in want], rows))

# 5. epoch budget + DEMO gap by budget
rows = [[esc(r["dataset"]), str(r["epochs"]), str(r["n_seeds"]), f"{r['delta_roc']:+.4f} ({r['wins_roc']}/{r['n_seeds']})",
         f"{r['delta_pr']:+.4f}", f"{r['p_pr']:.3f}" if r["p_pr"] is not None else "--"] for r in A["epoch_budget"]]
made.append(table("epoch_budget", "OUTPOST minus DEMO at two training budgets (paired on seed).", "epochs",
                  ["Dataset", "Epochs", "$n$", "$\\Delta$ ROC (wins)", "$\\Delta$ PR", "$p$ (PR)"], rows))
rows = [[esc(r["dataset"]), f"{r['demo_published']:.4f}", f"{r['demo_ours_200']:.4f}", f"{r['demo_ours_400']:.4f}",
         f"{r['gap_200']:+.4f}", f"{r['gap_400']:+.4f}"] for r in A["demo_gap_by_budget"]]
made.append(table("demo_gap", "DEMO reproduction gap against its published AUC-ROC, by training budget.", "demo_gap",
                  ["Dataset", "Published", "Ours @200", "Ours @400", "Gap @200", "Gap @400"], rows))

# 6. NSReg reproduction
rows = [[esc(r["dataset"]), f"{r['published']:.4f}",
         f"{r['ours_201']:.4f} ({r['n_201']})" if r["ours_201"] else "--",
         f"{r['ours_400']:.4f} ({r['n_400']})" if r["ours_400"] else "--",
         f"{r['ours_400']-r['published']:+.4f}" if r["ours_400"] else "--"] for r in A["nsreg_reproduction"]]
made.append(table("nsreg_reproduction", "NSReg from its released code against its published AUC-ROC, at its shipped budget (201 epochs) and at the uniform 400.",
                  "nsreg_repro", ["Dataset", "Published", "Ours @201 ($n$)", "Ours @400 ($n$)", "$\\Delta$ @400"], rows))

# 7. unseen-class paired
rows = []
for r in A["paired_unseen"]:
    for who in ("vs_DEMO", "vs_NSReg"):
        x = r.get(who)
        if not x or not x.get("oracle ROC"):
            continue
        rows.append([esc(r["dataset"]), who[3:], d_fmt(x["oracle ROC"]), d_fmt(x["oracle PR"]), d_fmt(x["valsel ROC"]), d_fmt(x["valsel PR"])])
made.append(table("unseen_paired", "OUTPOST minus baseline on the unseen anomaly classes only (paired on seed, W/T/L).", "unseen",
                  ["Dataset", "vs", "$\\Delta$ ROC (oracle)", "$\\Delta$ PR (oracle)", "$\\Delta$ ROC (val-sel.)", "$\\Delta$ PR (val-sel.)"], rows))

# 8. the gate removed
if A.get("gate_off"):
    rows = [[esc(r["dataset"]), str(r["oracle ROC"]["n"]), d_fmt(r["oracle ROC"]), p_fmt(r["oracle ROC"]),
             d_fmt(r["oracle PR"]), d_fmt(r["valsel ROC"])] for r in A["gate_off"]]
    made.append(table("gate_off", "Removing the atlas gate at the paper's protocol: gate-off minus gate-on, OUTPOST, paired on seed (W/T/L). "
                      "Null on every graph tested. The atlas gate is a \\emph{non-parametric} quantile veto, so this is an accuracy result and not a size one: OUTPOST holds 0.157$\\times$ DEMO's parameters on Photo with the gate on or off. (An earlier caption cited 0.289$\\times$ \"with the gate\"; that figure belongs to the spectral fview gate, which is disabled in every reported arm.)",
                      "gate_off", ["Dataset", "$n$", "$\\Delta$ ROC (oracle)", "$p$", "$\\Delta$ PR (oracle)", "$\\Delta$ ROC (val-sel.)"], rows))
# 9. hyperparameter selection gap (validation-only winner vs best arm on test)
rows = []
for r in A.get("hyperparameter_selection", []):
    rows.append([esc(r["dataset"]), str(r.get("epochs", "")), str(r["n_arms"]), esc(r["val_winner"]),
                 f"{r['val_winner_test_roc']:.4f}" if r.get("val_winner_test_roc") is not None else "--",
                 esc(r["best_test_arm"]), f"{r['best_test_roc']:.4f}",
                 f"{r['selection_gap']:+.4f}" if r.get("selection_gap") is not None else "--"])
if rows:
    made.append(table("hparam_selection",
        "Hyperparameter selection gap: the arm validation chose -- mean val AUC over the three selection seeds, "
        "0.002 tie band, ties broken toward the smaller model and then toward the repository default -- against the "
        "best arm on test. Every arm, the default included, is scored on those same three seeds. The gap is what "
        "tuning on the test split would have bought. Amazon is the only dataset where validation leaves the default: "
        "it picks an arm 0.0071 \\emph{worse} on test, so two-thirds of its 0.0105 gap is the cost of following "
        "validation rather than the reward for searching.",
        "hparam", ["Dataset", "Epochs", "Arms", "Validation picked", "Test ROC", "Best test arm", "Test ROC", "Gap"], rows, "lccl" + "clcc"))

# 10. the same selection rule applied to the baseline (the fairness objection)
rows = []
for r in A.get("baseline_selection", []):
    rows.append([esc(r["dataset"]), esc(r["winner"]),
                 "kept" if r["retained_default"] else "\\textbf{changed}",
                 f"{r['n_in_band']}/{r['n_arms']}",
                 f"{r['val_spread']:.4f}",
                 f"{r['released_oracle']:.4f}" if r.get("released_oracle") is not None else "--",
                 f"{r['selected_oracle']:.4f}" if r.get("selected_oracle") is not None else "--",
                 f"{r['delta_oracle']:+.4f}" if r.get("delta_oracle") is not None else "--",
                 f"{r['delta_valsel']:+.4f}" if r.get("delta_valsel") is not None else "--"])
if rows:
    made.append(table("baseline_selection",
        "The baseline swept under OUTPOST's own selection rule: mean validation AUC over seeds 0, 1 and 42, "
        "0.002 tie band, ties to the released configuration. NSReg, three seeds, 400 epochs; CS excluded for cost. "
        "Validation changes the winner on Photo and Amazon and the baseline gets better in both. On Amazon the same "
        "rule costs OUTPOST 0.0071 oracle AUC-ROC and hands NSReg 0.0047.",
        "baselinesel",
        ["Dataset", "Validation picked", "vs released", "in band", "val spread",
         "Released ROC", "Selected ROC", "$\\Delta$ oracle", "$\\Delta$ val-sel."],
        rows, "llccccccc"))

print("wrote", len(made), "tables ->", OUT + "/{" + ",".join(made) + "}.tex")
