"""Pre-registration ledger: every P-numbered prediction, where it was registered,
what it said in one line, and the verdict recorded in that file.

Verdicts are READ from the prediction files' own headings/lines (not retyped),
so this ledger cannot disagree with them. One-line statements are curated here
because they are prose, not numbers.

    python analysis/scripts/prereg_ledger.py   # -> analysis/tables/prereg_ledger.md
"""
import glob, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

ONE_LINE = {
 1: "Between-class structural separation: anomaly classes differ in same-class fraction / feature dissimilarity / Dirichlet energy",
 2: "Within-class: same-class fraction correlates with detection score at the best hop",
 3: "Propagation depth: gain from hop 0 to hop 3 is positive for clustered classes, negative for scattered ones",
 4: "Benchmark validity: real fraud (Yelp) has lower same-class fraction than every semi-synthetic class",
 5: "ogbn-mag: OUTPOST stays below 0.60 AUC-ROC and does not meaningfully beat the published field",
 6: "ogbn-arxiv: OUTPOST competitive with the published field, not above it by more than seed noise",
 7: "Achieved AUC-ROC across six graphs ranks in the order of the training-free same-class fraction",
 8: "ogbn-mag class 263 (feature-visible) remains the best-detected class under a trained model",
 9: "Validation-only hyperparameter selection on Photo cannot discriminate arms (tie band)",
 10: "Label budget: OUTPOST's advantage persists across the anomaly-count sweep",
 11: "SimSample harm on Photo is explained by removal of edge subsampling (mechanism)",
 12: "The SimSample-off arm does not vary with budget (control)",
 13: "Photo/Computers harm vs Yelp gain tracks the above-budget node fraction",
 14: "CS shows the largest SimSample harm of any dataset (89% of nodes above budget)",
 15: "Validation-only tuning retains the default on every dataset (no discrimination)",
 16: "Best-on-test minus validation-selected arm is non-trivial on at least one dataset",
 17: "Within Photo, Delta(b) is non-monotone: >=0 at b=5,10; most negative at b=25; back toward 0 at b=100",
 18: "|Delta| at b=100 < |Delta| at b=25 (interior maximum)",
 19: "The SimSample-off arm does not move with b",
 20: "Gate+PL removed together costs more than the sum of each alone (redundant components)",
 21: "PL-off costs more than gate-off on Computers and CS (conformal nested inside PL)",
 22: "Conformal-off alone stays near zero on Computers and CS",
 23: "Amazon: one-regime law predicts best AUC 0.615 (band 0.38-0.85)",
 24: "Amazon: feature-visible, achievable AUC near hop-0 (0.878)",
 25: "Removing PL on Amazon is positive (real-data property) or negative (semi-synthetic property)",
 26: "T-Finance: the seven-graph law's ceiling prediction (0.789) is within one residual sd of the measured 0.800 (consistency, not discrimination)",
 27: "T-Finance: trained OUTPOST exceeds the prototype ceiling by +0.04 to +0.13 (lands in [0.84, 0.93])",
 28: "T-Finance: removing PL is within +/-0.010 or positive (PL not load-bearing on a third real graph)",
 29: "400-epoch sweep: validation retains the default on Photo and Computers (all arms in the tie band)",
 30: "400-epoch sweep: test-selection bonus at 400 is at least the 200-epoch value (Photo >= +0.015)",
 31: "Amazon sweep: default retained and test-selection bonus below 0.005 (above 0.010 falsifies)",
 32: "Amazon at the validation-selected config (T_hidden16) is worse on test than the default at n=10, by 0.003-0.012",
 33: "Amazon: the OUTPOST-vs-DEMO oracle deficit widens to between -0.006 and -0.020 under honest selection",
 34: "Amazon: removing PL at hidden_dim 16 stays within +/-0.010 (PL nullity is a property of the graph, not the width)",
 35: "NSReg's validation-selected arm keeps its advantage at n=10 (Photo >= +0.005, Amazon >= +0.002)",
 36: "BLIND: DEMO's matched grid at 400 epochs retains its published config on all four datasets; test-selection bonus below +0.02",
 37: "BLIND: one validation rule does not equalise the methods - Amazon deltas differ in sign across methods",
 38: "T-Finance: the atlas gate is inert on a seventh graph (within +/-0.010, p > 0.05)",
 39: "T-Finance: removing conformal is negative (-0.030 to 0.000); null would mean NO component works on real data",
 40: "Amazon's validation preference for T_hidden16 survives at ten seeds (robustness note, not a re-selection)",
}
FILES = {**{i: "analysis/PHASE0_FINDINGS.md" for i in (1, 2, 3, 4)},
         **{i: "analysis/tables/prediction_ogb.md" for i in (5, 6, 7, 8)},
         9: "analysis/tables/prediction_tuning.md",
         **{i: "analysis/tables/prediction_budget.md" for i in (10, 11, 12, 13, 14)},
         15: "analysis/tables/prediction_hparam_all.md", 16: "analysis/tables/prediction_hparam_all.md",
         **{i: "analysis/tables/prediction_curve.md" for i in (17, 18, 19)},
         **{i: "analysis/tables/prediction_mechanism.md" for i in (20, 21, 22)},
         **{i: "analysis/tables/prediction_amazon.md" for i in (23, 24, 25)},
         **{i: "analysis/tables/prediction_tfinance.md" for i in (26, 27, 28)},
         **{i: "analysis/tables/prediction_hparam400.md" for i in (29, 30, 31)},
         **{i: "analysis/tables/prediction_selection_final.md"
            for i in (32, 33, 34, 35, 36, 37)},
         **{i: "analysis/tables/prediction_tfinance_decomp.md" for i in (38, 39)},
         40: "analysis/tables/prediction_selection_final.md"}
# Stop the qualifier at a full stop or a bold marker: without that the capture
# runs on into the sentence that follows the verdict word and the ledger cell
# reads "CONFIRMED.** Predicted: oracle AUC-ROC exceeds th".
VERDICT_RE = re.compile(r"(CONFIRMED[^\n|.*]{0,40}|FALSIFIED[^\n|.*]{0,40}|SPLIT[^\n|.*]{0,60}|PARTIALLY FALSIFIED|NOT SCORABLE[^\n|.*]{0,40}|NOT TESTED[^\n|.*]{0,40}|directionally confirmed[^\n|.*]{0,40}|is falsified|Null\.)")


def verdict_for(i, path):
    if not os.path.exists(path):
        return "file missing"
    txt = open(path, encoding="utf-8").read()
    # a heading or bold line that names this P and a verdict word
    # LAST match wins, not the first. These files are append-only records: a
    # verdict written today sits below the one it corrects, and taking the first
    # match made the ledger report "CONFIRMED" for a P36 that had been split
    # hours earlier.
    hits = []
    for m in re.finditer(rf"(^|\n)(#+ |\*\*)?P{i}\b[^\n]*", txt):
        line = m.group(0)
        v = VERDICT_RE.search(line)
        if v:
            hits.append(v.group(1).strip())
    if hits:
        return hits[-1]
    for m in re.finditer(rf"(^|\n)(#+ |\*\*)?P{i}\b[^\n]*", txt):
        line = m.group(0)
        v = VERDICT_RE.search(line)
        if v:
            return v.group(0).strip(" .*")
    # amazon-style prose: "P23 is falsified", "P24 is directionally confirmed", P25 "**Null.**"
    for m in re.finditer(rf"P{i}\b[^\n]{{0,80}}", txt):
        v = VERDICT_RE.search(m.group(0))
        if v:
            return v.group(0).strip(" .*")
    if i == 25 and "**Null.**" in txt:
        return "NULL (neither branch)"
    if i in (1, 2, 3):
        return "Phase-0 diagnostic; see PHASE0_FINDINGS.md (superseded by the 7-graph law, §6.3)"
    if i == 4:
        return "CORRECTED — false with ogbn-mag in the sample (§6.1); restated"
    return "no verdict recorded"


rows = ["# Pre-registration ledger\n",
        "Generated by `analysis/scripts/prereg_ledger.py`; verdicts are read from each prediction file. "
        "A prediction with no recorded verdict is an open item.\n",
        "| P | registered in | prediction (one line) | verdict |", "|---|---|---|---|"]
open_items = []
for i in range(1, max(ONE_LINE) + 1):
    f = FILES[i]; v = verdict_for(i, f)
    if v.startswith("no verdict"):
        open_items.append(i)
    rows.append(f"| P{i} | `{f}` | {ONE_LINE[i]} | {v} |")
rows.append(f"\nOpen (no verdict recorded): {', '.join(f'P{i}' for i in open_items) if open_items else 'none'}.")
open("analysis/tables/prereg_ledger.md", "w", encoding="utf-8").write("\n".join(rows) + "\n")
print("wrote analysis/tables/prereg_ledger.md; open items:", open_items or "none")
