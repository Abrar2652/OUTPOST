#!/usr/bin/env python3
"""Assert every rendered appendix table has as many rows as its source data.

Written after a variable collision in render_appendix.py silently replaced the
37-row per-class detectability table with 6 rows of cross-validation results. The
page rendered, every section was present, and both existing checks passed at 261
cells and 1018 claims - because both compare numbers that ARE there against the
appendix data. Neither can see data that quietly vanished.

The only signal was the HTML getting smaller after a section was added.

    python analysis/scripts/check_appendix_complete.py
"""
import json, re, sys

A = json.load(open("analysis/appendix.json"))
HTML = open("analysis/appendix.html", encoding="utf-8").read()

# (section id, expected row count, label). A source that is a list contributes
# len(); anything else states its own count.
def n_law_classes():
    return len(A.get("detectability_law", {}).get("classes", []))


def n_law_lodo():
    return len(A.get("detectability_law", {}).get("leave_one_dataset_out", []))


EXPECT = [
    ("main",       lambda: len(A.get("main_table", [])),      "main table rows"),
    ("paired",     lambda: len(A.get("paired_vs_demo", [])),  "OUTPOST vs DEMO rows"),
    ("ablations",  lambda: len(A.get("ablations", [])),       "ablation arms"),
    ("law",        n_law_classes,                             "per-class law rows"),
    ("law",        n_law_lodo,                                "leave-one-dataset-out rows"),
    ("noise",      lambda: len(A.get("selection_noise", {}).get("datasets", [])),
                                                              "selection-noise rows"),
    ("params",     lambda: len(A.get("parameter_counts", [])), "module param rows"),
    ("params",     lambda: len(A.get("parameter_counts_all", [])), "trained param rows"),
    ("nsreg",      lambda: len(A.get("paired_vs_nsreg", [])), "OUTPOST vs NSReg rows"),
]


def section(sid):
    i = HTML.find(f'id="{sid}"')
    if i < 0:
        return None
    j = HTML.find("<section", i + 10)
    return HTML[i:j if j > 0 else len(HTML)]


def row_counts(seg):
    out = []
    for tbl in re.findall(r"<table>.*?</table>", seg, re.S):
        body = re.search(r"<tbody>(.*?)</tbody>", tbl, re.S)
        out.append(len(re.findall(r"<tr", body.group(1))) if body else 0)
    return out


def main():
    problems, checked = [], 0
    for sid, fn, label in EXPECT:
        want = fn()
        if not want:
            continue
        seg = section(sid)
        if seg is None:
            problems.append(f"section '{sid}' missing from appendix.html ({label})")
            continue
        counts = row_counts(seg)
        checked += 1
        # a table in this section must match the source exactly; rows are never
        # aggregated or filtered at render time, so any shortfall is data loss
        if want not in counts:
            problems.append(
                f"{label}: appendix.json has {want} but section '{sid}' renders "
                f"tables of {counts} rows - no table matches")
    print(f"appendix completeness: {checked} table(s) checked against their source, "
          f"{len(problems)} problem(s)")
    for p in problems:
        print("  PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
