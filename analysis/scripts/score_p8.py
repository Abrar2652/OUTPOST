#!/usr/bin/env python3
"""Score P8: does ogbn-mag class 263 stay the best-detected class under a trained model?

P8 (prediction_ogb.md): "class 263 has same-class fraction 0.000 yet hop-0 AUC 0.854
and a negative smoothing gain (-0.187): it is feature-visible... It should remain the
best-detected mag class under a trained model too, and propagation should not be what
detects it."

Unscorable until now because it needs PER-CLASS detection from a trained model, which
needs per-node scores. campaigns/mag_scores.json re-ran all 15 rotations x 5 seeds with
--save-scores, so the scores exist at both the oracle and the deployable epoch.

For each rotation the model saw one anomaly class as "seen"; every anomaly class is
scored against the normals using that rotation's node scores, then averaged over
rotations and seeds. A class's own rotation is excluded from its average - being the
seen class is a different task from being detected unseen.

    python analysis/scripts/score_p8.py -> analysis/tables/p8_per_class.json
"""
import glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ".")


def main():
    from sklearn.metrics import roc_auc_score
    from utils import load_data
    _, labels, _ = load_data("ogbn-mag")
    y = np.asarray(labels)

    files = sorted(glob.glob("results/scores/ogbn-mag_outpost_s*_A_main_rot*_best.npy"))
    if not files:
        print("no saved mag scores"); return 1
    classes = sorted({int(f.rsplit("_rot", 1)[1].split("_")[0]) for f in files})
    # "Normal" is every node that is not in ANY anomaly class - ogbn-mag has 349
    # paper classes and label 0 is simply one of them, so treating class 0 as the
    # normal pool used 2,174 nodes instead of ~700k and scored a different task.
    normals = np.where(~np.isin(y, classes))[0]
    print(f"  {len(files)} score files, {len(classes)} anomaly classes, "
          f"{len(normals):,} normal nodes")

    per = {c: [] for c in classes}
    for f in files:
        rot = int(f.rsplit("_rot", 1)[1].split("_")[0])
        s = np.load(f).astype(np.float32)
        for c in classes:
            if c == rot:
                continue                      # its own rotation: seen, not unseen
            idx = np.where(y == c)[0]
            if not len(idx):
                continue
            yy = np.concatenate([np.ones(len(idx)), np.zeros(len(normals))])
            ss = np.concatenate([s[idx], s[normals]])
            per[c].append(float(roc_auc_score(yy, ss)))

    rows = [{"class": c, "n_rotations": len(v),
             "mean_auc": round(float(np.mean(v)), 4),
             "sd": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else None}
            for c, v in per.items() if v]
    rows.sort(key=lambda r: -r["mean_auc"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    # Does the training-free diagnostic predict trained detection at all? P8 is a
    # claim about one class; these are the same claim across all 15. METHODOLOGY 6.6
    # quoted two of these from an ad-hoc session with no generating script, so they
    # are computed here, from the corrected normal pool, and land in the artifact.
    corr = None
    dpath = "analysis/tables/spectral_ogbn-mag.csv"
    if os.path.exists(dpath):
        import csv
        from scipy.stats import spearmanr
        diag = {}
        for r in csv.DictReader(open(dpath)):
            hops = [float(r[k]) for k in ("auc_hop0", "auc_hop1", "auc_hop2", "auc_hop3")]
            diag[int(r["class"])] = {"auc_hop0": hops[0], "ceiling": max(hops),
                                     "med_same_frac": float(r["med_same_frac"])}
        pair = [(diag[r["class"]], r) for r in rows if r["class"] in diag]
        if len(pair) >= 3:
            y = [r["mean_auc"] for _, r in pair]
            corr = {"n_classes": len(pair),
                    "y": "trained mean AUC on unseen rotations", "against": {}}
            for key in ("med_same_frac", "ceiling", "auc_hop0"):
                rho, pv = spearmanr([d[key] for d, _ in pair], y)
                corr["against"][key] = {"spearman_rho": round(float(rho), 4),
                                        "p_value": round(float(pv), 4)}
            # the figure and 6.6's "over-predicts worst" sentence both read these
            corr["spearman_rho"] = corr["against"]["auc_hop0"]["spearman_rho"]
            corr["p_value"] = corr["against"]["auc_hop0"]["p_value"]
            corr["x"] = "auc_hop0"
            over = sorted(((d["ceiling"] - r["mean_auc"], r["class"]) for d, r in pair),
                          reverse=True)[:3]
            corr["largest_overprediction"] = [
                {"class": c, "ceiling_minus_trained": round(float(g), 4)} for g, c in over]

    top = rows[0]
    c263 = next((r for r in rows if r["class"] == 263), None)
    out = {"prediction": "P8: mag class 263 stays the best-detected class under a "
                         "trained model",
           "per_class": rows,
           "best_class": top["class"], "best_auc": top["mean_auc"],
           "class_263": c263, "diagnostic_vs_trained": corr,
           "confirmed": bool(c263 and c263["rank"] == 1)}
    json.dump(out, open("analysis/tables/p8_per_class.json", "w"), indent=1)

    print(f"\n  {'rank':>4s} {'class':>6s} {'mean AUC':>9s} {'sd':>7s} {'n':>4s}")
    for r in rows:
        mark = "  <-- P8's class" if r["class"] == 263 else ""
        print(f"  {r['rank']:4d} {r['class']:6d} {r['mean_auc']:9.4f} "
              f"{r['sd'] if r['sd'] is not None else float('nan'):7.4f} "
              f"{r['n_rotations']:4d}{mark}")
    if corr:
        print(f"\n  vs trained AUC, {corr['n_classes']} classes:")
        for k, v in corr["against"].items():
            print(f"    {k:>14s}  rho = {v['spearman_rho']:+.3f}  p = {v['p_value']:.3f}")
        print("    largest over-prediction (ceiling - trained): " + ", ".join(
            f"class {o['class']} by {o['ceiling_minus_trained']:.3f}"
            for o in corr["largest_overprediction"]))
    print(f"\n  VERDICT: {'CONFIRMED' if out['confirmed'] else 'FALSIFIED'} - "
          f"class 263 ranks {c263['rank'] if c263 else '?'} of {len(rows)}")
    print("\n-> analysis/tables/p8_per_class.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
