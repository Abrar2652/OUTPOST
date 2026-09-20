# Pre-registration: the component decomposition on T-Finance (P38–P39)

Written 2026-09-07 22:45 UTC, **before `campaigns/tfinance_decomp.json` runs**.
No T-Finance gate or conformal arm exists on disk; only `C_nopl` has been run
there, and it is already scored (P28, `prediction_tfinance.md`).

## Why

The gate is measured null on six graphs — Photo −0.0002, Computers −0.0016,
CS +0.0001, Yelp +0.0005, Amazon −0.0002, ogbn-arxiv −0.0012, every p ≥ 0.19.
Six is enough to say "inert everywhere we looked", but the appendix's `gate_off`
map already reserves a T-Finance row and nothing fills it, so the honest options
are to run it or to delete the row. T-Finance is also the graph that broke the
"real graphs are scattered" reading (§6.1) and the one where pseudo-labelling is
significantly harmful, so it is the most informative place left to test whether
*any* component of OUTPOST does work on real data.

## Arms

Under the validation-selected SimSample arm (`A_sim0.0`, from
`arm_selection.json`), ten seeds each, 400 epochs, one rotation:
`C_nogate` (atlas gate removed) and `C_noconformal` (conformal step removed).
Twenty jobs, ~34 min per job.

## Predictions

**P38 (the gate is inert on a seventh graph).** `C_nogate` minus `A_sim0.0`,
oracle AUC-ROC, is within **±0.010** and its Wilcoxon p is above 0.05. The gate
has never moved a dataset by more than 0.0016 in either direction; a T-Finance
effect outside ±0.010 would be the first, and would mean the gate's inertness is
a property of the six graphs tested rather than of the component.

**P39 (conformal is the only component with a chance of surviving on real data).**
`C_noconformal` minus `A_sim0.0`, oracle AUC-ROC, is **negative** — between
−0.030 and 0.000. Grounds: conformal carried more than half of the pseudo-labelling
effect on Computers, and on Amazon it was the one component with a positive point
estimate for its removal (+0.0009, null). Predicting a negative value here is the
harder call, and if it lands null (within ±0.005) then *every* component of
OUTPOST is inert on every real graph and §5.2 must say exactly that.

## What would change the paper

If both land null, the honest headline for §5.2 becomes: on the three real
graphs, no component of the method is load-bearing — pseudo-labelling is null or
harmful, the gate is inert, conformal is inert — and the entire measured benefit
of OUTPOST over its ablations is a semi-synthetic-benchmark effect. That is a
stronger claim than the paper currently makes and it is registered here before
the runs so it cannot be assembled afterwards.

---

## Verdicts, scored 2026-09-13 (n=10, both arms complete)

Base is the validation-selected `A_sim0.0`; both arms ran ten seeds at 400 epochs.

| arm | Δ oracle AUC-ROC | seeds favouring removal | p | Δ val-selected |
|---|---|---|---|---|
| `C_nogate` | **−0.0000** | 4/10 | 0.9219 | +0.0034 |
| `C_noconformal` | **+0.0027** | 8/10 | 0.1309 | +0.0050 |

**P38 — CONFIRMED.** Predicted: the gate is inert within ±0.010 with p > 0.05.
Measured: −0.0000, p = 0.92. The atlas gate is now measured null on a **seventh**
graph, and the largest movement it has produced anywhere remains 0.0016.

**P39 — FALSIFIED.** Predicted: removing conformal would be *negative*, between
−0.030 and 0.000, on the grounds that conformal carried more than half the
pseudo-labelling effect on Computers. Measured **+0.0027** — removal helps
slightly, on 8 of 10 seeds, though not significantly (p = 0.13). The prediction
required a negative sign and did not get one.

### The consequence this file registered in advance

> *"If both land null, the honest headline for §5.2 becomes: on the three real
> graphs, no component of the method is load-bearing … and the entire measured
> benefit of OUTPOST over its ablations is a semi-synthetic-benchmark effect."*

+0.0027 is inside the ±0.005 band that file called null, and P38 is null outright.
So the registered consequence applies. The complete component picture on the three
real graphs, every arm at n=10 except Yelp's conformal (n=5):

| graph | gate | conformal | pseudo-labelling |
|---|---|---|---|
| Yelp | +0.0005 (p 0.63) | −0.0002 (p 0.18) | **+0.0059** (p 0.027) |
| Amazon | −0.0002 (p 0.32) | +0.0009 (p 1.00) | −0.0018 (p 0.43) |
| T-Finance | −0.0000 (p 0.92) | +0.0027 (p 0.13) | **+0.0097** (p 0.004) |

**No component of OUTPOST helps on any graph whose anomalies were observed rather
than injected.** Seven of the nine cells are null; the two that clear significance
are both *negative contributions* — removing pseudo-labelling improves Yelp and
T-Finance. Against this, on the semi-synthetic graphs pseudo-labelling is worth
0.038 to 0.065 and conformal 0.005 to 0.037.

This is the strongest and least comfortable form of §5.2's claim, and it was
written down as the outcome to expect before either arm ran.
