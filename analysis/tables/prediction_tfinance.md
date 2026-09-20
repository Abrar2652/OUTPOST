# Pre-registration: T-Finance (P26–P28)

Written **before any model trains on T-Finance**. The graph was converted on
2026-09-06 (`analysis/scripts/make_tfinance.py`: 39,357 accounts, 10 features,
1,804 = 4.58% labelled anomalous, 42.4M edges, mean degree 1,079), the
training-free diagnostic was computed (`analysis/tables/spectral_tfinance.csv`),
and the two-epoch smoke tests of OUTPOST and NSReg passed. Nothing below is
fitted to a T-Finance result.

## What the diagnostic says, and what it changes

| | median same-class frac | hop-0 AUC | best over hops | smoothing gain |
|---|---|---|---|---|
| T-Finance (class 1) | **0.629** | 0.8 | 0.8 | -0.006 |

Two things follow immediately, before any training.

**1. A real fraud graph that is homophilous.** §6.1 currently reads: the graphs
whose anomaly classes are scattered are Yelp's real fraud (0.16) and ogbn-mag
(0.00–0.14), while the semi-synthetic benchmarks are homophilous (0.55–1.00).
T-Finance is real, and its anomalies sit at 0.629 — inside the semi-synthetic
range. The "real = scattered" reading of §6.1 does not survive a third real
graph and must be restated as what the data actually supports: *the two real
graphs the field reports (Yelp, Amazon) are scattered; real fraud graphs are not
uniformly so.* This is recorded now so it cannot be discovered after the runs
and rationalised.

**2. This graph does not discriminate the one-regime from the two-regime law.**
Its smoothing gain is -0.006 — flat — so it sits on the boundary, and both
forms predict the same ceiling. The law fitted on the seven other graphs
(`best_auc = 0.6056 + 0.2910 × same_frac`, residual sd 0.1228) gives
**0.789** (band 0.543–1.000); the two-regime form, with the
gain at zero, gives the hop-0 value **0.800**. The measured diagnostic ceiling is
0.800. Amazon was the discriminating test; T-Finance is a *consistency* test,
and is registered as one.

## Predictions

## P26 — CONFIRMED at registration (consistency check)

**P26 (the law is consistent on an eighth graph).** The training-free ceiling
falls within the seven-graph law's band: it is 0.800 against a predicted
0.789, an error of 0.011, inside one residual sd. *Scored now from the
diagnostic alone; recorded as CONFIRMED at registration time, and stated as a
consistency check, not as evidence for the two-regime form over the one-regime
form.*

**P27 (the trained model exceeds the prototype ceiling on real data).** On the
two real graphs so far the trained OUTPOST beat the instrument's ceiling by
+0.114 (Yelp: 0.7448 vs 0.631) and +0.075 (Amazon: 0.9534 vs 0.878).
OUTPOST's oracle AUC-ROC on T-Finance will exceed 0.800 by **between +0.04 and
+0.13**, i.e. land in **[0.84, 0.93]**. Below 0.84: the "trained model
adds propagation the diagnostic cannot see" reading of §6.2 is Yelp/Amazon-
specific. Above 0.93: the ceiling is not a ceiling in any useful sense
and §6 should stop calling it one.

**P28 (pseudo-labelling is not load-bearing on a third real graph).** `C_nopl`
minus the validation-selected full arm, oracle AUC-ROC, will be **within ±0.010**
or positive (as Yelp +0.0058, Amazon −0.0018). If it is below −0.02 — PL
essential, as on Computers (−0.0645) and CS (−0.0648) — the claim that PL's
benefit tracks semi-synthetic construction (§5.2, §10.1-1) is falsified on
real data and must be withdrawn.

## Configuration, fixed in advance

Yelp's block verbatim except `input_dim = 10`; untuned, as Amazon. The SimSample
arm is chosen on **validation only** from `A_sim1.0` / `A_sim0.0`
(`analysis/tables/arm_selection.json`, written by `build_appendix.py`). `C_nopl`
runs under both arms so P28 can be scored under whichever is selected. DEMO is
the full published method with an APPNP PPR; NSReg runs from its released code.
Ten seeds each. Per-node scores are saved for every OUTPOST run.

---

## Verdicts, scored 2026-09-07 after all 60 T-Finance runs completed

All arms are n=10, 400 epochs, one rotation (T-Finance is a binary graph).
Validation selected `A_sim0.0` (mean val AUC 0.9370 against `A_sim1.0`'s 0.9322),
so every comparison below is against that arm, as registered.

| arm | n | val AUC | oracle AUC-ROC | val-selected AUC-ROC |
|---|---|---|---|---|
| `A_sim0.0` (selected) | 10 | 0.9370 | 0.9087 ± 0.0112 | 0.8946 ± 0.0125 |
| `A_sim1.0` | 10 | 0.9322 | 0.9215 ± 0.0039 | 0.9105 ± 0.0142 |
| `C_nopl_sim0.0` | 10 | 0.9439 | 0.9183 ± 0.0153 | 0.9085 ± 0.0159 |
| `C_nopl_sim1.0` | 10 | 0.9334 | 0.9224 ± 0.0041 | 0.9127 ± 0.0123 |
| DEMO (`B_demo_mix`) | 10 | — | 0.9161 ± 0.0081 | 0.9029 |
| NSReg (`E400_nsreg`) | 10 | — | 0.9238 ± 0.0093 | 0.9156 |

**P26 — CONFIRMED at registration** (unchanged; scored from the diagnostic alone).

**P27 — CONFIRMED.** Predicted: oracle AUC-ROC exceeds the 0.800 prototype
ceiling by +0.04 to +0.13, landing in [0.84, 0.93]. Measured on the selected arm:
**0.9087**, a gain of **+0.109** over the ceiling — inside the predicted band, and
inside it on the unselected arm too (`A_sim1.0` 0.9215, +0.122). The trained model
exceeds the training-free ceiling on a third real graph, so the §6.2 reading is not
Yelp/Amazon-specific.

**P28 — CONFIRMED, and in the strong direction.** Predicted: removing PL moves
oracle AUC-ROC by within ±0.010, or positively. Measured: `C_nopl_sim0.0` minus
`A_sim0.0` = **+0.0097** oracle (nine of ten seeds favour removal, Wilcoxon
p = 0.0039) and **+0.0139** val-selected (9/10, p = 0.0039). PL is not merely
inert on T-Finance, it is *significantly harmful*. Under the unselected arm the
same difference is +0.0009, so the direction does not depend on the arm.

**What this settles.** Pseudo-labelling — the only load-bearing component on the
semi-synthetic benchmarks (Computers −0.065, CS −0.065, Photo −0.038) — is now
measured on three real graphs and helps on none: Yelp +0.0058, Amazon −0.0018,
T-Finance +0.0097. On the one where the effect is statistically significant it is
significant *against* the component. The claim that PL's benefit tracks
semi-synthetic benchmark construction (§5.2, §10.1-1) survives its third test and
is strengthened, not withdrawn.

**A note against the paper's own method.** T-Finance is the only graph where
OUTPOST is beaten by both baselines under both selection rules: NSReg 0.9238 and
DEMO 0.9161 against OUTPOST's selected 0.9087. OUTPOST's *unselected* arm (0.9215)
would have beaten DEMO — validation picked the worse of its two arms, costing
0.0128 oracle AUC-ROC. That is the same failure mode Amazon shows, on a second
real graph, and it is reported as such rather than by quietly reporting the better arm.
