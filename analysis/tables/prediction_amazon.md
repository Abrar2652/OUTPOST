# Pre-registration: Amazon-Fraud (P23–P25)

Written **before any model is trained on Amazon**. The dataset was built on
2026-09-03 (`analysis/scripts/make_amazon.py`), the training-free diagnostic was
computed (`analysis/tables/spectral_amazon.csv`), and this file was committed
before the first run was launched. Nothing below is fitted to an Amazon result.

## Why Amazon, and why now

Two claims in this paper rest on a single real dataset, and n=1 is not a
comparison.

1. **Oracle inflation.** Photo, Computers and CS inflate by 0.045–0.08 between
   best-over-epochs and peak-validation selection; Yelp inflates by 0.0023 — a
   34× contrast that the paper uses to argue the small semi-synthetic benchmarks
   are what makes oracle selection pay. Every real datapoint in that argument is
   Yelp.
2. **Pseudo-labelling reverses sign.** PL is the only load-bearing component
   (Computers −0.0645, CS −0.0648) and it is *harmful* on Yelp (+0.0058,
   p=0.027). Whether that is a property of real anomaly data or a property of
   Yelp is currently unanswerable.

Amazon-Fraud is the cheapest way to make both n=2: 11,944 real reviewers, 6.87%
labelled fraudulent, standard in the papers we compare against, and small enough
that the dense PPR the full DEMO baseline needs is 570 MB rather than terabytes.

## The diagnostic, computed before training

| | median same-class frac | hop-0 AUC | smoothing gain |
|---|---|---|---|
| Amazon (class 1) | **0.080** | **0.878** | **−0.174** |

Amazon is a *feature-visible* graph: its anomalies are almost separable from raw
features alone, and propagation actively destroys that signal. It sits at the
31st percentile of same-class fraction among the 35 fitted classes, and it is the
regime P8 identified in ogbn-mag class 263 (same-frac 0.000, hop-0 0.854, gain
−0.187).

## The two versions of the law disagree, by a lot

The law fitted on all 35 classes is `best_auc = 0.5904 + 0.3084 × same_frac`,
residual sd 0.1167.

- **P23 (naive, one-regime law).** Amazon's best achievable AUC-ROC will be
  **0.615**, 95% band 0.382–0.848. This is what the law as *reported in §6*
  predicts, using only the same-class fraction.
- **P24 (two-regime law).** Amazon is feature-visible, so the same-class fraction
  does not govern it and the achievable AUC will be **near its hop-0 value,
  0.878** — far above P23's point estimate and at the very top of its band.

**These differ by 0.263 AUC, so Amazon discriminates between them.** This is the
first genuinely out-of-sample test the law has had: the 35 fitted classes were
all used to estimate ρ, and leave-one-dataset-out re-fits on the remainder.
Amazon is a 7th graph with a prediction registered in advance.

Outcomes and what each means:

| result | verdict |
|---|---|
| AUC ≈ 0.61 | P23 holds, P24 fails. The one-regime law predicts out of sample; the feature-visible exception is not needed. |
| AUC ≈ 0.88 | **P24 holds, P23 fails.** The law as written in §6 is under-specified and must be stated in its two-regime form, with the same-class fraction governing only the propagation regime. |
| AUC ≈ 0.70–0.80 | Both wrong. The law has no out-of-sample predictive validity at the class level and should be reported as a within-sample correlation only. |

We expect P24. That expectation is recorded here so that confirming it counts for
something and so that a 0.61 result cannot be re-described afterwards as what the
law meant all along.

## P25 — pseudo-labelling on a second real graph

**P25.** If PL's harm on Yelp is a property of *real* anomaly data rather than of
Yelp specifically, `C_nopl` on Amazon will be **positive** (removing PL helps),
as it is on Yelp (+0.0058). If PL's benefit tracks *semi-synthetic construction*
instead, `C_nopl` on Amazon will be **negative**, like Computers (−0.0645) and CS
(−0.0648).

This is the only prediction here we genuinely do not have a lean on, and it is
the one that decides whether §5's component story generalises or is an artefact
of how the small benchmarks were built.

## Configuration, fixed in advance

Amazon inherits Yelp's config block verbatim except `input_dim` (25 vs 32). It is
**not tuned**: transferring an untuned configuration to a new dataset is a
stronger claim than tuning one, and tuning here would forfeit it.

The single exception is `sim_topk_frac`, which has flipped sign between graphs
(+0.0263 on Yelp, −0.0586 on Photo) and cannot honestly be set by transfer. Both
arms are run and **validation AUC alone** selects between them, per the paper's
own selection protocol. The test set is not consulted.

---

## Results, 2026-09-04

Thirty runs (two OUTPOST arms × 10 seeds, full DEMO × 10 seeds), then ten
`C_nopl` runs under the validation-selected arm. Zero failed jobs.

### Arm selection, validation only

| arm | val AUC (n=10) |
|---|---|
| `A_sim0.0` (SimSample off) | **0.9692 ± 0.0140** |
| `A_sim1.0` (SimSample on) | 0.9510 ± 0.0189 |

Validation selects **SimSample off**. Yelp selects it on. So SimSample's
direction is not a property of real anomaly data; it still tracks nothing we can
name.

### P23 / P24 — the law's first out-of-sample test

| | predicted | actual |
|---|---|---|
| **P23** naive one-regime law | 0.615, band 0.382–0.848 | **0.9534 ± 0.0048** |
| **P24** two-regime, feature-visible | ~0.878 | 0.9534 |

**P23 is falsified.** The actual value is 0.105 above the ceiling of its 95%
band. The law as written in §6 — best achievable AUC as a linear function of
median same-class fraction — does not predict a 7th graph it was not fitted on.

**P24 is directionally confirmed and under-predicts by 0.075.** Amazon is
feature-visible (hop-0 AUC 0.878, smoothing gain −0.174) and its achievable AUC
is high, as the two-regime reading says. But the trained model beats raw
features by 0.075, so "feature-visible ⇒ near hop-0" is not the whole story
either; propagation still adds something even where the diagnostic says it
subtracts.

Consequence for the paper: §6 must state the law in two-regime form, with the
same-class fraction governing only the propagation regime, and must report this
out-of-sample result alongside the within-sample ρ. The within-sample ρ = 0.80
over 35 classes is real but it is not predictive validity, and this is the test
that shows the difference.

### P25 — pseudo-labelling on a second real graph

| `C_nopl` − full, Amazon, n=10 | Δ | W/L | p |
|---|---|---|---|
| oracle ROC | −0.0018 | 3/7 | 0.43 |
| oracle PR | −0.0057 | 2/8 | 0.11 |
| valsel ROC | +0.0003 | 4/6 | 0.63 |

**Null.** P25 offered two outcomes — positive like Yelp, or negative like
Computers/CS — and Amazon gave neither. Across the four datasets where PL has
been ablated:

| dataset | kind | PL removed |
|---|---|---|
| Computers | semi-synthetic | **−0.0645** (0/5) |
| CS | semi-synthetic | **−0.0648** (0/5) |
| Amazon | real | −0.0018 (3/7, n.s.) |
| Yelp | real | +0.0058 (9/10, p=0.027) |

The claim this supports is narrower than P25's positive branch but still sharp:
**the one component that carries OUTPOST's wins on semi-synthetic benchmarks
contributes nothing measurable on either real dataset.** That is a benchmark
finding, not a method finding, and it is the version of the PL story the paper
should make.

### The head-to-head

| Amazon, n=10 | OUTPOST | DEMO | Δ | W/L | p |
|---|---|---|---|---|---|
| oracle ROC | 0.9534 | 0.9555 | −0.0021 | 3/7 | 0.19 |
| oracle PR | 0.8391 | 0.8389 | +0.0002 | 5/5 | 0.92 |
| **valsel ROC** | 0.9420 | 0.9505 | **−0.0085** | **0/10** | **0.0020** |
| valsel PR | 0.8180 | 0.8296 | −0.0116 | 2/8 | 0.11 |

(The val-selected p was first reported here as 0.0077: `results.csv` rounds
to four decimals and one seed's two values collided, which the test read as
a tie. Recomputed from the per-rotation shards at full precision it is
0/0/10, p=0.0020, matching `stats.py`. Every paired test in the appendix now
reads the shards.)

A tie under oracle selection and a clean loss under deployable selection — the
mirror image of Photo, where oracle loses and val-selected wins. Two datasets
now change verdict between the two selection rules, in opposite directions.

### Oracle inflation, now seven graphs

Amazon inflates by **+0.0114**. The ordering is monotone: semi-synthetic
(0.045–0.080) > OGB (0.019–0.022) > real (0.002–0.011). The benchmark-validity
claim no longer rests on one dataset.
