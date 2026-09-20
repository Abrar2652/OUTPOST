# Pre-registered predictions for the OGB graphs

Written 2026-08-07, **before** OUTPOST was trained on ogbn-arxiv or ogbn-mag,
from the training-free diagnostic alone (`analysis/tables/spectral_ogbn-*.csv`,
produced by `spectral_diagnostic.py`, which never trains a model). The runs that
test these are tagged `A_main` in `results/results.csv`.

The point of writing them down first: the detectability law is only worth
anything if it forbids outcomes. Below are the outcomes it forbids.

## What the diagnostic says, before any training

| graph | anomaly classes | median same-class neighbour fraction | prototype best AUC |
|---|---|---|---|
| ogbn-arxiv | 4 | 0.46 – 0.82 | 0.62 – 0.73 |
| ogbn-mag | 15 | **0.00 – 0.14** (11 of 15 are exactly 0.000) | 0.49 – 0.87 |

An ogbn-mag anomaly class is a venue holding under 0.03% of papers — 129 to 199
nodes in a graph of 736,389. The median such node has **no same-class
neighbour at all**. By the law of METHODOLOGY section 6 these classes sit in the
scattered regime, where low-pass propagation averages an anomaly into normality
rather than concentrating it.

## Predictions

**P5 (ogbn-mag is at chance, and not because of our method).** OUTPOST's
ogbn-mag AUC-ROC will fall below 0.60, i.e. it will not meaningfully beat the
published field (ConsisGAD 0.4909, SpaceGNN 0.4626, NSReg 0.4836, DEMO 0.4967 —
all at or below chance). AUC-PR will stay near the 0.004–0.006 band. *A large
OUTPOST gain on ogbn-mag would falsify the law*, because the law says the signal
is not in the graph to be found.

**P6 (ogbn-arxiv lands in between).** arxiv's classes are homophilous
(0.46–0.82) but not feature-visible (hop-0 AUC 0.49–0.61), which is the regime
where propagation is the whole lever. OUTPOST will therefore be competitive with
the published field on arxiv (DEMO 0.6364 ROC / 0.3329 PR) rather than at
chance, and will not exceed it by more than seed noise: nothing in OUTPOST's
components addresses a regime that already works.

**P7 (the ordering is predicted, not fitted).** Across the six graphs, the rank
order of OUTPOST's achieved AUC-ROC will track the rank order of the diagnostic's
median same-class fraction, computed without training and without labels at test
time. Concretely: cs > photo ≈ computers > arxiv > yelp ≈ mag.

**P8 (mag's one exception stays exceptional).** ogbn-mag class 263 has
same-class fraction 0.000 yet hop-0 AUC 0.854 and a *negative* smoothing gain
(−0.187): it is feature-visible, the other regime of the two-regime law. It
should remain the best-detected mag class under a trained model too, and
propagation should not be what detects it.

---

# Outcomes (recorded 2026-08-11, after the runs)

## P6 — CONFIRMED

ogbn-arxiv, OUTPOST, 5 seeds: **0.6237 ± 0.0083 AUC-ROC, 0.3053 ± 0.0067 AUC-PR**
(val-selected 0.6048 / 0.2965).

| | AUC-ROC | AUC-PR |
|---|---|---|
| DEMO | 0.6364 | 0.3329 |
| ConsisGAD | 0.6216 | 0.3148 |
| NSReg | 0.6182 | 0.3230 |
| SpaceGNN | 0.6133 | 0.3301 |
| **OUTPOST** | **0.6237** | **0.3053** |

P6 said arxiv would be *competitive rather than at chance*, and would *not exceed
the field*, because nothing in OUTPOST addresses a regime where propagation
already works. Both halves hold: 0.6237 is nowhere near chance, sits inside the
published band on ROC (above three of four), and does not beat it. This is the
prediction being right in the boring direction, which is the direction that
counts — it was written down when it could still have come out otherwise.

## P7 — PARTIALLY FALSIFIED

Predicted ordering: `cs > photo ≈ computers > arxiv > yelp ≈ mag`.
Observed (OUTPOST AUC-ROC): `cs 0.9832 > photo 0.8310 ≈ computers 0.8277 >
yelp 0.7448 > arxiv 0.6237`.

**Yelp and arxiv are inverted.** Yelp has a same-class fraction of 0.16 against
arxiv's 0.46–0.82, so the law's topology-only ordering puts arxiv comfortably
above it; the measurement puts Yelp 0.12 above arxiv. P7 as written is wrong on
that pair.

A post-hoc explanation is available and is labelled as post hoc: P7 conflated
*detectability under the training-free prototype detector* with *achieved AUC
under a method carrying a dataset-conditional component*. SimSample is active on
Yelp (median degree 168, above the sampling budget of 25) and inactive on arxiv,
so the intervention lifts Yelp above where topology alone places it. That is
consistent with the paper's thesis, but it was not what P7 said, and the
prediction should be recorded as failed rather than reinterpreted. The correct
statement of the law's ordering claim applies to the *training-free* instrument,
where it still holds — not to a tuned method's achieved scores.

## P5 — SPLIT: the threshold held, the claim it encoded did not
## (the section below is the pre-result record; the verdict follows it)

Measured on an RTX A5000: ogbn-mag peaks at **11.8 GB** and runs at **77 s per
post-warmup epoch**, so one rotation of the 400-epoch protocol is **8.6 hours**
and one seed over its 15 rotations is **128 GPU-hours**. Raising `batch_size`
from 256 to 1024 — numerically inert here, since the loop takes one full-batch
step per epoch — brings that to 44 s/epoch and 73 GPU-hours per seed. Still
multiple days per seed on one card, against 45 rotation-jobs for three seeds.

ogbn-mag is therefore **not run**, and P5 and P8 remain untested. That is stated
rather than quietly dropped: P5 is the prediction that would most cheaply
falsify the law, so its absence is a real gap in the evidence, not a formality.

The temptation worth naming: mag could be run at a reduced epoch budget for a
fraction of the cost. It should not be. P5 predicts OUTPOST stays *below* 0.60,
and fewer epochs under best-over-epochs selection can only push the score down —
a shortened run would make our own prediction easier to confirm. A biased test of
a pre-registered prediction is worth less than no test.

What survives without it: the training-free diagnostic on ogbn-mag is already
computed (11 of 15 classes have a median same-class fraction of exactly 0.000)
and contributes its 15 classes to the n=35 regression in §6. The published field
is at chance there. Only OUTPOST's own point on that graph is missing.

### Result, 2026-09-01: five seeds, full 400-epoch budget

The budget was not reduced, for the reason given above. Five seeds × 15
rotations, 82 GPU-hours per seed, zero failed jobs.

| seed | 0 | 1 | 2 | 3 | 42 | mean |
|---|---|---|---|---|---|---|
| AUC-ROC | 0.5941 | 0.5868 | 0.5945 | 0.5947 | 0.5913 | **0.5923 ± 0.0034** |
| AUC-PR | 0.0101 | 0.0102 | 0.0099 | 0.0102 | 0.0102 | **0.0101 ± 0.0001** |

| method | ogbn-mag AUC-ROC | our margin |
|---|---|---|
| **OUTPOST** | **0.5923** | — |
| DEMO | 0.4967 | +0.0956 |
| ConsisGAD | 0.4909 | +0.1014 |
| NSReg | 0.4836 | +0.1087 |
| SpaceGNN | 0.4626 | +0.1297 |

**P5 has to be scored as split, and the half that failed is the half that
mattered.**

- *"AUC-ROC will fall below 0.60"* — **held.** 0.5923, and the highest single
  seed is 0.5947.
- *"i.e. it will not meaningfully beat the published field"* — **failed.** We beat
  the best published method by +0.0956 with 5/5 seeds above it, and the spread
  (sd 0.0034) is an order of magnitude smaller than the margin.
- *"AUC-PR will stay near the 0.004–0.006 band"* — **failed.** 0.0101, which is
  1.87× the best published AUC-PR.

The prediction treated "below 0.60" and "will not meaningfully beat the field"
as the same statement. They are not, because the field is *below chance*: a score
can be poor in absolute terms and still be far above every published
alternative. Writing the threshold rather than the comparison was an error in
operationalisation, and it is worth recording as such rather than claiming the
confirmed half.

### This is the condition P5 itself named as falsifying the law

P5 says, in its own text: *"A large OUTPOST gain on ogbn-mag would falsify the
law, because the law says the signal is not in the graph to be found."* A
+0.0956 margin over the whole published field, unanimous across seeds, is that
gain. The law's own stated falsification condition has been met, and the paper
cannot both headline the mag result and leave §6's law unqualified.

Two readings are available and the data does not yet separate them:

1. **The law is about absolute detectability and remains intact.** 0.5923 is a
   poor score; 11 of 15 mag classes have a median same-class fraction of exactly
   0.000; the law predicted mag would be near the bottom and it is. What the law
   never claimed is that all methods fail *equally* in that regime.
2. **The law is refuted as stated.** It says the signal is not there to be found;
   we found roughly 0.09 AUC of it that four published methods did not.

Reading 1 requires rewriting the law's claim to be about the achievable ceiling
rather than about any method's ability to approach it — a real weakening. The
`valsel` number matters here: under deployable selection mag is 0.5708, still
above the entire field's oracle-selected published numbers, so this is not a
selection artefact.

**Recommended framing: report mag as a regime result, not a leaderboard result —
every published method sits at chance on the largest graph and OUTPOST is
measurably above it, while remaining poor in absolute terms. Over-claiming mag as
"solving" the graph would be both wrong and the fastest available route to
rejection.** P8 remains testable from the same shards and has not been scored.

## What would falsify the law

- OUTPOST reaching AUC-ROC > 0.60 on ogbn-mag through propagation-based
  components (P5). **Not met on the threshold (0.5923), met on the comparison (+0.0956 over the best published method, 5/5 seeds). See the
  P5 verdict above.**
- The six-graph ordering of P7 failing in a way not attributable to seed
  variance.
- Class 263 not being among the best-detected mag classes (P8).

## Recorded caveat on an earlier claim

The n=16 version of this analysis stated that real fraud's same-class fraction
(0.160) is "below every semi-synthetic class (0.55–1.00)". With ogbn-mag in the
sample that sentence is **false**: 14 of 34 semi-synthetic classes now sit below
0.160. The claim it was standing in for survives in a sharper form and is
restated in METHODOLOGY section 6 — the graphs the field ranks methods on
(Photo, Computers, CS) are the homophilous ones, and both graphs whose classes
are scattered (Yelp's real fraud, ogbn-mag's rare venues) are where every
published method collapses to chance. ogbn-mag was not part of the sample when
the law was fitted, so it is an out-of-sample confirmation of the law's negative
prediction rather than a datapoint that produced it.


## P8 — NOT SCORABLE from the recorded shards (2026-09-06)

P8 predicts that ogbn-mag class 263 stays the best-detected class under a trained
model. Scoring it needs per-class test scores; the rotation shards store per-rotation
AUCs over all anomalies and over the unseen set, and `write_rotation` strips the
per-node score arrays to keep shards small. The five mag seeds would have to be
re-run with score dumps (≈410 GPU-hours) to score it. Recorded as untested, not
as confirmed.

---

## P8 — FALSIFIED (scored 2026-09-18, the last prediction to close)

P8 said ogbn-mag class 263 — same-class fraction 0.000, hop-0 AUC 0.854, smoothing gain −0.187 — "should remain the best-detected mag class under a trained model too". It was unscorable for five weeks because per-class detection needs per-node scores, which the original mag runs did not save. `campaigns/mag_scores.json` re-ran all 15 rotations at five seeds with `--save-scores`; 75 score files now exist at both the oracle and the deployable epoch.

Each anomaly class is scored against the ~700k normal nodes using every rotation's saved scores, excluding that class's own rotation (being the seen class is a different task from being detected unseen), then averaged over 70 rotation-seeds.

| rank | class | mean AUC-ROC | sd |
|---|---|---|---|
| 1 | 216 | 0.6765 | 0.1432 |
| 2 | 282 | 0.6641 | 0.1110 |
| 3 | 206 | 0.6566 | 0.1423 |
| 4 | 195 | 0.6489 | 0.1218 |
| 5 | 231 | 0.6157 | 0.1891 |
| **6** | **263** | **0.6157** | **0.0761** |
| 9 | 151 | 0.6039 | 0.0653 |
| 15 | 2 | 0.4135 | 0.1421 |

**Class 263 ranks sixth of fifteen, not first.** Its training-free advantage — the highest hop-0 AUC of any mag class at 0.854 — does not survive training. The trained model brings it to 0.6157, indistinguishable from class 231 at the same value and below four others.

**What this costs the two-regime law.** §6.2's feature-visible regime says such classes are detectable from raw features and that propagation cannot help them. The diagnostic half of that holds. The claim that the advantage *persists under a trained model* does not: whatever the trained model learns, it does not preserve the ordering the diagnostic predicts. This is the second out-of-sample failure of the law's predictive content, after P23 on Amazon and the cross-validation in §6.5, and it points the same way — the law describes the classes measured, and does not forecast.

**One property of 263 does survive.** Its standard deviation across rotation-seeds is **0.0761**, the second-lowest of the fifteen against a median near 0.15. It is the most *stable* class to detect even though it is not the best. That is consistent with feature-visibility — it does not depend on which class the model happened to see — and it is the part of P8 worth keeping.

**A scoring bug caught before reporting.** The first run used label 0 as the normal pool. ogbn-mag has 349 paper classes and 0 is simply one of them, so the comparison ran against 2,174 nodes instead of ~700k and ranked 263 eighth. The verdict was the same either way, but the numbers were not, and a falsification quoted from the wrong denominator is not evidence.

---

## P8 — FALSIFIED

*Scored 2026-09-18; the last prediction to become scorable.*

P8 needed per-node scores, which only the `--save-scores` re-run of seeds 4–8 produced. All 75 rotations are now saved, and each anomaly class is ranked by how well the trained model separates it from normals, averaged over rotations (`analysis/scripts/score_p8.py`).

**Class 263 ranks 6th of 15**, not first. Its trained AUC is 0.6363 against class 216's 0.6946.

Worse for the prediction's reasoning: 263's training-free hop-0 AUC is 0.854, so the trained model scores it **0.218 lower than the diagnostic did**. The class was singled out because the diagnostic found it easy without propagation; training does not find it easy at all.

### What this exposes about the law

These 15 classes are the largest group in the 37 the detectability law is fitted on, and they are the only ones where per-class *trained* scores now exist. That allows the test the law has never had — does the diagnostic predict what a trained model achieves?

| | Spearman ρ | p |
|---|---|---|
| same-class fraction vs trained AUC | **-0.160** | 0.568 |
| diagnostic ceiling vs trained AUC | **+0.161** | 0.567 |
| training-free detectability (hop-0 AUC) vs trained AUC | **-0.021** | 0.940 |

No relationship exists in any of the three. The same-class fraction, the law's predictor, is if anything *negatively* related to trained performance here, and hop-0 detectability — the quantity "feature-visible" is defined by — ranks the classes independently of what training achieves. The three classes with the highest diagnostic ceilings are among the worst over-predicted: 176 (0.811 → 0.475, −0.336), 143 (0.874 → 0.613, −0.262), 263 (0.854 → 0.616, −0.238).

*(Corrected 2026-09-19: the first two rows read −0.238 / +0.111 and the over-predictions −0.323 / −0.245 / −0.218 before the normal-pool fix, which had scored anomaly classes against label 0's 2,174 nodes rather than all ~700k non-anomaly nodes. All three rows now come from `analysis/scripts/score_p8.py` into `p8_per_class.json`. Both nulls survive; only the magnitudes moved.)*

This is the same verdict §6.5 reached by cross-validation, arrived at independently and at the level of individual classes: **the law describes the classes it was measured on and does not predict trained behaviour.** The pooled ρ = 0.778 is a diagnostic-versus-diagnostic correlation — `best_auc` is itself a max over propagation hops — and it does not transfer to what training achieves.

The honest reading of P8 is not that one class behaved oddly. It is that "feature-visible" was a property of the instrument, not of the data.
