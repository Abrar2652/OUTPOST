# Yelp results & ablations (OUTPOST)

> **Dated snapshot.** Written in the three-seed, single-machine phase (July–August
> 2026). Its numbers were superseded by the ten-seed re-runs on ckg12 under the
> uniform 400-epoch protocol; the current value of every claim is in
> `analysis/appendix.json` (and the appendix page built from it), with the
> chronology in `METHODOLOGY.md` §5.2, §6.2–6.3 and `analysis/REVIEWER_PROOFING.md`.
> Kept as the record of what was believed at the time and why.


*Authoritative record of the Yelp campaign. All numbers are aggregate over the
single fraud class, best-over-epochs unless a val-selected column is given.
Protocol matches the baselines: 50 labeled fraud + 5% normals train, 30 fraud +
1% normals val, rest test; 400 epochs; sampled-subgraph regime ([25,10]).
Runs produced by `driver_yelp.py` / `batch_lean.py` / `batch_gate.py` (these
call the same `train_outpost_v4` as `main.py`; they do not write results.csv).*

## Baselines (published, from the DEMO paper Table 2)

| Method | Yelp AUC-ROC | Yelp AUC-PR |
|---|---|---|
| DEMO | 0.7097 | 0.2238 |
| NSReg (best published PR) | 0.7015 | **0.3029** |
| ConsisGAD | 0.6988 | 0.2970 |

## OUTPOST — headline result (lean config: synthesis OFF)

The strongest, replicated configuration. Multi-sample mixup + hull-escape halo
are **disabled** (`use_mixup: false, use_halo: false`); spectral gate on.

| seed | ROC | PR | val-sel ROC | val-sel PR |
|---|---|---|---|---|
| 42 | 0.7234 | 0.3293 | 0.7073 | 0.3293 |
| 0  | 0.7546 | 0.4079 | 0.7467 | 0.3586 |
| 1  | 0.7311 | 0.3426 | 0.7307 | 0.3426 |
| **mean** | **0.7364** | **0.3599** | 0.7282 | 0.3435 |

**Every seed beats every published method on both metrics.** Mean PR 0.3599 is
+0.136 over DEMO (+61%) and +0.057 over the best published number (NSReg).
The win survives honest val-selection (mean 0.3435 > NSReg's oracle 0.3029).

## Ablations

### Synthesis is counterproductive on real fraud (the key finding)

Full config (synthesis ON) vs lean (OFF), seed 42, best-over-epochs PR:

| config | ROC | PR |
|---|---|---|
| full (mixup+halo ON) | 0.7235 | 0.3012 |
| **lean (mixup+halo OFF)** | 0.7234 | **0.3293** |

Removing OUTPOST's anomaly-synthesis block **raises** PR by +0.028 at seed 42,
and the lean config replicates far above full across seeds (mean 0.3599). This
is consistent with Phase 0: real fraud is scattered/diverse, so synthesizing
from 50 seen frauds narrows rather than expands the decision boundary.
NOTE: this ablation removes mixup AND halo together; a mixup-only vs halo-only
split is not yet run. OUTPOST's mixup follows DEMO's *paper* Definition 3.1
(convex embedding fusion); DEMO's *code* implements mixup differently (PPR
structural consistency), so this is NOT a direct test of DEMO's mechanism.

### Component ablations (full config, seed 42, best-over-epochs)

| removed component | ROC | PR | Δ PR vs full |
|---|---|---|---|
| — (full v4+SG) | 0.7235 | 0.3012 | — |
| spectral gate (A1) | 0.7232 | 0.3015 | ≈ 0 |
| pseudo-labeling (A2) | 0.7176 | 0.2852 | −0.016 |
| synthesis (A3 = lean) | 0.7234 | 0.3293 | **+0.028** |
| conformal→fixed τ (A4) | 0.7229 | 0.2979 | −0.003 |

### Spectral-gate 2×2 (lean config, gate on vs off), best-over-epochs PR

| seed | gate ON | gate OFF | Δ |
|---|---|---|---|
| 42 | 0.3293 | 0.3160 | +0.013 |
| 0  | 0.4079 | 0.4016 | +0.006 |
| 1  | 0.3426 | 0.3379 | +0.005 |
| **mean** | **0.3599** | **0.3518** | **+0.008** |

The gate is consistently positive (3/3 seeds) but small relative to seed
variance — a minor contributor, not a load-bearing mechanism.

## Honest status / open items

- results.csv Yelp cell reports the lean **seed-42** run (0.7234/0.3293) for
  protocol parity with the single-run baseline table; this multi-seed mean
  (0.7364/0.3599) is the robust headline.
- Photo cells (0.9066/0.6295) are still single-seed **full** config; a
  multi-seed pass and the small-dataset lean/full comparison remain TODO.
- mixup-only vs halo-only split not yet run.
- Direct DEMO-mechanism test (their PPR-mixup on Yelp) not run — infeasible on
  the current machine (dense PPR ~17 GB); flagged, not executed.

## Compression sweep (efficiency result, 2 seeds each)

Does the lean model hold performance when shrunk? Grid over hidden width and
the (marginal) spectral gate, seeds {42,0}. Yelp features are only 32-dim, so
the h=64 model is over-parameterized. Reference lean h=64: s42 0.3293, s0
0.4079 PR. NSReg best-published PR = 0.3029.

| config | params | vs DEMO | PR s42 | PR s0 | mean PR | verdict |
|---|---|---|---|---|---|---|
| h=64 (ref) | 27,202 | 0.80x | 0.3293 | 0.4079 | 0.369 | — |
| h=32, gate on | 11,714 | 0.34x | 0.3312 | 0.3723 | 0.352 | holds |
| **h=32, gate off** | **7,361** | **0.22x** | 0.3297 | 0.3803 | **0.355** | **holds — recommended** |
| h=16, gate on | 7,042 | 0.21x | 0.3132 | 0.3434 | 0.328 | beats field, slipping |
| h=16, gate off | 2,689 | 0.08x | 0.2746 | 0.3420 | 0.308 | breaks (s42 < NSReg) |

**Recommended efficient config: h=32, spectral gate OFF, synthesis OFF —
7,361 params (0.22x DEMO), mean PR 0.355 (both seeds > NSReg 0.303),
ROC ~0.74 (> DEMO 0.71).** Dominates h=16-gate-on at equal param budget, so
the gate is dropped. Confirms the spectral gate is removable (Yelp A/B already
showed +0.008) and that the win survives 5x parameter compression. h=16-no-gate
(0.08x) is the breaking point.

Caveat: 2 seeds per config; the headline h=32-no-gate should get seed 1 for a
3-seed number matching the reference. Produced by batch_compress.py.

---

## Gap-close batch (2026-07-27) — all numbers regenerable via `analysis/build_tables.py`

### G1. Compression headline is now 3 seeds

h=32, spectral gate OFF, synthesis OFF — **7,361 params (0.22x DEMO)**:

| seed | ROC | PR | at-best-val PR |
|---|---|---|---|
| 42 | 0.7281 | 0.3297 | 0.3297 |
| 0 | 0.7468 | 0.3803 | 0.3779 |
| 1 | 0.7283 | 0.3459 | 0.3412 |
| **mean** | **0.7344** | **0.3520** | 0.3496 |

**All three seeds beat NSReg's best-published PR (0.3029)** and DEMO's ROC
(0.7097) at 0.22x DEMO's parameters. Gap closed.

### G2. Photo re-measured under the corrected config — exact reproduction

The duplicate-`lambda_mixup` fix was validated: with the corrected yaml the
runs reproduce the earlier override-based numbers **exactly** (0.9089/0.6417,
0.8407/0.5326, 0.8424/0.5375; mean 0.8640/0.5706). Confirms both the fix and
run-to-run determinism.

### G3. Does synthesis help on clustered anomalies? — directionally yes, but WEAK

The thesis predicts synthesis *helps* where anomalies are clustered (Photo,
same-frac 0.56-0.94) and *hurts* where scattered (Yelp, 0.16). Matched-seed
Photo comparison:

| seed | full (synth ON) | lean (synth OFF) | delta |
|---|---|---|---|
| stream-42 | 0.6417 | 0.5355 | **+0.106 full** |
| t0 | 0.5326 | 0.5335 | -0.001 (tie) |
| t1 | 0.5375 | 0.5696 | -0.032 lean |
| mean | **0.5706** | 0.5462 | +0.024 full |

**Honest reading: not a clean confirmation.** The mean favours synthesis by
+0.024 PR, but that is driven entirely by one seed (stream-42); the paired
split is 1 win / 1 tie / 1 loss, well inside the +-0.03-0.07 trajectory band.

Current state of the crossover claim (synthesis helps clustered, hurts
scattered):

| dataset | anomaly same-frac | paired seeds | effect of REMOVING synthesis |
|---|---|---|---|
| Photo | 0.56-0.94 (clustered) | 3 | -0.024 PR (hurts, but 1-1-1 split) |
| Yelp | 0.16 (scattered) | **1** | +0.028 PR (helps) |

The direction matches the thesis on both datasets, but the Yelp side rests on
a **single** paired seed (full-config Yelp was only ever run at seed 42).
Completing full-config Yelp at seeds 0 and 1 would give 3 paired seeds per
dataset and turn this into a proper interaction result. That is the single
highest-value remaining experiment for the paper's central claim.

---

## CORRECTION (2026-07-28): the "synthesis hurts on Yelp" effect was overstated

Earlier text in this file and in commit messages compared **single-seed** full
config (seed 42, PR 0.3012) against the **3-seed mean** of the lean config
(0.3599) and reported the gap as ~+0.06 PR / "0.30 -> 0.36". That is an
apples-to-oranges comparison: seed 42 happens to be a poor draw for the full
config. Full-config Yelp has now been run at seeds 0 and 1 (`X_yelp_full_s*`),
giving a matched 3-seed comparison.

### Matched Yelp comparison (h=64, gate ON, identical seeds)

| seed | synthesis ON (full) | synthesis OFF (lean) | delta |
|---|---|---|---|
| 42 | 0.3012 | 0.3293 | +0.0281 |
| 0 | 0.3907 | 0.4079 | +0.0172 |
| 1 | 0.3426 | 0.3426 | +0.0000 (exact tie) |
| **mean** | **0.3448** | **0.3599** | **+0.0151** |

ROC is unchanged (full 0.7366 vs lean 0.7364).

**Corrected claim: removing synthesis gives ~+0.015 PR on Yelp, not +0.06.**
It never hurts (2 wins, 1 tie, 0 losses across seeds), but the effect is small
relative to the +-0.03-0.07 trajectory band.

### The crossover, stated honestly

| dataset | anomaly same-frac | effect of REMOVING synthesis (3 paired seeds) |
|---|---|---|
| Photo | 0.56-0.94 (clustered) | **-0.024 PR** (hurts; 1 win / 1 tie / 1 loss) |
| Yelp | 0.16 (scattered) | **+0.015 PR** (helps; 2 wins / 1 tie / 0 losses) |

The interaction runs in the predicted direction on both datasets — synthesis
pays off only where anomalies are clustered — but with n=3 per cell and effect
sizes inside the noise band, this is **suggestive, not established**. It should
be presented as a consistent directional trend with per-seed data shown, never
as a demonstrated effect.

### Good news: the headline performance result does NOT depend on this

Both configurations beat the best published Yelp numbers at 3-seed mean:

| config | ROC | PR | vs NSReg 0.3029 | vs DEMO 0.2238 |
|---|---|---|---|---|
| full (synthesis ON) | 0.7366 | 0.3448 | +0.042 | +0.121 |
| lean (synthesis OFF) | 0.7364 | 0.3599 | +0.057 | +0.136 |
| lean h=32, gate off (0.22x params) | 0.7344 | 0.3520 | +0.049 | +0.128 |

The win over the published field is robust to the synthesis choice, to the
gate, and to 5x parameter compression.

---

## HopMix (per-node view fusion) — TESTED AND REJECTED (2026-07-28)

**Hypothesis.** Phase 0 showed propagation erases scattered anomalies (Yelp
hop-0 AUC 0.631 -> ~0.55 propagated) and Step 1 found raw-feature evidence
surviving where the GNN goes blind (AUC-in-C 0.63 vs 0.53). So scoring every
node through the GNN should waste signal. HopMix (`use_hybrid`,
`OUTPOST_V4HM`) adds a propagation-free MLP scorer and blends per node,
`s = (1-w_v) s_gnn + w_v s_mlp`, with `w_v` from a label-free local-scatter
context. +2,210 params (9,571 = 0.28x DEMO).

**Result: it HURTS, consistently.** Matched against the h=32 gate-off baseline:

| seed | baseline PR | HopMix PR | delta | w_anom | w_norm |
|---|---|---|---|---|---|
| 42 | 0.3297 | 0.2876 | **-0.0421** | 0.321 | 0.306 |
| 0 | 0.3803 | 0.3708 | **-0.0095** | 0.260 | 0.249 |
| 1 | 0.3459 | 0.3350 | **-0.0109** | 0.294 | 0.275 |
| **mean** | **0.3520** | **0.3311** | **-0.0208** | | |

ROC identical (0.7344 both). Worse on **3/3 seeds**.

**The interpretability check passed while the mechanism failed.** `w_anom >
w_norm` in 3/3 seeds (mean gap +0.015) — the model *does* learn to lean on raw
features more for anomalies, exactly as the Phase-0 law predicts. But the
learned blend puts substantial global weight (0.25-0.32) on a view that is far
weaker overall (raw-feature AUC ~0.63 vs the GNN's ~0.73), so it dilutes the
strong scorer everywhere to rescue a minority of nodes. The law is real; this
particular way of exploiting it is not.

**Kept as a flag (`use_hybrid`, default OFF)** so the negative result stays
reproducible; it is an honest ablation showing the design space was explored.

### Running tally of novel-mechanism attempts

| attempt | outcome |
|---|---|
| Adaptive band gate | abandoned pre-test — occupied by AMNet/BWGNN/GHRN/SAGAD |
| Spectral gate (`use_fview_gate`) | +0.008 PR — real direction, too small to matter |
| HopMix (`use_hybrid`) | **-0.021 PR — actively harmful** |

Three attempts, no positive mechanism. The measured contribution remains the
**analysis** (detectability law, benchmark-validity critique) plus a **leaner,
cheaper model** that beats the published field — not a new component.

---

## SimSample: similarity-ordered neighbour sampling — **ACCEPTED** (2026-07-29)

Zero-parameter mechanism (`sim_topk_frac`). Neighbour lists are sorted once by
feature cosine similarity; a fraction of each hop's sampling budget is drawn
from the most similar neighbours instead of uniformly. Label-free, no added
parameters, no sampling-time cost (ordering materialised at build time).

**Mechanism check (run BEFORE training, the discipline earlier attempts
lacked):** sampled-neighbour label agreement 0.8590 (uniform) -> 0.8784
(frac 0.5) -> 0.8861 (frac 1.0). Labels used only to measure purity.

**Result — Yelp, h=32, gate off, lean, 3 seeds, 7,361 params (0.22x DEMO):**

| config | ROC | PR | val-sel PR | delta PR | wins | purity |
|---|---|---|---|---|---|---|
| baseline (uniform) | 0.7344 | 0.3520 | 0.3496 | — | — | 0.8590 |
| SimSample 0.5 | 0.7503 | 0.3847 | 0.3761 | +0.0327 | 3/3 | 0.8784 |
| **SimSample 1.0** | **0.7584** | **0.4025** | **0.3972** | **+0.0506** | **3/3** | 0.8861 |

Per-seed PR deltas — frac 0.5: +0.053 / +0.034 / +0.011; frac 1.0: +0.068 /
+0.054 / +0.030. **6/6 runs improve.**

**Why this is causal, not a draw:**
1. Every seed improves at both strengths (6/6).
2. **Monotone dose-response**: neighbour purity 0.859 -> 0.878 -> 0.886 tracks
   delta PR 0 -> +0.033 -> +0.051. The effect scales with the intervention.
3. Effect (+0.051) exceeds the +-0.03-0.07 trajectory band, and the honest
   val-selected metric moves equally (+0.048) — so it is not oracle
   peak-harvesting.

**vs published:** PR 0.4025 = **+32.9% over NSReg's best-published 0.3029** and
+79.9% over DEMO's 0.2238. Every seed and every val-selected value also clears
NSReg.

**Prior-art positioning (must be stated in the paper):** neighbour selection
for fraud graphs is established — CARE-GNN (CIKM'20) does RL-based label-aware
neighbour selection, GHRN (WWW'23) prunes edges spectrally. Ours is the
*zero-parameter, label-free, sampler-level* form, motivated by and predicted
from the Phase-0 detectability law. Position as "the law applied at the
sampling layer", cite that lineage, do not claim neighbour selection as new.

**Running tally of novel-mechanism attempts (updated):**

| attempt | outcome |
|---|---|
| Adaptive band gate | abandoned pre-test — prior art |
| Spectral gate | +0.008 — too small |
| HopMix | -0.021 — harmful, rejected |
| **SimSample** | **+0.051, 6/6 seeds, dose-response — ACCEPTED** |

### SimSample on Photo — large NEGATIVE effect (prediction wrong in magnitude)

Pre-stated prediction: the law says SimSample should help scattered anomalies
(Yelp) and do little on clustered ones (Photo), so a "small or neutral" Photo
effect was predicted. **That magnitude prediction was WRONG** — the effect is
large and negative:

| Photo (3 matched seeds) | ROC | PR |
|---|---|---|
| baseline (uniform) | 0.8640 | 0.5706 |
| SimSample 1.0 | 0.7751 | 0.4898 |
| **delta** | **-0.0889** | **-0.0808** |

The *direction* of the law's prediction holds (no benefit on clustered
anomalies) but the mechanism is evidently stronger than "does nothing".

### The crossover, with both effects large

| dataset | anomaly regime | SimSample 1.0 effect |
|---|---|---|
| Yelp (same-frac 0.16, deg ~167) | scattered, dense | **+0.051 PR** |
| Photo (same-frac 0.56-0.94, sparse) | clustered, sparse | **-0.081 PR** |

### CONFOUND — must be resolved before any claim

`sim_topk_frac=1.0` changes TWO things: neighbour purity **and** sampling
determinism (no randomness left in neighbour choice). Our own calibration
found the stochastic sampled-subgraph regime to be a key driver on Photo, so
the Photo damage may be lost stochasticity rather than purity — and by the same
logic part of the Yelp gain could be determinism, not similarity.

Two controls launched (`batch_control.py`):
- **C1 Yelp placebo** (`sim_shuffle=True`, frac 1.0): identical determinism,
  random neighbour ordering. Verified to reproduce baseline-level purity
  (0.8532 vs uniform 0.8590 vs SimSample 0.8821), so it isolates determinism
  cleanly. A ~+0.05 gain here would refute the purity explanation entirely.
- **C2 Photo frac=0.5**: retains half-stochastic sampling. If the damage
  largely disappears, Photo was hurt by determinism, not purity.

Until C1/C2 report, the SimSample result should be described as
"similarity-ordered sampling improves Yelp by +0.051 PR", NOT as
"neighbour purity causes the gain".

### CONTROLS RESOLVED (2026-07-29): purity is the cause, determinism is not

**C1 — Yelp placebo.** Deterministic sampling with RANDOM neighbour ordering:
identical loss of stochasticity, identical parameters, only the ordering
differs.

| Yelp config | purity | ROC | PR | delta PR |
|---|---|---|---|---|
| baseline (uniform, stochastic) | 0.8590 | 0.7344 | 0.3520 | — |
| **PLACEBO (determ., random order)** | 0.8532 | 0.7327 | 0.3536 | **+0.0016** |
| SimSample 0.5 | 0.8784 | 0.7503 | 0.3847 | +0.0327 |
| **SimSample 1.0** | 0.8821 | 0.7584 | **0.4025** | **+0.0506** |

**Determinism alone buys nothing (+0.002).** The entire +0.051 is attributable
to *which* neighbours are sampled. Purity and PR move together monotonically
across all four conditions (0.853/0.859/0.878/0.882 -> 0.354/0.352/0.385/0.403).

**C2 — Photo dose-response.** The damage also scales with purification
strength, so it is a genuine purity effect on clustered anomalies, not lost
stochasticity:

| Photo config | ROC | PR | delta PR |
|---|---|---|---|
| baseline (uniform) | 0.8640 | 0.5706 | — |
| SimSample 0.5 | 0.8017 | 0.5039 | -0.0667 |
| SimSample 1.0 | 0.7751 | 0.4898 | -0.0808 |

### Defensible claim (now earned)

Neighbour purification helps precisely where anomalies are **scattered among
camouflage edges** (Yelp +0.051 PR, placebo-controlled) and **hurts where
anomalies are clustered** (Photo -0.081 PR), with the regime identifiable in
advance from a label-free graph statistic (same-class fraction / neighbour
purity). SimSample is therefore a **conditional** component whose on/off switch
is *predicted by the Phase-0 law*, not tuned per dataset.

Evidence quality: dose-response in both directions + placebo control +
mechanism verified before training + 3 seeds per cell + honest val-selected
metrics move with the oracle ones.

---

## CRITICAL CORRECTION (2026-07-30): the SimSample "crossover" conflates TWO mechanisms

A direct manipulation check — measuring what `sim_topk_frac=1.0` actually does
to the sampled neighbourhoods — shows the flag performs a **different
intervention** on dense vs sparse graphs:

| dataset | median deg vs budget | purity delta | edges/batch | what actually happened |
|---|---|---|---|---|
| yelp | 168 vs 25 | **+0.0183** | 0.98x | selects similar neighbours — INTENDED mechanism |
| photo | 22 vs 25 | **-0.0058** | **1.38x** | no purification; removes edge subsampling |
| computers | 22 vs 25 | **-0.0030** | **1.36x** | no purification; removes edge subsampling |

**On Photo and Computers, purity did not increase — it slightly DECREASED.**
Because median degree (22) is below the hop-1 budget (25), deterministic top-k
takes all ~22 distinct neighbours whereas uniform sampling-with-replacement
dedups to ~14. So the flag removes a **dropout-like edge subsampling
regulariser** rather than purifying anything.

### Consequences (what must be withdrawn)

1. **The "crossover" is NOT one mechanism with two signs.** Yelp's +0.051 comes
   from a purity increase; Photo's -0.081 and Computers' -0.013 come from lost
   edge-dropout regularisation. Presenting them as a single interaction would
   be wrong.
2. **Rule R2 ("purification hurts clustered anomalies") was never tested.** No
   purification occurred on the clustered datasets. The pre-registered
   Computers prediction got the *direction* right for the *wrong reason* —
   which is not a successful forecast.
3. **The CS withdrawal is doubly justified**: degree 6 vs budget 25 makes it an
   even more extreme edge-count manipulation, not a purity test.

### What still stands (unaffected)

- **Yelp +0.051 AUC-PR**: purity genuinely rose (+0.018), edge count was flat
  (0.98x), and the placebo control (deterministic + random ordering) gained
  nothing (+0.002). On dense graphs the mechanism works as described and is
  causally attributed.
- The headline performance (Yelp 0.7584 ROC / 0.4025 PR beating all published
  methods), the parameter/efficiency claims, and the Phase-0 detectability law
  are all independent of this correction.

### Honest framing for the paper

Claim only what is supported: *"On dense graphs where degree exceeds the
sampling budget, similarity-ordered neighbour selection raises neighbourhood
purity and improves detection of scattered anomalies (+0.051 AUC-PR,
placebo-controlled). Where degree falls below the budget the operation cannot
select and instead removes edge subsampling, which is harmful — so the method
is applicable only in the dense regime, identifiable in advance from
degree vs budget without labels."*

Do NOT claim a purity-driven crossover across regimes.

### Also noted: the "12% faster" manipulation check is invalid

The Colab runs showed SimSample ~13% faster (1635s vs 1880s mean), cited as
evidence the manipulation worked. Mechanically it should be ~1.36x SLOWER on
Computers (more surviving edges -> more aggregation work). The timing
difference is therefore unexplained and most likely VM variance; it is not
evidence about the manipulation. The purity/edge-count measurement above is
the valid check.
