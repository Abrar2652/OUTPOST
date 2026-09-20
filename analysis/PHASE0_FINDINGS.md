# Phase-0 Findings: Unseen-Anomaly Detectability Is a Topological Property

> **Dated snapshot.** Written in the three-seed, single-machine phase (July–August
> 2026). Its numbers were superseded by the ten-seed re-runs on ckg12 under the
> uniform 400-epoch protocol; the current value of every claim is in
> `analysis/appendix.json` (and the appendix page built from it), with the
> chronology in `METHODOLOGY.md` §5.2, §6.2–6.3 and `analysis/REVIEWER_PROOFING.md`.
> Kept as the record of what was believed at the time and why.


*2026-07-22 · produced by `analysis/spectral_diagnostic.py` (seed 42, per-class
CSVs in `analysis/results/`). Pre-registered predictions P1–P4 are stated in
the script docstring; none were altered after seeing results.*

## Motivation

OUTPOST v4 and DEMO — different mechanisms, same low-pass GraphSAGE-family
backbone — converge to statistically indistinguishable performance on Photo
(matched-seed means 0.8640/0.5706 vs 0.8559/0.5419), and both are bounded by
the same failure: unseen-class recall (Photo class 7 stuck at ROC ≈ 0.55–0.77
for every method tried). Phase 0 asks *why*, with a training-free instrument
that removes the model from the equation: spherical k-means prototypes over
5% labeled normals at propagation depths k ∈ {0..3}, scored per anomaly class.

## Headline result

**A single topological quantity — the median same-class neighbor fraction of
an anomaly class — predicts its best achievable detectability across every
dataset tested: Spearman ρ = 0.897 (p = 2.5e-6, n = 16 classes / 4 datasets).**

| dataset | class | same-class frac | AUC @ hop 0 | best AUC | smoothing gain |
|---|---|---|---|---|---|
| photo | 0 | 0.94 | 0.644 | **0.912** | **+0.268** |
| photo | 7 | 0.56 | 0.530 | 0.556 | **−0.095** |
| computers | 5 | 0.94 | 0.442 | **0.924** | +0.482 |
| computers | 9 | 0.81 | 0.462 | 0.867 | +0.405 |
| computers | 0 | 0.80 | 0.484 | 0.793 | +0.310 |
| computers | 3 | 0.60 | 0.488 | 0.759 | +0.271 |
| computers | 6 | 0.55 | 0.332 | **0.384** | +0.053 |
| cs | (8 classes) | 0.64–1.00 | **0.87–0.99** | 0.88–1.00 | ≈ 0 |
| **yelp (real fraud)** | 1 | **0.16** | 0.631 | 0.631 | **−0.081** |

## The two-regime law

Detectability of an anomaly class under low-pass graph learning is governed by
two factors, in order:

1. **Feature visibility.** If raw features separate the class (CS: hop-0 AUC
   0.87–0.99), it is detectable regardless of topology; propagation is
   unnecessary and mildly harmful (all CS gains ≈ 0 or negative).
2. **Homophily amplification.** If features are insufficient (Photo,
   Computers, Yelp: hop-0 AUC 0.33–0.64), propagation is the only lever, and
   it is a *homophily amplifier*: among these 8 feature-invisible classes,
   same-class fraction predicts best AUC at ρ = 0.929 (p = 9e-4) and predicts
   the smoothing gain itself at ρ = 0.786 (p = 0.02). Homophilous classes are
   amplified into detectability (+0.27 to +0.48); scattered classes are
   **averaged into normality** (gains ≤ +0.05, often negative — Photo class 7
   ends *below chance* at hop 3).

Within classes the same force operates per node: same-class neighbor density
correlates with per-node anomaly score in 14/16 classes (P2; e.g. ρ = +0.64,
p ≈ 1e-44 for Photo class 0). Feature contrast does *not* explain the between-
class gap (Photo P1 feat-dissim sepAUC 0.551 vs same-frac sepAUC 0.839).

## The benchmark-validity finding (P4)

Real fraud (Yelp) has same-class fraction **0.160** — dramatically more
scattered than even the hardest synthetic class (Photo 7: 0.556) — with weak
features and negative smoothing gain. The synthetic benchmarks' *easy*
anomaly classes (same-frac 0.8–1.0) are homophilous minority clusters whose
detectability is an artifact of relabeling coherent classes as "anomalies."
**Methods selected for winning on Photo/Computers/CS are being selected for
homophily exploitation — precisely the property that fails on real fraud.**
This retrodicts DEMO's Yelp AUC-PR collapse (0.2238, behind NSReg's 0.3029).

## Implications for OUTPOST / the paper

1. **Explains the Photo plateau**: every current method (DEMO, OUTPOST v1–v4)
   scores class 7 through a low-pass bottleneck; no amount of head/loss/
   threshold tuning moves it. The binding constraint is the backbone's
   spectral response, not the open-set machinery.
2. **Explains the self-training pathology**: scattered anomalies arrive at
   the classifier pre-smoothed into normality, so confidence-based
   pseudo-labeling certifies them normal (measured: ~6.5k pool nodes
   pseudo-normal on Photo, absorbing nearly all of class 7).
3. **Prescribes the mechanism**: pseudo-label gating and scoring must be
   *frequency-aware* — a node's eligibility for a "normal" pseudo-label
   should depend on its local spectral signature, not its (low-pass)
   confidence. This is the Phase-2 design target.
4. **Names the win condition**: the method's value concentrates where
   features are weak AND anomalies are scattered — Yelp — with predicted
   parity (not gains) on CS-like feature-visible benchmarks. The paper should
   state this asymmetry in advance; it is the falsifiable signature of the
   theory.

## Threats to validity

- `same_frac` uses ground-truth labels; it is a *diagnostic*, not a deployable
  signal. Phase 2 must find label-free proxies (local Dirichlet energy,
  prototype-disagreement across hops) and show they track it.
- Prototype detector is one instrument; GNN-based detectors could differ
  (mitigated: v4/DEMO unseen-class results on Photo match its ordering).
- 16 classes is a small sample for the cross-dataset regression; ogbn-arxiv
  should be added (large, many rotation classes) before publication.
- CS ceiling effects compress its contribution to P1/P3 (acknowledged above;
  it anchors the feature-visibility regime instead).

## Verdict

Phase 0 **passes decisively**. Proceed to: (1) label-free spectral-signature
proxies; (2) the spectrum-gated self-training mechanism; (3) a suppression
bound formalizing "low-pass smoothing + confidence gating ⇒ scattered-anomaly
erasure" as a function of local heterophily.

---

## Addendum (2026-07-23): Step-3 Photo A/B of the Spectral Gate

Matched-seed A/B of `use_fview_gate` (commit 4a29898) against v4 references,
fixed split 42 (aggregate ROC/PR):

| training seed | v4 | v4+SG | delta |
|---|---|---|---|
| stream-42 | 0.9089 / 0.6417 | 0.8756 / 0.5580 | -0.033 / -0.084 |
| 0 | 0.8407 / 0.5326 | 0.8292 / 0.5255 | -0.012 / -0.007 |
| 1 | 0.8424 / 0.5375 | 0.8849 / 0.5684 | +0.043 / +0.031 |
| mean | 0.8640 / 0.5706 | 0.8632 / 0.5506 | -0.001 / -0.020 |

Verdict: neutral on Photo (per-seed swings match known trajectory variance
+-0.03..0.07). Consistent with the Step-1 measurement that the gate's signal
is at chance on Photo's feature-bland scattered class -- there is nothing for
the F-view to act on -- but this is predicted-null evidence, not confirmation.
The mechanism's decisive test is Yelp, where Step-1 measured the signal at
AUC 0.63 inside the blind region. Safety criterion (no small-dataset damage)
is met within noise; PR mean (-0.020) to be rechecked after Yelp tuning.
