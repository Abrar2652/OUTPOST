# Why low-pass propagation erases scattered anomalies: a signal-to-noise account

*Companion theory for the Phase-0 empirical law (`PHASE0_FINDINGS.md`).
Goal: explain, from first principles, the measured two-regime behaviour —
propagation **amplifies** homophilous anomaly classes (+0.27..+0.48 AUC) and
**erases** scattered ones (gain ≤ 0, below chance) — and predict the crossover.*

Positioning: the propagation mechanics below specialize known
oversmoothing / heterophily spectral analysis (Oono & Suzuki 2020; NT & Maehara
2019; the GHRN/BWGNN right-shift view). The **novelty** is (i) applying it to
*open-set anomaly detectability* rather than classification accuracy, (ii) the
resulting crossover-homophily prediction, and (iii) coupling it to the
self-training fixed point that certifies erased anomalies as normal.

---

## 1. Setup

Graph `G=(V,E)`, symmetric-normalized propagation operator
`Â = D^{-1/2}(A+I)D^{-1/2}`, depth-`k` features `X^{(k)} = Â^k X`.

Binary generative model (one anomaly vs normal). Every node carries
`x_u = m_{c(u)} + ε_u`, with class mean `m_c` and i.i.d. noise
`ε_u ~ (0, σ² I)`. Write the **anomaly signal** `δ = m_a − m_n` (`Δ := ||δ||`).

For an anomaly node `v`, let its **homophily ratio** be
`h_v = |{u ∈ N⁺(v): u anomalous}| / d_v`, where `N⁺(v)` is the closed
neighbourhood and `d_v = |N⁺(v)|`. (`h_v` low ⇒ scattered; high ⇒ clustered.)

**Detectability.** Take any prototype/linear normal-vs-anomaly detector. Its
per-node discriminability is monotone in the **signal-to-noise ratio**
`SNR(v) = (deviation of E[x_v] from the normal mean, along δ) / (std of x_v)`.
AUC is a monotone function of SNR under this model, so we track SNR.

Baseline (no propagation): `SNR₀ = Δ / σ`.

---

## 2. One-hop transformation (the core result)

**Assumption A1 (mean-field neighbourhood).** Treat `Â`'s action on `v` as the
`d_v`-point degree-weighted average over `N⁺(v)`; approximate weights as uniform
`1/d_v` (exact for regular graphs; first-order otherwise).

**Assumption A2 (independent noise, shared anomaly signal).** Neighbour noises
are independent; anomalous neighbours share mean `m_a`, normal ones `m_n`.

**Proposition 1 (one-hop SNR).** Under A1–A2,

```
   E[x_v^{(1)}] − m_n = h_v · δ          (signal shrinks by h_v: bias to normal)
   Var[x_v^{(1)}]      = σ² / d_v         (noise shrinks by 1/√d_v: averaging)
   ⇒  SNR₁(v) = h_v · √d_v · SNR₀ .
```

*Proof.* `x_v^{(1)} = (1/d_v) Σ_{u∈N⁺(v)} x_u`. Linearity of expectation gives
`E = h_v m_a + (1−h_v) m_n = m_n + h_v δ`. Independence (A2) gives
`Var = (1/d_v²)·Σ σ² = σ²/d_v`. Divide signal `h_vΔ` by noise `σ/√d_v`. ∎

The single factor `κ_v := h_v·√d_v` governs everything: propagation helps iff
`κ_v > 1`.

---

## 3. Consequences

> **STATUS 2026-07-25 — the closed-form crossover below is FALSIFIED.**
> `verify_theory.py` tested `sign(gain) = sign(h − d^{-1/2})` on all 16 classes:
> 8/16 (chance). Real degrees are large (22–160) so `d^{-1/2}` is tiny and the
> rule predicts "amplify" almost everywhere, but feature-visible classes erase
> regardless of `h`. The *continuous* margin still ranks with the gain
> (Spearman +0.756, p=7e-4), so the ingredients are right but the functional
> form is not. Two failures of §2's model cause this: (i) neighbour noise is
> correlated, so it does NOT fall like `√d`; (ii) there is an AUC ceiling —
> classes already separable at hop 0 (`SNR₀` high) can only lose. A correct
> statement must carry BOTH feature-visibility (`SNR₀`) and homophily, and a
> realistic noise-reduction factor `ρ(d) ≪ √d`. Retained below as the naive
> first model and its refutation; §5 verification stands as the falsifier.

**Corollary 1 (two regimes + crossover) — NAIVE, FALSIFIED.** One propagation
step increases detectability iff `h_v > h*(v)` with the **crossover homophily**

```
        h*(v) = d_v^{−1/2}.       [not supported by data — see STATUS above]
```

- **Homophilous** anomalies (`h_v > h*`): noise-averaging beats signal-bias ⇒
  SNR rises ⇒ *amplification*.
- **Scattered** anomalies (`h_v < h*`): signal collapses faster than noise
  falls ⇒ SNR drops ⇒ *erasure*.

For typical effective degrees `d_v ≈ 3–10`, `h* ≈ 0.32–0.58`. The Phase-0 sign
flip sits exactly here: Photo class 7 (`h=0.56`) → gain **−0.095**; Computers
class 6 (`h=0.55`) → **+0.05**; every class with `h ≥ 0.80` → large positive
gain. The theory predicts the crossover band and the data's sign change lands in
it. (Exact constant depends on degree/noise; we claim the band, not 3 digits.)

**Corollary 2 (exponential erasure with depth).** Iterating Proposition 1 under
a homogeneous-neighbourhood approximation, `SNR_k ≈ (h_v √d_v)^k · SNR₀`. For
scattered anomalies (`κ_v<1`) detectability decays **geometrically** in depth —
driven below any fixed threshold in finitely many hops. This is the mechanism
by which a scattered class reaches *below-chance* AUC at hop 3 (Photo class 7:
0.530 → 0.435).

---

## 4. The self-training fixed point (softer — needs assumptions)

Let `f_t` be the detector at round `t`, `τ` the pseudo-normal confidence
threshold, and let round `t+1` train on pseudo-labels from `f_t`.

**Proposition 2 (sketch).** If a scattered anomaly `v` has propagated score
`s_t(v) < τ` (Cor. 1–2 give conditions on `h_v, d_v, k`), then `v` is pseudo-
labelled normal and contributes a normal-class gradient at `t+1`, which does not
increase `s_{t+1}(v)`; hence `{s_t(v)}` is non-increasing and `v` is an absorbing
"certified-normal" state. Confidence-gated self-training therefore has a fixed
point in which all sufficiently scattered anomalies are labelled normal.

*Status.* The monotonicity step needs a stated assumption on the training
dynamics (e.g. a linear/NTK-regime detector, or a margin argument). This is the
part to firm up; it is the theoretical form of the measured pathology (~6.5k
Photo pool nodes pseudo-normal, absorbing nearly all of class 7).

---

## 5. A new, falsifiable prediction (checkable on data we already have)

Corollary 1 predicts **`sign(smoothing_gain_c) = sign(h_c − d_c^{−1/2})`** per
anomaly class `c`, using only `same_frac` (=`h_c`) and the class's mean degree
`d_c` — both already in `analysis/results/spectral_*.csv` (degree is one extra
column to emit). Verification is a few seconds of pandas, **no training runs**:
for each of the 16 classes, check the sign of `smoothing_gain` against the sign
of `(h_c − d_c^{−1/2})`. A high agreement rate (the theory predicts ~all 16)
turns Cor. 1 from plausible into tested.

---

## 6. Honest limitations

- A1 (uniform mean-field) ignores exact symmetric-norm weights and degree
  heterogeneity; a per-node version keeps `Â_vu` explicitly.
- A2 (isotropic shared-signal noise) ignores intra-anomaly diversity — real
  fraud has *variable* `δ_u`, which only strengthens erasure (adds signal
  variance) but weakens the clean `h_v√d_v` form.
- AUC↔SNR monotonicity is exact for Gaussian equal-covariance; approximate
  otherwise.
- Prop. 2 is a sketch, not yet a theorem.

## 7. What this buys the paper

A first-principles account that (a) *derives* the measured two-regime law,
(b) predicts a crossover `h* = d^{-1/2}` confirmed by the existing sign-flip,
(c) explains why every low-pass method (DEMO, OUTPOST) hits the same Photo-class-7
wall, and (d) frames the self-training pathology as a fixed point. This is the
component that lifts the contribution from "strong empirics + analysis" toward a
top-venue theoretical claim — provided §4 is firmed up and §5 is run.
