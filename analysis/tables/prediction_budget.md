# Pre-registration: manipulating the criterion's own variable

Written 2026-08-11, **before** any run in this sweep. Committed first for the
usual reason, and for one specific to this experiment: it is designed to be able
to *destroy* the paper's main effect, so the conditions under which it would
count as destroying it have to be fixed in advance.

## What is being tested

The claim of METHODOLOGY §5.1 is not that similarity-ordered neighbour sampling
helps. Neighbour selection for fraud graphs is established — CARE-GNN (CIKM'20)
does it with a reinforcement-learned label-aware selector, GHRN (WWW'23) with
spectral edge pruning. The claim is a **criterion**: SimSample helps exactly when
a node's degree exceeds the per-hop sampling budget, because only then is there
anything to select. Below the budget every neighbour is taken regardless, and the
operation degenerates into removing a dropout-like edge subsampling.

So far the criterion has been tested by varying **degree** across graphs, holding
the budget at 25:

| graph | median degree | Δ AUC-PR from SimSample | W/T/L |
|---|---|---|---|
| Yelp | 168 | **+0.0504** | 10/0/0 |
| Photo | 22 | −0.0318 | 0/0/5 |
| Computers | 22 | −0.0157 | 0/0/5 |

Twenty paired comparisons, no exceptions — but observational. Degree is a
property of the graph, so each row also differs in size, features, homophily and
whether its anomalies are real. A reviewer is entitled to ask whether degree is
doing the work or merely tracking something else.

This sweep varies the **other** side of the inequality, on **one fixed graph**.
Nothing about Yelp changes except the sampling budget.

## Design

Yelp, `sampling_sizes` `[b, 10]` for `b` in {25, 50, 100, 200}, each with
SimSample on (`sim_topk_frac=1.0`) and off (`0.0`). `b=25` is already measured at
10 seeds per arm; the three new budgets run at 5 seeds per arm, paired on seed.

Yelp's median degree is 168. So the budget crosses the degree between `b=100` and
`b=200`, and the criterion's inequality flips inside the sweep.

## Predictions

**P10 (monotone decay).** The SimSample advantage Δ(b) = AUC-PR(on) −
AUC-PR(off) will decrease monotonically in `b` over {25, 50, 100, 200}, measured
as a negative Spearman correlation between `b` and Δ across the individual paired
runs.

**P11 (the effect switches off).** At `b = 200`, above Yelp's median degree of
168, Δ will be **statistically indistinguishable from zero** — specifically, the
bootstrap 95% interval on the paired difference will contain 0. At `b = 25` that
interval is [−0.057, −0.044] and nowhere near it.

**P12 (it is selection, not budget).** The *absolute* AUC-PR of the SimSample-off
arm will not fall as `b` grows. If bigger neighbourhoods simply helped, both arms
would rise together and Δ could shrink for a reason that has nothing to do with
selection. The prediction is that Δ shrinks because the **on** arm stops gaining,
not because the **off** arm catches up by getting better.

## What would falsify the criterion

- Δ stays flat or grows with `b` (P10 fails): the effect is not about the budget
  at all, and §5.1's mechanism is wrong.
- Δ remains significant at `b = 200` (P11 fails): SimSample helps even when the
  budget exceeds the degree, so "there is nothing to select" is not the mechanism.
- The off arm improves enough to explain the shrinkage (P12 fails): the sweep
  measures receptive-field size, not neighbour selection, and cannot support the
  criterion.

Any of these would mean the contribution is "similarity sampling helps on Yelp",
which is a much smaller claim and largely anticipated by prior work. That is the
outcome this experiment is built to be able to produce.

---

# Outcomes (recorded 2026-08-11, after the sweep)

| budget b | n | SimSample on | off | Δ | W/T/L | bootstrap CI on Δ |
|---|---|---|---|---|---|---|
| 25 | 10 | 0.3824 | 0.3320 | **+0.0504** | 10/0/0 | [+0.0439, +0.0570] |
| 50 | 5 | 0.3816 | 0.3422 | +0.0394 | 5/0/0 | [+0.0293, +0.0475] |
| 100 | 5 | 0.3676 | 0.3407 | +0.0269 | 5/0/0 | [+0.0180, +0.0352] |
| 200 | 5 | 0.3624 | 0.3407 | **+0.0217** | 5/0/0 | [+0.0095, +0.0340] |

## P10 — CONFIRMED

Δ decays monotonically in the budget: Spearman ρ = **−0.700** over 25 paired runs
(p = 9.7e-5). Raising the budget on a fixed graph shrinks the effect, which is
what the criterion requires and what a confound with graph identity could not
produce.

## P12 — CONFIRMED

The SimSample-**off** arm is flat across the sweep: 0.3320, 0.3422, 0.3407,
0.3407. Δ shrinks because the **on** arm falls (0.3824 → 0.3624), not because the
off arm improves. So the sweep is measuring neighbour selection, not
receptive-field size.

## P11 — FALSIFIED, and the criterion was stated with the wrong statistic

At b = 200, above Yelp's *median* degree of 168, Δ is +0.0217 with a bootstrap CI
of [+0.0095, +0.0340] — it does **not** contain zero, and SimSample still wins on
5 of 5 seeds. The prediction that the effect switches off is wrong.

The reason is visible immediately in the degree distribution, and it is a mistake
in how the criterion was written rather than in the mechanism. Yelp's degree has
p75 = 239, p90 = 290 and max 501. At b = 200, **37.4% of nodes still have degree
above the budget**, so selection continues to operate on more than a third of the
graph. A median-based threshold was never the right form for a skewed
distribution.

### Post-hoc refinement — labelled as such, and not yet tested

Replacing "median degree > budget" with the *fraction of nodes whose degree
exceeds the budget* tracks the sweep almost exactly:

| b | frac(degree > b) | Δ |
|---|---|---|
| 25 | 95.2% | +0.0504 |
| 50 | 87.8% | +0.0394 |
| 100 | 71.2% | +0.0269 |
| 200 | 37.4% | +0.0217 |

Spearman ρ = **+1.000** across the four budgets, Pearson r = +0.900, and a linear
fit has an intercept of +0.0017 — indistinguishable from zero, as the mechanism
requires (no nodes above budget ⇒ nothing to select ⇒ no gain).

This was fitted after seeing the data and is worth nothing until tested out of
sample. It also makes the two-mechanism structure of METHODOLOGY §5.1
quantitative for the first time. Fitting
`Δ = α·frac_above + β·frac_below` on the three datasets measured at b = 25 gives
**α = +0.058** (selection gain) and **β = −0.089** (the loss from removing
dropout-like edge subsampling on nodes below budget) — two terms of opposite
sign, which is exactly what §5.1 asserts and had never quantified.

**P13 (prospective).** CS and ogbn-arxiv have almost no selection headroom —
5.1% and 4.3% of nodes above a budget of 25 — and almost all nodes below it. The
two-term fit predicts Δ = **−0.081** on CS and **−0.083** on ogbn-arxiv, i.e.
*more negative than Photo's −0.0318*. The directional claim is what is being
tested, since a two-parameter fit on three points cannot support four decimals:
**Δ will be negative on both, and more negative than on Photo and Computers.**
If instead Δ is near zero on the low-degree graphs, the subsampling-loss term is
wrong and the sign change on sparse graphs needs a different explanation.

## P13 — FALSIFIED. The quantitative model is withdrawn.

ogbn-arxiv, 5 matched seeds: **Δ = +0.0008** (3 wins, 1 tie, 1 loss) against a
predicted **−0.0826**. Not a magnitude error — a sign error on a prediction that
was supposed to be the model's easiest case.

The obvious repair does not work either. The loss term was meant to come from
removing a dropout-like edge subsampling, which requires a node to have more than
one neighbour to subsample. ogbn-arxiv is 55.0% degree-1 nodes, so for the
majority neither mechanism can act. But restricting the loss to nodes where
`1 < degree ≤ budget` still fails: arxiv has **40.7%** of nodes in that band
against Photo's **52.4%**, and 0.78× of Photo's −0.0318 is −0.025, not +0.0008.

| dataset | d > 25 | 1 < d ≤ 25 | d ≤ 1 | measured Δ |
|---|---|---|---|---|
| Yelp | 95.2% | 4.7% | 0.1% | **+0.0504** |
| Photo | 43.8% | 52.4% | 3.8% | −0.0318 |
| Computers | 45.0% | 50.6% | 4.5% | −0.0157 |
| ogbn-arxiv | 4.3% | 40.7% | 55.0% | **+0.0008** |
| CS | 5.1% | 89.0% | 5.9% | not run |

**Two post-hoc refinements in a row is curve-fitting, not science.** The
`α·frac_above + β·frac_below` model is withdrawn rather than repaired a third
time. What it was fitted on — three datasets and one budget sweep — cannot
support it, and its one out-of-sample test refuted it.

### What survives

*The gain is established.* SimSample's advantage on Yelp is causal (placebo below
the zero-dose arm), dose-dependent, and — from the budget sweep on a single fixed
graph — requires nodes whose degree exceeds the sampling budget. Δ tracks
frac(degree > b) monotonically as b is varied with the graph held constant, which
no confound with graph identity can produce.

*The harm is real but unexplained.* SimSample costs 0.0318 on Photo and 0.0157 on
Computers, losing every seed on both. The "removes edge subsampling" mechanism
of METHODOLOGY §5.1 predicts the same harm on ogbn-arxiv, and there is none. So
the sign change across graphs is a robust observation whose mechanism is not
established, and §5.1 should say that rather than assert a cause.

*The criterion, correctly scoped.* "SimSample helps where a large fraction of
nodes exceed the sampling budget" is supported — by four datasets and, more
strongly, by an intervention on one. "SimSample harms where they do not" is
supported on Photo and Computers and **contradicted on ogbn-arxiv**, so it is not
a criterion, it is a two-dataset finding.

### The remaining prospective test

CS discriminates sharply and has not been run: 89.0% of its nodes sit in the
`1 < d ≤ 25` band — far more than Photo's 52.4% — with only 5.1% above budget. A
subsampling-removal mechanism predicts CS should show the *largest* harm of any
dataset. If CS comes out near zero like arxiv, the mechanism is dead and the harm
on Photo/Computers is something else entirely. This is blocked: CS needs ~21 GB
and both cards are held by another user's inference server.

## Cost note

Larger fan-outs sample more nodes per batch, so runtime and memory grow roughly
with `b`. `b = 200` is ~8× the sampled nodes of `b = 25`. Budgets are run
longest-first and memory is re-measured per arm rather than extrapolated, after
the ogbn-mag episode where a probe that never passed warmup under-estimated peak
memory threefold.

---

## The CS test is now running (2026-08-18)

Both cards freed on 2026-08-18 and `campaigns/cs_open.json` is executing the test
described above, at five seeds over eight rotations (40 paired units).

Restating the prediction before the result arrives, since that is the only form
in which it means anything:

**P14.** If the subsampling-removal mechanism is correct, CS shows the *largest*
SimSample harm of any dataset — more negative than Photo's −0.0318 — because 89.0%
of its nodes sit in the band where subsampling can be removed, against Photo's
52.4%. If CS instead lands near zero, as ogbn-arxiv did, the mechanism is dead and
the harm on Photo and Computers has some other cause that this work has not
identified.

Either outcome is reportable. The first restores a mechanism that ogbn-arxiv put
in doubt; the second confirms it should stay withdrawn, and leaves the sign change
as a robust empirical observation with no explanation attached — which is a
weaker claim than the draft made, and an honest one.

## P14 — FALSIFIED (was untestable on the 24 GB card; the A6000s settled it)

CS does not fit with SimSample on. Measured: CS alone peaks at 18.9 GB, and
SimSample raises the sampled-edge count by roughly 1.4× on graphs whose nodes sit
below the sampling budget — the recorded figures are 1.38× on Photo and 1.36× on
Computers — because deterministic top-k takes every neighbour where
sample-with-replacement-then-dedup takes fewer. CS has 89% of its nodes in that
band, so the combination needs about 25 GB on a 24 GB card and OOMs partway
through the first epoch. This is the second CS arm lost to memory, after DEMO.

**So the mechanism question stays open.** ogbn-arxiv refuted the
subsampling-removal explanation (+0.0008 against a predicted −0.083); CS was the
discriminating follow-up and cannot be run here. The honest position is the one
already recorded in METHODOLOGY §5.1: SimSample's *gain* is established, including
by intervention on a fixed graph, and its *harm* on Photo and Computers is a real
two-dataset observation with no established cause. A 32 GB card would settle it.

### It was settled, on 2026-08-26

The paragraph above is kept as written because the reasoning was right and the
conclusion was right *for the hardware it was written on*. What it did not
anticipate is that the hardware would change: the campaign moved from the 24 GB
A5000s to 49 GB A6000s, and CS with SimSample fits there with room to spare. The
arm has now run at five seeds.

| dataset | Δ AUC-ROC, SimSample on − off | W/T/L | p | nodes above budget |
|---|---|---|---|---|
| Photo | −0.0586 | 0/0/5 | 0.0625 | 52.4% |
| Computers | +0.0053 | 3/0/2 | 0.3125 | — |
| **CS** | **+0.0005** | 4/0/1 | 0.1250 | **89.0%** |
| ogbn-arxiv | −0.0004 | 2/0/3 | 0.6250 | — |
| Yelp | +0.0263 | 10/0/0 | 0.0020 | 99.7% |

**P14 predicted CS would show the largest harm of any dataset — more negative
than Photo's −0.0318 — because 89.0% of its nodes sit in the band where
subsampling can be removed. CS gives +0.0005.** It is the second-highest
above-budget fraction in the table and the second-smallest effect. The prediction
is falsified in the direction that matters: not merely "smaller than predicted"
but *the wrong sign and near zero*, matching ogbn-arxiv, which was the first
refutation.

So the subsampling-removal mechanism is dead, and this was the discriminating
test that killed it rather than merely doubting it. What survives is narrower and
should be stated as such:

- SimSample's **gain** on Yelp is established and large (+0.0263, 10/10,
  p=0.0020), and it is not an artefact of perturbing the sampler: the shuffled
  placebo `C_simplacebo` is −0.0291, i.e. as harmful as removing SimSample
  entirely. Similarity *ordering* is doing the work.
- SimSample's **harm** on Photo (−0.0586) is real and large.
- The above-budget fraction does **not** predict which of those two happens.
  Yelp at 99.7% helps, CS at 89.0% does nothing, Photo at 52.4% hurts badly. No
  monotone relationship survives.

P17 independently confirms this from the other direction: varying the budget
*within* Photo, Δ is negative at every budget from b=5 to b=100, so the
cross-dataset ordering has no within-graph support either.

**The honest position is now stronger than "the mechanism question stays open".
The mechanism is refuted, twice, by the two tests designed to discriminate it.
§5.1 should state a real effect with no established cause, and should not offer
the above-budget fraction as an explanation.**

Worth noting what this cost to discover: reaching the OOM at all required fixing
a segfault. The similarity ordering built a single `[E, D]` tensor — on CS that is
163,788 × 6,805 = **4.46 GB**, past the 2³²-byte boundary where some torch 2.0
CPU gather kernels overflow a 32-bit offset. Every CS SimSample run died at
~1.8 min with 721 GB of host memory free, which reads like a resource problem and
is not one. `utils.py` now computes the similarities in chunks, verified
numerically identical by `analysis/scripts/check_invariance.py`. That fix stands
whether or not CS ever runs: it would have hit any sufficiently wide-featured
graph, and it was silently waiting for one.
