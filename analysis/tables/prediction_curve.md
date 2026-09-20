# Pre-registration: the SimSample criterion, measured within a single graph

Written 2026-08-21, before any run in this sweep.

## Why

Across datasets the effect of SimSample is **non-monotone** in the fraction of
nodes whose degree exceeds the sampling budget:

| dataset | frac(deg > 25) | Δ AUC-PR |
|---|---|---|
| Yelp | 95.2% | **+0.050** |
| Computers | 45.0% | −0.0131 |
| Photo | 43.8% | −0.0321 |
| CS | 5.1% | +0.0025 |
| ogbn-arxiv | 4.3% | +0.0008 |

Gain at the top, harm in the middle, nothing at the bottom. Three pre-registered
mechanisms for the harm have now been falsified (P11, P13, P14), so the pattern
currently has **no explanation and five points**, each from a different graph
that differs in size, features, homophily and label provenance. A referee is
entitled to say the ordering is confounded with graph identity and the "criterion"
is a curve fitted through five unrelated points.

Photo alone spans the whole range once the budget is varied:

| budget b | 5 | 10 | 25 | 50 | 100 |
|---|---|---|---|---|---|
| frac(deg > b) | 85.1% | 71.6% | **43.8%** | 13.8% | 3.8% |

So the entire cross-dataset range can be reproduced **on one fixed graph**, with
nothing varying except the budget. Same design that settled P10 on Yelp.

## Design

Photo, `sampling_sizes [b, 10]` for b in {5, 10, 50, 100}, each with
`sim_topk_frac` 1.0 and 0.0, five seeds, paired on seed and rotation.
b = 25 is already measured (`A_main` vs `C_sim`, Δ = −0.0321, 0/10).

## Predictions

**P17 (the shape is real, not cross-graph).** Δ(b) on Photo will reproduce the
non-monotone shape: **positive or near-zero at b = 5 and b = 10** (85.1% and
71.6% above budget, the Yelp-like regime), **most negative near b = 25**
(43.8%), and **returning toward zero at b = 100** (3.8%, the CS/arxiv-like
regime, where selection can act on almost nothing).

**P18 (the harm has an interior maximum).** |Δ| at b = 100 will be smaller than
|Δ| at b = 25. If harm instead grows as the budget grows, "there is nothing left
to select" cannot be what makes it vanish on CS and arxiv, and the phenomenology
of §5.1 is wrong in a fourth distinct way.

**P19 (the off-arm is not what moves).** As in P12, the SimSample-**off** arm
will not vary systematically with b in a way that could explain Δ. If it does,
this sweep measures receptive-field size rather than neighbour selection.

## What would falsify the criterion outright

Δ(b) monotone in b, or flat, or negative at b = 5. Any of those means the
cross-dataset ordering was confounded with graph identity, the criterion has no
within-graph support, and §5.1 should be reduced to "SimSample helps on Yelp" —
a much smaller claim, largely anticipated by CARE-GNN and GHRN.

This experiment is designed to be able to produce that outcome. Three of the four
predictions made about this mechanism so far have been falsified; there is no
reason to expect this one is safe.


# Outcome (recorded 2026-09-06, from `analysis/appendix.json`)

Photo, Δ = SimSample on − off, AUC-ROC, n=5 per budget (b=25 is `C_sim` − `A_main`):

| b | Δ | W/L | p |
|---|---|---|---|
| 5 | -0.0600 | 0/5 | 0.062 |
| 10 | -0.0635 | 0/5 | 0.062 |
| 50 | -0.0452 | 0/5 | 0.062 |
| 100 | -0.0548 | 0/5 | 0.062 |
| 25 | -0.0586 | 0/5 | 0.062 |

## P17 — FALSIFIED

Δ is negative at every budget, including b=5 and b=10 where the prediction
required positive or near-zero. The non-monotone cross-dataset shape does not
reproduce within one graph.

## P18 — CONFIRMED in sign only

|Δ| at b=100 (0.0548) is smaller than at b=25 (0.0586), as
predicted — by 0.0038, inside seed noise at n=5. Not
evidence for an interior maximum; P17's failure makes the question moot.

## P19 — CONFIRMED

The SimSample-off arm does not move with b: b=5 0.8306 (n=5), b=10 0.8382 (n=5), b=25 0.8374 (n=10), b=50 0.8315 (n=5), b=100 0.8293 (n=5),
range 0.0089, non-monotone. The sweep measured neighbour selection, not
receptive-field size.
