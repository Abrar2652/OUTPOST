# Pre-registration: the mechanism experiment (P20–P22)

Registered in `campaigns/mechanism.json` on 2026-09-03 before the runs; scored
here from `analysis/appendix.json` on 2026-09-04. Δ is arm minus full model,
AUC-ROC, paired on seed; negative means the component earns its place.

## Why it was run

OUTPOST beats DEMO on Computers (+0.0728, 10/0/0) and CS (+0.0196, 10/0/0) and
the paper could not say why. Two facts motivated the design: on Photo the gate
alone was inert (+0.0003) while gate+PL together cost −0.0382, which looked
like an interaction; and the ablation suite had been run on Photo and Yelp —
where OUTPOST ties or wins least — while Computers and CS had only `C_sim`.

## Predictions, as registered

- **P20.** The redundancy is general: on Computers, `L_lean` (gate+PL off) is
  more negative than either flag alone, *and by more than their sum*. If
  additive, the interaction is a Photo/CS quirk and the mechanism claim dies.
- **P21.** `C_nopl` is more negative than `C_nogate` on both datasets, because
  PL-off also disables the conformal threshold computed inside `if use_pl`.
- **P22.** `C_noconformal` alone stays near zero on both, as on Yelp
  (−0.0002), because conformal acts only through the PL branch that is still on.

## Results

| dataset | kind | gate off | conformal off | PL off | gate+PL off |
|---|---|---|---|---|---|
| Computers | semi-synthetic | -0.0012 (3/2, p=1.000, n=5) | -0.0366 (0/5, p=0.062, n=5) | -0.0645 (0/5, p=0.062, n=5) | -0.0645 (0/5, p=0.062, n=5) |
| CS | semi-synthetic | +0.0000 (1/1, p=1.000, n=5) | -0.0046 (0/5, p=0.062, n=5) | -0.0648 (0/5, p=0.062, n=5) | -0.0648 (0/5, p=0.062, n=5) |
| Photo | semi-synthetic | +0.0003 (2/3, p=1.000, n=5) | — | — | -0.0382 (0/10, p=0.002, n=10) |
| Yelp | real | +0.0005 (4/3, p=0.553, n=10) | -0.0002 (0/1, p=0.317, n=5) | +0.0058 (9/1, p=0.027, n=10) | +0.0058 (8/2, p=0.027, n=10) |
| Amazon | real | — | — | -0.0018 (3/7, p=0.432, n=10) | — |

## Verdicts

**P20 — FALSIFIED.** On Computers `L_lean` = -0.0645 and `C_nopl` = -0.0645;
on CS -0.0648 and -0.0648. The pair equals the PL ablation alone to four
decimals on both datasets. There is no interaction; removing the gate on top
of PL costs nothing. The Photo "interaction" was an artefact of comparing
`L_lean` against a gate ablation while `C_nopl` had never been run on Photo.

**P21 — CONFIRMED.** PL-off costs ~50× more than gate-off on both datasets.

**P22 — SPLIT.** Near zero on CS (-0.0046) and Yelp, as predicted; but
-0.0366 on Computers, which is not near zero. On Computers the conformal
threshold is over half of PL's whole contribution (-0.0366 of -0.0645);
on CS it is almost none of it. The code-level explanation — conformal is nested
inside PL — is right about the structure and wrong about how much it carries,
and that amount differs by dataset.

## What is actually true, and what the paper can say

The atlas gate is inert on every dataset measured. Pseudo-labelling is the
only load-bearing component, and it is load-bearing only on the
semi-synthetic graphs: essential on Computers and CS (≈ −0.065), null on
Amazon, harmful on Yelp. OUTPOST's measured mechanism is *GraphSAGE plus
conformal-thresholded pseudo-labelling*, and its benefit tracks how the
benchmark was constructed rather than anything about anomaly detection. That
is a benchmark finding and should be reported as one.


## Addendum (2026-09-05): the missing Photo cell

`C_nopl` on Photo ran in the gate-drop campaign: **-0.0377** (0/5, p=0.062,
n=5), against `L_lean` -0.0381. Same pattern as Computers and CS: the pair
equals PL alone. P20's falsification now rests on three datasets, not two.
