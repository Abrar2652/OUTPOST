# NSReg under the same validation-only selection rule

Threat 10.1-5 was that NSReg ran only the released `mag_cs` configuration while
OUTPOST had a validation sweep. `campaigns/nsreg_tune.json` closes it with an
lr × weight-decay grid on the four cheap datasets at 400 epochs and three seeds.
The rule was fixed in that campaign's `_comment` before it ran — *"the wrapper
records `val_auc` on OUTPOST's validation split, and selection is on that
alone"* — and is applied by `analysis/scripts/select_nsreg.py` with the same
0.002 tie band and the same tie-break toward the released configuration that
`select_hparams.py` uses for OUTPOST. CS is excluded for cost (77 min per
rotation × 8 rotations × 4 configurations).

Test metrics were read only after each winner was fixed.

| dataset | validation winner | default retained | in 0.002 band | val spread | released oracle | selected oracle | Δ oracle | Δ val-selected |
|---|---|---|---|---|---|---|---|---|
| Photo | `NT_lr0.003_wd0.0` | **no** | 1/4 | 0.0037 | 0.8573 | **0.8679** | **+0.0106** | +0.0118 |
| Computers | `E400_nsreg` | yes | 4/4 | 0.0008 | 0.8339 | 0.8339 | — | — |
| Yelp | `E400_nsreg` | yes | 1/4 | 0.0058 | 0.7488 | 0.7488 | — | — |
| Amazon | `NT_lr0.003_wd0.0` | **no** | 2/4 | 0.0044 | 0.9553 | **0.9600** | **+0.0047** | **+0.0225** |

Three seeds (0, 1, 42); the two datasets whose winner changed are being taken to
ten seeds by `campaigns/sel_final.json`, and the result is pre-registered as P35.

## What this changes

**The baseline gets stronger, and it gets stronger where it already won.** NSReg
was already reproducing +0.04 to +0.10 above its published numbers under
OUTPOST's protocol. Tuning it under the same rule OUTPOST gets adds +0.0106
oracle on Photo and +0.0225 val-selected on Amazon. Every OUTPOST-vs-NSReg
comparison on those two datasets must be recomputed against the tuned arm.

**The asymmetry is the finding.** The identical rule, applied to two methods on
the same graph, moves them in opposite directions. On Amazon it hands NSReg
+0.0047 oracle and costs OUTPOST −0.0071 (`prediction_hparam400.md`, P31). On
Photo and Computers, where validation is saturated for OUTPOST (13/13 and 10/10
arms inside the tie band), it is *not* saturated for NSReg on Photo — one arm of
four sits in the band, spread 0.0037. So "validation saturates on the
semi-synthetic benchmarks" is a statement about a method's loss surface, not
about the benchmark alone, and §5.3 must say which.

**Yelp is the counter-example that keeps the rule honest.** Yelp's spread is the
widest of the four (0.0058) and validation still retains the released
configuration, because the released configuration is genuinely the best of the
four there. A discriminating validation split is not automatically a misleading
one.

## A note on the tie-break

`select_hparams.py` breaks a tie inside the band toward the smaller model first
and the repository default second; `select_nsreg.py` breaks it toward the
released configuration. The difference is vacuous here: NSReg's grid varies only
learning rate and weight decay, so every arm has identical parameter count and
the size rule can never fire. On the two datasets where a tie decided the outcome
(Computers 4/4 in band, Yelp 1/4) the released configuration is what both rules
return. Recorded so the two procedures are not described as identical when they
are only equivalent on this grid.


---

## Ten-seed follow-up (2026-09-08): only one of the two selections survives

The table above is the three-seed selection the rule prescribes. Both changed
arms were then taken to ten seeds (`campaigns/sel_final.json`, pre-registered as
P35). One holds and one does not.

| dataset | val margin @3 | val margin @10 | paired test Δ @10 | seeds | p | outcome |
|---|---|---|---|---|---|---|
| Photo | +0.0037 | **+0.0008** | **−0.0009** | 4/10 | 0.625 | selection does not survive |
| Amazon | +0.0044 | **+0.0057** | **+0.0054** | 9/10 | 0.0059 | selection holds |

**Photo.** The validation margin that fired the selection collapses to inside the
0.002 tie band at ten seeds, and the test gain it promised (+0.0106 at three
seeds) is gone. NSReg's Photo row therefore **stays at its released
configuration**: there is nothing to move it for.

**Amazon.** Stable in both validation and test, and significant. NSReg's Amazon
row **moves to `NT_lr0.003_wd0.0`**, which makes the baseline 0.0054 stronger on
the graph where OUTPOST already does not win. That is a change against this
paper's own method, adopted because the rule and the data both support it.

**The pattern this belongs to.** Three selections in the whole study departed
from a default, all on three-seed margins: OUTPOST on Amazon, NSReg on Photo,
NSReg on Amazon. Two of the three do not survive ten seeds. A three-seed
validation mean carries a standard error near 0.0025 on these datasets, so a
0.002 tie band cannot tell a real difference from noise, and here it did not.
See `prediction_selection_final.md` (P35, P40).
