# Pre-registration: validation-only hyperparameter selection at 400 epochs, and Amazon's first sweep

Written 2026-09-07 04:55 UTC. Honest timing: the campaign (`campaigns/hparam400.json`, 99 arms)
was launched by the chain at 04:21 UTC and 15 Amazon shards existed on disk when this was written;
**none had been opened**. Photo and Computers 400-epoch arms: zero shards at writing time.
Selection criterion, tie band (0.002 mean `val_auc`) and tie-break are exactly those of
`prediction_tuning.md` and `prediction_hparam_all.md`; `select_hparams.py` discards every test
metric before a winner is fixed.

## Why this exists

Threat 10.1-1 in METHODOLOGY.md: validation-only selection was done at 200 epochs and the chosen
configuration was reused at the uniform 400-epoch protocol. A referee can ask whether the 400-epoch
main table is tuned at all. Amazon inherited Yelp's configuration and never had a sweep of its own.

## Arms

Photo: the twelve `T_*` arms of `prediction_tuning.md` re-run as `E400_T_*` at 400 epochs.
Computers: the nine arms of `prediction_hparam_all.md` as `E400_T_*`. Amazon: Yelp's twelve-arm grid
as `T_*` (Amazon is natively 400 epochs). Three selection seeds (0, 1, 42) in every case; the default
arm is not re-run — `E400_outpost` (Photo, Computers) and `A_sim0.0` (Amazon) already contain those seeds.

## Predictions

**P29.** Validation does not discriminate at 400 epochs either: on Photo and Computers every arm
falls inside the 0.002 tie band and the default is retained. Mechanism unchanged from P9/P15 —
validation AUC saturates while test moves.

**P30.** The test-selection bonus at 400 epochs is at least as large as at 200 on Photo
(+0.0158 at 200, so ≥ +0.015 at 400) and on Computers (+0.0047 at 200). Longer training widens the
spread of test outcomes across arms more than it widens validation; the bonus is a property of the
protocol, not of the budget.

**P31.** Amazon: validation retains the inherited default (all arms inside the tie band), and the
test-selection bonus is below 0.005 AUC-ROC — Amazon's seed sd is 0.0048, so no arm can move test by
more than seed noise. A bonus above 0.010 on Amazon would falsify this and would mean the real graph
is more tuning-sensitive than the semi-synthetic ones, the opposite of the paper's reading.

## What would change the paper

If P29 fails (validation picks a non-default arm at 400), the main-table configuration is replaced by
the validation winner at 400 and every OUTPOST 400-epoch number is re-run at that configuration.
If P30 fails (bonus shrinks at 400), the hyperparameter finding is restated as budget-dependent.

## Addendum, 2026-09-07 05:02 UTC — a partial read happened after registration

A pipeline test ran `select_hparams.py` on Amazon while only 7 of its 12 sweep arms had all three
seeds. The selector then lacked a completeness guard, wrote a provisional winner, and the appendix
build printed it: among those 7 arms validation preferred `T_hidden16`, whose test AUC-ROC was
visible in the appendix row. That partial output was seen by the author of this file **after** the
predictions above were written (file created 04:55 UTC; no shard opened before then). The
predictions are unchanged. The selector now refuses to pick a winner until every expected arm is
complete (`--expect-arms`), and the provisional file was deleted; the scored result will come from the
complete 12-arm sweep only.

---

## Verdicts, scored 2026-09-07 after all 99 sweep arms completed

Selection is `select_hparams.py` with the completeness guard: mean `val_auc` over
seeds 0, 1 and 42, validation split only, 0.002 tie band, ties to the default.
Records: `hparam_selection_photo_400.json`, `hparam_selection_computers_400.json`,
`hparam_selection_amazon.json`.

| | arms | val spread | arms in 0.002 band | validation winner | default retained |
|---|---|---|---|---|---|
| Photo @400 | 13 | 0.0016 | **13/13** | `E400_outpost` | yes |
| Computers @400 | 10 | 0.0012 | **10/10** | `E400_outpost` | yes |
| Amazon | 13 | **0.0090** | **1/13** | `T_hidden16` | **no** |

Test metrics, read only after each winner was fixed. **Every arm is scored on the
same three selection seeds**, the default included: averaging a ten-seed default
against three-seed challengers made Computers' 400-epoch gap read 0.0016 against
a seed-matched 0.0042, and the 200-epoch figures this prediction was registered
against (+0.0158 Photo, +0.0047 Computers) carried that same defect. The
seed-matched values replace them on both sides of the comparison.

| | validation picked | its test ROC | best arm on test | its test ROC | selection gap |
|---|---|---|---|---|---|
| Photo @200 | `A_main` | 0.8367 | `T_lr0.003` | 0.8532 | **+0.0166** |
| Photo @400 | `E400_outpost` | 0.8703 | `E400_T_lr0.003` | 0.8867 | **+0.0164** |
| Computers @200 | `A_main` | 0.8193 | `T_lambda_un1.5` | 0.8261 | **+0.0068** |
| Computers @400 | `E400_outpost` | 0.8485 | `E400_T_lambda_un1.5` | 0.8527 | **+0.0042** |
| Amazon @400 | **`T_hidden16`** | 0.9406 | `T_Kp2` | 0.9511 | **+0.0105** |

**P29 — CONFIRMED.** Predicted: at 400 epochs every Photo and Computers arm falls
inside the 0.002 tie band and the default is retained. Measured: 13/13 and 10/10
inside the band, spreads 0.0016 and 0.0012, default retained on both. Validation
is as saturated at 400 epochs as it was at 200, so the main-table configuration
stands and threat 10.1-1 is answered for the semi-synthetic graphs: the 400-epoch
table is not tuned, because there is nothing for validation to tune with.

**P30 — FALSIFIED in its comparative form; the Photo threshold survives.**
Predicted: the bonus at 400 is *at least as large* as at 200, with Photo ≥ +0.015
and Computers ≥ +0.0047. Seed-matched, the bonus **shrinks on both**: Photo
+0.0166 → +0.0164, Computers +0.0068 → +0.0042. Photo still clears the absolute
+0.015 floor that was written down, so that half of the prediction holds, but the
mechanism claimed — *"longer training widens the spread of test outcomes across
arms more than it widens validation"* — is wrong and is withdrawn. The honest
statement is that the test-selection bonus is **budget-invariant to slightly
decreasing**: a referee picking the best of ten to thirteen arms on test buys
roughly +0.004 to +0.017 at either budget.

Note that scoring this against the originally registered 200-epoch numbers
(+0.0158, +0.0047) would have made Photo a confirmation and Computers a near-miss.
The seed-matched recomputation makes the verdict worse for us on both, and it is
the correct comparison, so it is the one scored.

**P31 — FALSIFIED on both clauses.** Two were registered.

*Retention.* Validation does not keep the inherited default. It picks
`T_hidden16` by 0.0057 mean val AUC, with only 1 of 13 arms inside the band and a
validation spread (0.0090) five to seven times Photo's and Computers'. No tie band
below 0.0057 retains the default.

*Bonus.* Registered: below 0.005, with **"a bonus above 0.010 on Amazon would
falsify this"**. Measured: **+0.0105**. It clears the falsifier. An earlier scoring
of this file reported +0.0034 and called the clause confirmed; that number was the
gap from the *default*, not from the arm validation actually picked, and the two
are only the same quantity when the default is retained — which is precisely the
assumption P31's other clause broke. The registered quantity is the gap from the
validation-selected arm, and the corrected value falsifies the clause.

*What the +0.0105 is made of.* It decomposes cleanly, and the decomposition
matters more than the verdict: validation's pick sits **−0.0071** below the
default, and the best arm on test sits **+0.0034** above it. So Amazon is not a
graph where arms are far apart — the spread across arms really is close to seed
noise, as registered. It is a graph where **validation chooses badly**, and
two-thirds of the apparent "tuning on test" advantage is the penalty for following
validation rather than the reward for searching.

**The finding this produces.** *(Superseded 2026-09-08 by P40: the three-seed spread that made Amazon look discriminating is sampling noise — at ten selection seeds the margin between the two arms is -0.0004, inside the tie band. See `analysis/tables/prediction_selection_final.md`.)* Amazon is the one graph in the study whose
validation split *appears* to discriminate between configurations — and it
discriminates in the wrong direction. Where validation is saturated
(semi-synthetic) it is merely useless; where it is informative (real) it was here
actively misleading. That is a sharper form of the paper's selection-rule finding
than the saturation result alone.

**Consequence, executed.** This file's own rule — *"If P29 fails (validation picks
a non-default arm at 400), the main-table configuration is replaced by the
validation winner at 400 and every OUTPOST 400-epoch number is re-run at that
configuration"* — applies to Amazon. `campaigns/sel_final.json` takes
`T_hidden16` to the full ten seeds, with `C_nopl` at the same width so the Amazon
decomposition keeps a matched base. Registered in
`analysis/tables/prediction_selection_final.md` (P32–P34) before launch.
