# Pre-registration: validation-only hyperparameter selection on Photo

Written 2026-08-10, **before** any run in this sweep was launched. Committed
first because a hyperparameter search is the easiest place in a paper to launder
test-set information, and the only defence that survives review is a selection
rule fixed in advance and a record that it was.

## Why this sweep exists

Two open problems, one sweep:

1. `config.json` carries per-dataset values (`K_p`, `drop_out`, `hidden_dim`,
   `warmup_epochs`, `lambda_un`) whose selection history is recorded nowhere in
   the repository. If any of them were chosen against test feedback, the main
   table inherits that, and no amount of seed-averaging fixes it. This sweep
   re-derives them under a rule that is written down.
2. METHODOLOGY §8 says of the existing sensitivity analysis: "With 1–3 runs per
   point against ±0.03–0.07 variance, the `lambda_un` and `alpha_plus` curves are
   indicative only. A publication-grade sensitivity figure requires multi-seed
   sweeps that have not been run." The same runs are that figure.

## Selection criterion — fixed, and it is not a test metric

**Selection maximises the mean `val_auc`**: the AUC-ROC on the *validation*
split (30 held-out anomalies of the seen class + 1% of normals), at the epoch
where that validation AUC peaks, averaged over rotations and over the three
selection seeds.

This is deliberately **not** `valsel_auroc`. That column is a *test* metric read
at the peak-validation epoch — useful for reporting, disqualifying for
selection. `analysis/scripts/select_hparams.py` reads `val_selected.val_auc` from
the per-rotation records and nothing else; it does not load a test metric at any
point, and it refuses to print one before a winner is fixed.

Ties within 0.002 mean `val_auc` go to the configuration with fewer parameters,
then to the repository default. Stated now so the tie-break cannot be chosen
later.

## Grid — one factor at a time from the current default

Twelve arms on Photo, three seeds each (42, 0, 1). The default is not re-run:
the `A_main` rows already are that point at those seeds. One-factor-at-a-time
rather than a full grid because it is affordable, because it is interpretable,
and because each arm doubles as a sensitivity point.

| tag | change from default |
|---|---|
| `T_lambda_un0.5` | `lambda_un` 1.0 → 0.5 |
| `T_lambda_un1.5` | `lambda_un` 1.0 → 1.5 |
| `T_Kp4` | `K_p` 8 → 4 |
| `T_Kp12` | `K_p` 8 → 12 |
| `T_hidden32` | `hidden_dim` 64 → 32 |
| `T_hidden128` | `hidden_dim` 64 → 128 |
| `T_warmup15` | `warmup_epochs` 5 → 15 |
| `T_drop0.3` | `drop_out` 0.5 → 0.3 |
| `T_lean` | `use_mixup`, `use_halo` → false |
| `T_alpha0.01` | `alpha_plus` 0.05 → 0.01 |
| `T_gateq0.8` | `atlas_gate_q` 0.9 → 0.8 |
| `T_lr0.003` | `lr` 0.001 → 0.003 |

## Protocol after selection

1. The winner is fixed by mean `val_auc` over seeds 42, 0, 1.
2. It is then run at the remaining seeds (2, 3) to complete five, and the test
   metrics are read **once**, at that point.
3. Both the tuned and the default configuration are reported. If tuning helps,
   the paper reports the tuned number with this document as its provenance; if
   it does not, the default stands and that is reported too.

## What this cannot fix

Selection on validation removes test leakage from *this* sweep. It does not
retroactively certify the values that were in `config.json` beforehand, whose
history remains unknown — it replaces them with values that have one. And Photo's
binding constraint is not a hyperparameter: class 7 has a same-class neighbour
fraction of 0.556, a hop-0 AUC of 0.530, and falls *below chance* by hop 3. The
detectability law says there is little for any setting of these knobs to recover.
A large gain here would be surprising, and would itself want explaining.

## Prediction

**P9.** Mean test AUC-ROC on Photo under the validation-selected configuration
will improve on the default by **less than 0.02**, and will remain below DEMO's
published 0.9023. If it does not — if validation-only tuning closes a 0.07 gap —
then the gap was never a method difference and the paper's Photo discussion
should say so.

---

# Outcome (recorded 2026-08-11)

## P9 — CONFIRMED, and more strongly than stated

**All twelve arms fall inside the pre-registered 0.002 tie band on validation.**
Mean `val_auc` ranges 0.9898 to 0.9918, with per-arm standard deviations of
0.0006–0.0023: the arms are not distinguishable. The tie-break fires as written —
tie within 0.002, default retained — so the selected configuration **is the
repository default**, and validation-only tuning improves Photo by exactly
**0.000**.

The reason is visible in the numbers: Photo's validation AUC is saturated near
0.99, so the validation split has almost no power to rank configurations.

## The measurement this produced

Reading the test metrics *after* selection was fixed, the same twelve arms span
**0.8281 to 0.8532** AUC-ROC — a spread of **0.0252** — and the default sits
twelfth of thirteen:

| arm | test AUC-ROC | test AUC-PR |
|---|---|---|
| `T_lr0.003` | **0.8532** | 0.5432 |
| `T_lean` (synthesis off) | 0.8531 | 0.5340 |
| `T_hidden128` | 0.8456 | 0.5351 |
| … | … | … |
| `A_main` (default, **selected**) | 0.8310 | 0.5205 |
| `T_alpha0.01` | 0.8281 | 0.5239 |

So: **selecting on test would have bought +0.0222 AUC-ROC on Photo. Selecting
honestly on validation buys nothing.** That difference is the hyperparameter
test-selection bonus, measured directly rather than argued about.

Had the selection rule not been fixed first, the obvious move — pick the best arm
and report it — would have produced a Photo figure of 0.8532 and a claim of a
tuning improvement. Both would have been artefacts. This is the entire reason the
grid and the rule were committed before the runs.

Note also that `T_lean` (anomaly synthesis off) is second-best on test, which
agrees with the Yelp ablation where *adding* synthesis cost −0.032 AUC-PR. It is
still not selectable: validation cannot see the difference.

## Secondary use: this is the sensitivity analysis

METHODOLOGY §8.2 recorded that its sensitivity curves had 1–3 runs per point and
were "indicative only". These twelve arms at three seeds are the multi-seed
version for Photo. The result is that Photo is **insensitive** to every
single-factor perturbation tried: 0.0252 total spread in test AUC-ROC across
learning rate, hidden width, `K_p`, warmup, dropout, `alpha_plus`, gate quantile,
`lambda_un` and synthesis on/off — comparable to the 0.0174 seed standard
deviation of the default cell itself.

## What is still not fixed

This gives the Photo defaults a provenance they did not have, and shows no better
configuration exists nearby by validation. It does **not** retroactively certify
the values that were in `config.json` beforehand, and it does not cover
Computers, CS or Yelp, whose per-dataset values still have no recorded history.
