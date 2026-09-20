# Pre-registration: validation-only hyperparameter selection, Yelp / Computers / CS

Written 2026-08-19, **before** any run in this sweep. Same rule and same reason as
`prediction_tuning.md`, which covered Photo only.

## Why this exists

Yelp carries the paper's headline result, and its configuration —
`hidden_dim: 32`, `sim_topk_frac: 1.0`, synthesis off, `K_p: 3`, `drop_out: 0.2`,
`warmup_epochs: 10` — has **no recorded selection history anywhere in the
repository**. A referee who asks how those were chosen currently gets no answer,
and the default assumption for an unexplained per-dataset configuration is that
it was tuned against the test set. Computers and CS are in the same position.

This does not prove the original values were chosen honestly; nothing can, since
the history is gone. It replaces them with values that have a documented
provenance, and measures how much the choice was worth.

## Selection criterion — fixed, and not a test metric

**Maximise mean `val_auc`**: AUC-ROC on the *validation* split, at the epoch where
validation AUC peaks, averaged over rotations and selection seeds. Explicitly not
`valsel_auroc`, which is a *test* metric read at the peak-validation epoch and
would be test selection wearing a validation label. `select_hparams.py` reads
`val_selected.val_auc` and discards every test metric before returning.

Ties within 0.002 mean `val_auc` go to the configuration with fewer parameters,
then to the repository default. Fixed now so the tie-break cannot be chosen later.

## Grids — one factor at a time from each dataset's current default

Three seeds per arm (42, 0, 1). The default arm is not re-run: the `A_main` rows
are that point at those seeds.

**Yelp** (default `hidden_dim` 32, `sim_topk_frac` 1.0, `K_p` 3, `drop_out` 0.2,
`warmup_epochs` 10, `alpha_plus` 0.05, `lr` 0.001, synthesis off):
`hidden_dim` 16 / 64 · `K_p` 2 / 6 · `drop_out` 0.1 / 0.4 · `warmup_epochs` 5 / 20
· `alpha_plus` 0.01 / 0.10 · `lr` 0.003 · synthesis on

`sim_topk_frac` is deliberately excluded: it is the method's own component, its
value is dictated by the criterion of §5.1 rather than by tuning, and its curve is
already measured at five points as a dose-response. Tuning it here would confuse a
component ablation with a hyperparameter choice.

**Computers** and **CS** (defaults `hidden_dim` 64, `K_p` 8, `drop_out` 0.5,
`warmup_epochs` 5, `lambda_un` 1.0): `hidden_dim` 32 / 128 · `K_p` 4 / 12 ·
`drop_out` 0.3 · `warmup_epochs` 15 · `lambda_un` 0.5 / 1.5 · synthesis off

## Prediction

**P15.** As on Photo, validation will not discriminate: on each dataset the arms
will fall within the 0.002 tie band and the default will be retained, so
validation-only tuning improves nothing. Photo's twelve arms spanned 0.0020 in
mean `val_auc` against 0.0252 in test AUC-ROC — validation saturates while test
moves.

**P16.** The gap between the best arm *on test* and the validation-selected arm
will be non-trivial on at least one dataset — Photo's was +0.0222 AUC-ROC. That
gap is the hyperparameter test-selection bonus, and reporting it is the point:
it measures what tuning on test would have bought had the rule not been fixed.

If P15 fails and validation does pick a better configuration, the paper reports
the tuned number with this document as its provenance. Either outcome closes the
gap; only one of them changes a number.


# Outcome (recorded 2026-09-06, from `analysis/appendix.json` → `hyperparameter_selection`)

## P15 — CONFIRMED

Validation-only selection retained the default (`A_main`) on every dataset:
Photo (14 arms), Computers (11 arms), CS (11 arms), Yelp (14 arms). Validation did not
discriminate, exactly as on Photo.

## P16 — CONFIRMED

Test-selection gap (best arm on test minus the validation-selected arm, AUC-ROC):
Photo +0.0158, Computers +0.0047, CS +0.0010, Yelp +0.0170. Non-trivial on Photo and
Yelp. This is what tuning on test would have bought; it is reported and was not taken.
