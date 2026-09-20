# Reviewer-proofing: what was audited, fixed, and run

Working notes for the ICLR submission. Organised by the objection each item
answers, because that is the only ordering that makes the list checkable.

Status key: **done** · **running** (campaign in flight, resumable) · **open**

## Where the numbers ended up

Five of six datasets complete; ogbn-mag is not runnable on this hardware (§4.1,
`prediction_ogb.md`). Paired = against a DEMO baseline we ran ourselves at
identical seeds, hence identical splits — the only comparison here that admits a
significance test.

| dataset | OUTPOST | paired vs our DEMO | vs best published |
|---|---|---|---|
| **Computers** | 0.8201 / 0.5975 | **+0.052 / +0.089**, 24-1 and 23-2, p ≈ 1e-7 | below |
| **Yelp** (real fraud) | 0.7448 / 0.3824 | **+0.013 / +0.031**, 9-1 both, p < 0.01 | **above, 10/10 seeds** |
| **CS** | 0.9832 / 0.9554 | *(DEMO exceeds 24 GB)* | **above, 5/5 seeds** |
| Photo | 0.8310 / 0.5205 | +0.001 / +0.004, tie | below |
| ogbn-arxiv | 0.6237 / 0.3053 | *(not run)* | inside band; **confirms P6** |
| ogbn-mag | — | — | not runnable, 128 GPU-h/seed |

**Four paired wins, all at p < 0.01; two ties; no paired losses.** Against
published constants: 4 of 10 cells above.

The two framings disagree because our DEMO re-run lands a consistent −0.072 below
its published Photo and Computers numbers (§4.1). Under the published constants
Computers is a loss; under a baseline that actually runs here it is the strongest
win in the table. Both are reported.

---

## 1. "Your method is evaluated on two of the six datasets you claim"

The shipped `analysis/tables/table1_small.csv` and `table2_large.csv` carried the
row `OUTPOST - NOT RUN YET`. `results/reference_runs.csv` held nine runs — Photo,
Computers and Yelp only, from a laptop and a Colab session — and no CS,
ogbn-arxiv or ogbn-mag measurement existed at all.

**running.** All six datasets, five seeds (`campaigns/gpu7.json`,
`campaigns/gpu6_cs.json`), except ogbn-mag at three seeds — 15 rotations × 400
epochs is many GPU-hours per seed and it is the graph where every published
method already sits at chance. Progress at any moment:

```bash
python analysis/scripts/merge_rotations.py --status
```

---

## 2. "You compare against numbers transcribed from another paper"

Every baseline column came from the DEMO paper. No baseline had been re-run, so
no comparison in the table was paired, and the one competitor whose code ships
here (`--method demo`) **could not run at all**: `trainer.train` passed a CUDA
tensor as `node_idx` to a sampler that indexes a CPU tensor with it, so the DEMO
path raised `RuntimeError: indices should be either on cpu or on the same device`
before its first epoch. It also returned `None`, so `main.py` discarded the run
and wrote no row. That is why the baseline was never completed.

**done** — fixed both (`trainer.py`): pass host-side indices to the sampler, and
return the same `{best, val_selected, final}` record OUTPOST returns, so DEMO
reports under both selection protocols and lands in `results.csv`.

**running** — DEMO re-run at the *same seeds*, hence the same splits, on Yelp,
Photo, Computers and CS. That makes the comparison paired on (seed, rotation),
which is what turns "our mean is higher" into "it wins on matched data".

**done** — two fairness bugs caught before the DEMO arm ran, both of which would
have flattered us:

*Epoch budget.* `--method demo` takes DEMO's own hyperparameters, whose
`num_epochs` is 200, while the protocol budget on the large-scale graphs is 400.
Unfixed, DEMO would have received 200 epochs on Yelp against OUTPOST's 400 — and
under best-over-epochs selection more epochs can only help, so the headline
dataset's comparison would have been rigged in our favour. The Yelp DEMO arm now
runs 400 (`campaigns/build_gpu7.py`).

*Evaluation loader.* The OUTPOST arm builds its evaluation sampler once and
reuses it; the DEMO port rebuilt one every epoch. Measured on Yelp that alone was
62 s/epoch for DEMO against 7.6 s for OUTPOST — an 8× "efficiency advantage" that
belongs to the baseline's port, not to either method, and would have gone
straight into the efficiency table. DEMO now caches it too. The change is inert:
the sampler consumes no randomness at construction (`shuffle=False`, all
stochasticity in `__iter__`), and the same configuration returns identical
metrics either way, at 1.7× the speed. It also brings the 5-seed Yelp DEMO arm
from ~34 GPU-hours down to ~20.

What survives after both fixes is a real difference: DEMO is still ~4.7× slower
per epoch on Yelp and carries 34,209 parameters against OUTPOST's 7,361
(**0.215×**, confirming the 0.22× claimed in METHODOLOGY §5 — measured here, on
one machine, rather than counted from an architecture diagram).

### 2.1 The first paired result already contradicts a claim in the paper

Yelp, seed 42, same split for both arms:

| | our re-run DEMO | published DEMO | OUTPOST (same seed) |
|---|---|---|---|
| AUC-ROC | 0.7309 | 0.7097 | 0.7513 |
| AUC-PR | **0.3535** | **0.2238** | 0.3977 |

OUTPOST still wins the paired comparison (+0.020 ROC, +0.044 PR at a matched
split). But **DEMO re-run scores 0.3535 AUC-PR on Yelp against the 0.2238 its
paper reports** — and METHODOLOGY §6 used that 0.2238, "behind a 2023 baseline",
as the retrodiction that the detectability law explains. On our runs DEMO is not
behind NSReg's 0.3029 on Yelp at all; it is ahead of it.

Part of that gap is ours: we give DEMO the protocol's 400-epoch budget rather
than the 200 in its own hyperparameters (§2 above), and under best-over-epochs
selection more epochs can only help. So our DEMO number is the right one for the
*paired* comparison and is **not** directly comparable to the published constant.

Either way the narrative in METHODOLOGY §6 needs revising once the remaining four
seeds land. The law's benchmark-validity argument does not depend on DEMO
specifically collapsing on Yelp — it rests on the same-class-fraction measurement
and on ogbn-mag, where every published method really is at chance — but the
sentence that leans on 0.2238 cannot stand if our own re-run cannot reproduce it.
This is exactly what re-running a baseline is for, and it is the reason a
transcribed number should never carry a claim on its own.

### 2.3 The paired comparison cannot cover CS on this hardware

Both methods retain the autograd graph over the whole unlabelled pool in order to
take one full-batch step per epoch. On CS that pool is ~17,000 nodes with
6805-dimensional features. Measured allocated peaks: OUTPOST **18.9 GB**, which
fits a 24 GB A5000; DEMO ~**22 GB**, which does not — it fails partway through
the first unlabelled pass.

That is after both memory fixes to the DEMO port (cached evaluation loader,
no-grad weak view) and after tightening the allocator with
`max_split_size_mb:128`, which recovered about 5 GB of fragmentation and still
left it short. A 32 GB card would run it unchanged; no code change short of
altering the baseline's training loop will.

So **CS uses the transcribed DEMO constant**, with the weaker form of evidence
that implies, and the paired comparison covers Yelp, Photo and Computers. The
GPU-hours freed went to CS seeds 2 and 3, so the CS main-table cell is five
seeds rather than three. This is a stated limit, not a silent omission — but
METHODOLOGY §4 still needs the same note written into it, which I have not done.

A useful by-product: DEMO's weak view was being computed **with** gradients even
though `consistency_loss` uses it only for the hard pseudo-label (an integer
comparison) and the confidence mask (`.ge()`/`.le()`), taking the loss on the
strong view alone. Its autograd graph could not affect any gradient and was pure
cost. Now under `no_grad`, behind the `demo_weak_nograd` flag so the claim is
testable — `check_invariance.py --method demo` runs it both ways and confirms
identical metrics.

---

## 2.2 The Photo headline does not survive five seeds

`analysis/tables/published_baselines.csv` carries an `OUTPOST` row reading Photo
**0.9066 / 0.6295**. Five seeds, varying split and initialisation, give:

| | archived headline | **5 seeds** | published DEMO |
|---|---|---|---|
| AUC-ROC | 0.9066 | **0.8310 ± 0.0174** | 0.9023 |
| AUC-PR | 0.6295 | **0.5205 ± 0.0108** | 0.6330 |

Not one of the five seeds reaches the archived figure, and none reaches DEMO
(0/5 on both metrics, bootstrap CI excluding it). **On Photo, OUTPOST is behind
DEMO by ~0.07 AUC-ROC and ~0.11 AUC-PR.**

This is not a surprise so much as a confirmation of something METHODOLOGY §8
already warned about and §9 half-admitted: the archived Photo number came from a
single favourable trajectory draw (`legacy/reproducibility.csv` labels it "favourable
trajectory draw; matched-seed mean is 0.8640/0.5706"), and the archived Photo
deviations vary `train_seed` at a *fixed split*, so they capture initialisation
variance only and understate the real spread. Varying the split as well pulls the
mean down further still, to 0.8310.

The main-table pipeline reads `results/results.csv` and will report 0.8310 ±
0.0174, so the 0.9066 cannot reach a paper table by accident. But the claim in
METHODOLOGY §9 that OUTPOST is competitive on Photo has to go, and the paper
should say plainly that OUTPOST trades Photo/Computers performance for Yelp
performance — which is, in fact, exactly what the detectability law predicts it
should do, and a cleaner story than parity everywhere.

The paired Photo comparison against our own DEMO run is still in the queue and is
the number that should actually appear in the text.

---

## 3. "Three seeds against ±0.03–0.07 variance proves nothing"

**running** — five seeds (42, 0, 1, 2, 3) per cell, and `--seed` varies the split
*and* the initialisation, so the reported spread is the wider of the two
available variance notions rather than the flattering one.

**done** — `analysis/scripts/stats.py` distinguishes three comparisons that do
not admit the same test, and labels each with its strength:

| comparison | unit | test |
|---|---|---|
| a table cell | seed | mean, sd (ddof=1), bootstrap + Student-t 95% intervals |
| vs a baseline we ran | (seed, rotation), same split | Wilcoxon signed-rank, paired t, bootstrap CI on the difference, Cohen's dz, rank-biserial |
| vs a published constant | seed | one-sample t, count of seeds individually above — *labelled weak* |

Ablation arms are corrected within each dataset with Holm–Bonferroni; raw and
adjusted p-values are both printed.

**done** — per-rotation records (`results/rotations/*.json`) are now written as
each rotation finishes. The mean over rotations is the number the paper reports
but is also the number that cannot be tested; the paired tests need the
individual rotations, which the mean destroys.

**done** — **Yelp runs ten seeds, not five, and the reason is arithmetic rather
than a result.** The pairing unit is (seed, rotation). Yelp is binary and has one
rotation, so its unit count *is* its seed count, and at n=5 the smallest p-value
Wilcoxon can return is 2⁻⁴ = 0.0625 — even when all five pairs favour the same
arm, as they do for AUC-PR. Significance was unreachable by construction. n=10
puts the floor at 0.002. Fixed at ten and committed in
`campaigns/build_gpu7.py` before the extra runs were inspected; this is a power
fix, not a licence to add seeds until a threshold is crossed. Every other dataset
has 2–8 rotations and already reaches 10–40 pairs at five seeds.

### 3.1 Paired result, Yelp, ten seeds — complete

| metric | OUTPOST − DEMO | W/T/L | Wilcoxon | paired t |
|---|---|---|---|---|
| AUC-PR (oracle) | **+0.0308** | 9/0/1 | **0.0039** | 0.0008 |
| AUC-ROC (oracle) | **+0.0134** | 9/0/1 | **0.0059** | 0.0021 |
| AUC-PR (val-selected) | **+0.0272** | 8/0/2 | **0.0098** | 0.0047 |
| AUC-ROC (val-selected) | **+0.0154** | 9/0/1 | **0.0039** | 0.0007 |

Significant on both metrics under both selection protocols, on shared splits,
by the nonparametric test. At five seeds AUC-ROC had looked like a 4–1 tie; the
extra five resolved it to 9–1 and moved the point estimate up slightly. That is
the argument for sizing n by the test's floor rather than by convention.

Unpaired, against the published constants, all ten seeds clear the best
published method: AUC-ROC 0.7448 vs DEMO 0.7097, AUC-PR 0.3824 vs NSReg 0.3029,
10/10 on each, bootstrap CI excluding the constant.

---

## 3.2 The Yelp ablations, five seeds: SimSample is the method

Paired on seed against the full system (`A_main`, same seeds), Δ AUC-PR:

| arm | Δ | W/T/L | bootstrap CI on Δ |
|---|---|---|---|
| `C_simplacebo` deterministic, *random* order | **−0.0534** | 0/0/5 | [−0.0616, −0.0453] |
| `C_nosim` SimSample off | **−0.0497** | 0/0/5 | [−0.0598, −0.0408] |
| `C_synth` add anomaly synthesis | −0.0318 | 1/0/4 | [−0.0702, +0.0005] |
| `C_hopmix` add HopMix fusion | −0.0116 | 1/0/4 | [−0.0255, −0.0007] |
| `C_noconformal` fixed 0.95 threshold | −0.0028 | 0/1/4 | [−0.0045, −0.0010] |
| `C_nogate` atlas gate off | **+0.0003** | 3/0/2 | [−0.0010, +0.0017] |
| `C_fview` add spectral gate | +0.0035 | 3/0/2 | [−0.0011, +0.0090] |
| `C_nopl` pseudo-labelling off | **+0.0060** | 3/0/2 | [−0.0012, +0.0140] |

**The placebo control replicates cleanly.** Turning SimSample off costs 0.0497;
replacing it with *deterministic sampling in random order* costs 0.0534 — the
same, within noise. The gain therefore comes from similarity ordering, not from
having removed sampling randomness. That is the causal claim of METHODOLOGY §7.3,
and it survives five seeds with an interval nowhere near zero.

**Two of the components listed as "in" do nothing.** The atlas gate is inert
(+0.0003, CI [−0.0010, +0.0017] — a tight null), and removing pseudo-labelling
is if anything an improvement (+0.0060). METHODOLOGY §8 records pseudo-labelling
as worth −0.016, i.e. that it helps; at five matched seeds the sign is reversed.
Only SimSample and, marginally, the conformal threshold (−0.0028, matching the
recorded −0.003) do measurable work on Yelp.

That is a smaller method than the paper currently describes, and it should be
described that way: **OUTPOST's Yelp result is SimSample**, on a backbone whose
other pieces are neutral there.

Every p-value above sits at the n=5 Wilcoxon floor of 0.0625 and nothing survives
Holm correction — the same power problem as §3, for the same reason. `C_nosim`,
`C_simplacebo`, `C_nogate` and `C_nopl` are being extended to ten seeds: two
because they carry the causal claim, two because they carry a *null*, and a null
needs more power to assert than an effect does.

---

## 4. "Your baseline is crippled — you disabled its energy term"

Recorded honestly in METHODOLOGY §4, but as an assertion. `compute_beta` takes a
`batch_size=1` loader over the training set and does a `create_graph` backward
per node per epoch, so its cost scales with labelled training nodes × epochs.

**done, and the objection is closed.** DEMO run to completion on Photo with the
term ON and OFF, same seeds (`E_demo_energy` / `E_demo_noenergy`, band 4 of the
GPU-7 queue):

| DEMO on Photo, 3 matched seeds | AUC-ROC | AUC-PR | median s / rotation |
|---|---|---|---|
| energy term **ON** | 0.8408 ± 0.0321 | 0.5236 ± 0.0286 | **2196** |
| energy term **OFF** | 0.8395 ± 0.0231 | 0.5235 ± 0.0214 | **337** |
| paired difference | +0.0013 (p = 0.83) | +0.0001 (p = 0.99) | **6.5×** |

**The term costs 6.5× the compute and changes nothing measurable.** Running the
baseline without it was therefore not a handicap. The assertion in METHODOLOGY §4
that the component was omitted for cost is now a measurement rather than a claim.

*Correction.* An earlier version of this table read 0.8225 ± 0.0064 for the ON
arm and concluded the term makes DEMO "slightly worse". That was written at n=2
of 3 seeds; the third seed moved the mean to 0.8408 and the per-seed differences
turn out to be of both signs (−0.0037, −0.0043, +0.0119). The correct claim is
*no measurable effect*, not a negative one. The stale figure was caught by
`analysis/scripts/audit_claims.py`, which is what that script is for.

### 4.1 What this does NOT explain

The energy term was the leading candidate for the Photo gap, and it is refuted.
Our DEMO reaches 0.8395 with the term off and 0.8408 with it on; published DEMO
reports **0.9023**. Two of our own implementations — OUTPOST at 0.8310 and DEMO
at 0.8300–0.8408 — sit ~0.06–0.07 below that constant no matter how the baseline
is configured.

Separately, and reassuringly, **our Photo numbers are consistent with this
repository's own archived runs once the archive's honest mean is used**. The
archived seed-42 draws are 0.9089 (`stream-42`), 0.8407 (`train_seed=0`) and
0.8424 (`train_seed=1`); `legacy/reproducibility.csv` already labels the first a
"favourable trajectory draw" with a "matched-seed mean of 0.8640". Our seed-42
run gives 0.8555, inside that family. So the reproduction is sound — what was
never sound was the 0.9066 headline in `published_baselines.csv`, which was built
on the outlier draw.

That leaves one open discrepancy, and it is with the DEMO paper rather than with
anything here: **we cannot reproduce published DEMO on Photo**, with or without
its energy term. It may be a protocol difference we have not identified. It
should be reported as an unresolved reproduction gap, not asserted as an error in
their work — but it also means the Photo column of the baseline table should not
be treated as settled ground truth.

---

## 5. "ρ = 0.897 over 16 classes from 4 graphs is not a law"

Three things were missing, and the shipped script could not run: it `chdir`s two
levels up from `analysis/scripts/`, landing in `analysis/`, where `data/` does
not exist. The command in METHODOLOGY §11 failed on a clean checkout.

**done** — fixed the path, added both OGB graphs (`--dataset all+ogb`), and wrote
`analysis/scripts/detectability_law.py`, which reports:

| | n=16, 4 graphs (original) | n=35, 6 graphs (now) |
|---|---|---|
| Spearman ρ | +0.897 | **+0.803** |
| within-dataset permutation p | — | **5e-5** (20,000 shuffles) |
| cluster bootstrap 95% CI on ρ | [+0.500, +1.000] | **[+0.607, +0.921]** |
| leave-one-dataset-out MAE | — | 0.10–0.22 AUC |

The permutation test shuffles *within* each graph, so the null is "same-class
fraction carries no between-class information beyond what the graph itself
explains". The bootstrap resamples *graphs*, not classes, because the unit that
could have come out differently is the benchmark. The original figure reproduces
exactly under `--no-ogb`; adding two graphs lowers ρ and halves the interval.

**done, and it cost us a claim.** With ogbn-mag in the sample, "real fraud's
0.160 is below every semi-synthetic class (0.55–1.00)" is **false** — 14 of 34
semi-synthetic classes now sit below it, and 11 of ogbn-mag's 15 anomaly classes
have a median same-class fraction of exactly 0.000. The corrected statement is
stronger, not weaker: the graphs the field ranks methods on are the homophilous
ones, and *both* graphs whose classes are scattered are where every published
method collapses to chance — Yelp's real fraud and ogbn-mag. ogbn-mag was not in
the sample the law was fitted on, so this is out-of-sample confirmation of the
law's negative prediction. Rewritten in METHODOLOGY §6.1.

---

## 6. "You found the pattern after seeing the results"

**done** — `analysis/tables/prediction_ogb.md`, written from the training-free
diagnostic *before* OUTPOST was trained on either OGB graph, states P5–P8 and
names what would falsify each. P5 forbids our own method from doing well on
ogbn-mag; a large OUTPOST gain there refutes the law rather than flattering it.

---

## 7. "How do I know your infrastructure changes did not move the numbers?"

Two changes alter how a run executes: splitting rotations across processes
(`--rotations`, needed because ogbn-mag's 15 rotations are ~25 GPU-hours
serially) and holding the feature table on the GPU (`features_on_gpu`, worth
1.9× on CS, whose 6805-dimensional features otherwise spend the run crossing the
PCIe bus).

A third alters the DEMO baseline: its weak view now runs under `no_grad`
(`demo_weak_nograd`).

**done** — `analysis/scripts/check_invariance.py` runs each configuration both
ways and requires every per-rotation metric to match *exactly*. All three pass:

```
features_on_gpu:    identical over 2 rotation(s)
rotation sharding:  identical over 2 rotation(s)
demo_weak_nograd:   identical over 2 rotation(s)   [--method demo]
```

Sharding is also seed-correct by construction: each rotation re-seeds from
`--seed` before building its split, so a rotation's split and initialisation are
a function of (seed, rotation) alone and cannot depend on execution order.

---

## 8. "Which runs produced this table?"

**done** — `results/results.csv` accumulates every run of every campaign, so
main-table cells now filter on the `A_main` tag. Without that filter an ablation
arm — by construction a different configuration of the same dataset — would be
averaged into the score of the method it ablates. `make_paper_tables.py` and
`stats.py` both filter, and both drop duplicate rows left by an interrupted and
resumed campaign, which would otherwise shrink a cell's reported deviation.

**done** — `results/environment.json` records library versions, driver, GPU,
protocol constants and a content hash of every source file that can change a
number.

---

## 9. "Is this transductive? Are you pseudo-labelling the test set?"

Yes, and yes. **done** — stated plainly in METHODOLOGY §3.1 rather than left to
be discovered: the unlabelled pool is `idx_test['all']`, inherited from DEMO,
which is the standard transductive protocol here — the whole graph is present at
training time and what is withheld is labels, not nodes. No test label is read;
the splits are disjoint by construction; both methods use the identical pool. The
numbers are transductive and are not a claim about inductive generalisation.

---

## 10. "Your sensitivity analysis has one to three runs per point"

METHODOLOGY §8 said so itself: "A publication-grade sensitivity figure requires
multi-seed sweeps that have not been run."

**open, campaign written** — `campaigns/sensitivity.json`: `hidden_dim` and
`alpha_plus` on Yelp, `lambda_un` on Photo, five seeds per point. Each default
value is not re-run — the `A_main` rows already are that point at those seeds,
and `stats.py` joins them in.

---

## Still open

- **Hyperparameter provenance.** `config.json` carries per-dataset values
  (`K_p`, `drop_out`, `hidden_dim`, `warmup_epochs`, `lambda_un`) whose selection
  history is not recorded anywhere in the repository. If any were chosen against
  test-set feedback, the main table inherits that. The multi-seed sensitivity
  sweep bounds how much it could matter; it cannot reconstruct what was done.
  This needs an explicit statement in the paper from whoever tuned them.
- **ogbn-mag at three seeds, not five**, and **CS DEMO at three seeds** — stated
  compute limits, not silent omissions.
- **Per-metric oracle selection.** `best_auroc` and `best_aupr` may come from
  different epochs. This is what the published baselines do, so the column is
  comparable, but it is more generous than a single-epoch oracle and the
  `valsel_*` column is the one to read as deployable.

---

## 11. An unplanned finding: the oracle protocol is not dataset-neutral

Recording both selection columns for every run was defensive bookkeeping. It
turned up something worth a section of the paper.

| dataset | n | oracle − val-selected, AUC-ROC | AUC-PR |
|---|---|---|---|
| **Yelp** (real fraud) | 10 | **+0.0023** | +0.0119 |
| CS | 5 | +0.0634 | +0.1261 |
| Photo | 5 | +0.0685 | +0.0392 |
| Computers | 2 | +0.0747 | +0.1060 |

Choosing the best epoch on the test set is worth **essentially nothing on Yelp
and 0.06–0.13 on the semi-synthetic graphs**. The quantity is trajectory
variance: where training oscillates a per-epoch maximum harvests the peaks, and
where it converges there is nothing to harvest. Yelp converges.

Every published baseline is reported under that protocol. So the semi-synthetic
columns of the benchmark table carry an inflation that the real-fraud column does
not — the datasets are not on a common footing under the field's own reporting
convention. Confirming the size of it for *other* methods needs their per-epoch
trajectories, which are not published; what is established here is the effect for
our method on shared splits, and the mechanism is generic to the backbone family
they all share.

This is independent of the topological argument in §5 and points the same way. It
also sharpens the Photo discussion: OUTPOST is behind published DEMO by ~0.07
under the oracle column, which is about the size of the oracle bonus itself on
that dataset.

---

## 12. A crash I introduced, and what it says about the invariance checks

The non-mutating graph wrapper of §7 — added so that `features_to_device` could
not leak one rotation's device-resident features into the next — introduced a
latent fault of its own. `DeviceFeatures.__getattr__` was written as:

```python
def __getattr__(self, name):
    return getattr(self._x, name)     # wrong
```

`self._x` inside `__getattr__` is the standard Python footgun. When `_x` is
present the lookup succeeds and nothing is wrong; when it is ever missing — an
`__init__` that raised partway, a copy, an unpickle — the lookup re-enters
`__getattr__`, and Python does not always convert that into a `RecursionError`.
Deep enough C-level attribute recursion overruns the C stack and the process
dies with **SIGSEGV and no traceback**.

Symptomatically this was maddening: CS runs dying at 0–2 epochs with 721 GB of
host memory free and GPUs idle, siblings on the same card surviving to epoch 29,
one Photo seed lost the same way. Every signal pointed at hardware or resource
pressure. It was a five-line class.

Fixed by reading the attribute through `object.__getattribute__`, which turns the
recursion into a clean `AttributeError`. Verified two ways: a unit check that an
object with no `_x` now raises rather than crashing, and the rotation that had
failed twice completing 20 epochs. The invariance checks still pass, so no number
moved.

**The part worth keeping.** The invariance suite passed throughout — as it did
during the earlier SimSample mutation bug (§7). Both times it was testing that
results *agree between two configurations*, which says nothing about a fault that
kills the process before either configuration produces a result. Equality checks
verify agreement, not liveness, and a crash is invisible to them. The campaign
runner's failure log was the only thing that caught either bug, which is an
argument for treating run-level failure rates as a first-class signal rather than
noise to be retried away — and a caution about the one-retry mechanism added
alongside it, which would have masked this had the rate been lower.

---

## 13. The campaign moved between machines, and the environment record hid it

**This is the most serious reproducibility problem found in this work, and the
infrastructure built to catch exactly this kind of thing did not catch it.**

The campaign **moved to a different machine** partway through:

| | before | after |
|---|---|---|
| host | ckg10 | **ckg12** |
| GPU | RTX A5000, 24.5 GB | **RTX A6000, 49.1 GB** |
| driver | 580.173.02 | **535.309.01** |
| torch | 2.0.1+cu117 | **2.7.1+cu126** |

I first noticed only the torch difference and wrote this section up as a library
upgrade. It is a machine change; the torch, CUDA, driver and GPU all differ
together. The site-packages directory is dated 2026-08-11 and the CUDA
error-message format changed at the same time (`torch.cuda.OutOfMemoryError`
before, `torch.OutOfMemoryError` after), which is the trace that gave it away —
but the GPU model went unnoticed for days longer, because every scheduling
decision was made against a 24 GB budget that had silently become 49 GB.

**This makes the confound broader than a library version.** Results before and
after the move differ in GPU architecture as well as in torch and CUDA, so
floating-point behaviour is not guaranteed identical even at fixed seeds.

`results/environment.json` did not preserve the evidence. It is regenerated by
`refresh_all.sh`, and it *overwrote itself* — so the record of the environment
the first 141 runs executed in was replaced by the environment the last 106 ran
in. A snapshot that overwrites itself cannot tell you the environment changed;
that is the whole failure. Fixed: `record_environment.py` now also appends to
`results/environment_history.jsonl` whenever the versions differ from the last
entry, so the record is a history rather than a photograph.

### Which results are affected

141 runs pre-upgrade, 106 post. Six cells contain seeds from **both** eras:

| cell | pre | post | pre AUC-ROC | post | diff | Mann–Whitney p |
|---|---|---|---|---|---|---|
| photo / outpost / A_main | 5 | 5 | 0.8310 | 0.8442 | +0.0132 | 0.69 |
| computers / outpost / A_main | 2 | 3 | 0.8277 | 0.8149 | −0.0128 | 0.80 |
| computers / demo / B_demo | 2 | 3 | 0.7715 | 0.7650 | −0.0065 | 1.00 |
| cs / outpost / A_main | 4 | 1 | 0.9834 | 0.9823 | −0.0011 | 1.00 |
| photo / demo / E_demo_energy | 2 | 1 | 0.8225 | 0.8775 | +0.0550 | 0.67 |
| photo / outpost / T_hidden32 | 1 | 2 | 0.8327 | 0.8303 | −0.0024 | 1.00 |

No cell shows a detectable shift. But the per-cell n is 1–5 a side, so this
establishes only that the upgrade did not produce an effect large enough to see
at that power — not that it produced none. **Photo's main cell and both arms of
the Computers paired comparison are among the affected cells**, and the Computers
result is the strongest paired win in the paper.

The Yelp results, which carry the headline claim, are entirely pre-upgrade and
unaffected. So is the SimSample budget sweep.

### What should be said in the paper

That the campaign spanned a torch major-version upgrade on shared infrastructure;
which cells span it; that no shift was detectable at the available power; and that
the honest remedy is to re-run the six affected cells under one version. That
re-run has not been done here. It is the single most valuable outstanding piece
of compute in this project — more valuable than any additional dataset — because
it removes a confound from the two cells that currently carry a headline number.

### A consequence that cost hours

`torch_scatter` and `torch_sparse` remain compiled against 2.0.1 and now fail to
load with `undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSsb`. PyG catches
that ImportError and falls back to pure-torch scatter — but only after `dlopen`
has already mapped the broken objects into the process. That is a strong
candidate for the intermittent `SIGSEGV`s that killed CS runs mid-training with
no traceback, ample free memory, and no reproducible epoch count: a fault in
native code, in a thread `faulthandler` cannot attribute.

`.stubs/` contains modules that raise `ImportError` before any native code loads,
and jobs run with it first on `PYTHONPATH`. This is isolated to these runs and
changes no installed package.

**Confirmed.** With the stubs in place a full 200-epoch CS rotation completed in
15.2 minutes at 10.94 GB — the first CS lean rotation to finish at all. Every
prior attempt died between epochs 2 and 30, at wall-clock times from 0.7 to 4.2
minutes. Same seed, same rotation, same configuration; the only change is that
two broken native libraries are no longer loaded into the process.

The diagnostic road to that was long and the wrong turns are worth recording,
because each looked convincing:

1. *Hardware or resource pressure.* Ruled out: 721 GB host RAM free, GPUs idle,
   sibling jobs on the same card unaffected.
2. *A recursion trap in `DeviceFeatures.__getattr__`* (§12). A real bug, correctly
   fixed, and **not the cause** — runs kept crashing afterwards. It was claimed as
   the fix on the strength of one 20-epoch survival, which was not evidence: an
   earlier run had survived 29 epochs before the fix existed. Over-claimed and
   retracted the same session.
3. *A deterministic fault*, inferred from two crashes at matching wall-clock
   times. Also wrong — the crash epoch varied from 2 to 30 across runs, and the
   matching timings were coincidence.

### The actual cause: an allocator override that only became fatal after the upgrade

`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`, which the campaign runner set to
cut allocator fragmentation. Controlled A/B — same dataset, seed, rotation and
configuration, run to completion, only this variable differing:

| arm | result |
|---|---|
| campaign env, **with** `max_split_size_mb:512` | **SIGSEGV at 20 epochs** (rc=139) |
| identical, **without** it | **200 epochs, rc=0**, 15 min |

It fits everything retrospectively. Under torch 2.0.1 the setting was harmless:
every pre-upgrade campaign ran with it, thousands of epochs, and never crashed
once. The crashes begin only after the environment moved to 2.7.1+cu126 (§13) —
including the Photo seed lost on 2026-08-11, the day of the upgrade. The one run
that had ever completed a CS rotation was a manual command that happened not to
set it.

Removed from the runner, with a note not to reintroduce it without re-running
that A/B. No modern equivalent (`expandable_segments:True`) has been substituted:
after four wrong diagnoses, adding another allocator setting on the strength of a
guess would be repeating the mistake.

**No completed result is affected.** The failure mode is a crash, not silent
corruption — a run either died and produced nothing, or finished and is valid.

### The four wrong diagnoses, in order

Recorded because the pattern is more instructive than the answer:

1. **Hardware or resource pressure** — ruled out, but only after wasted GPU hours.
2. **A recursion trap in `DeviceFeatures.__getattr__`** (§12) — a real latent bug,
   correctly fixed, *not the cause*. Declared fixed on a single 20-epoch survival,
   when an earlier run had survived 29 epochs without the fix.
3. **A deterministic fault**, inferred from two crashes at matching wall-clock
   times — wrong; the crash epoch ranged from 2 to 30 and the match was chance.
4. **Broken `torch_scatter`/`torch_sparse`** loaded by `dlopen` before PyG's
   guarded import could fail — a genuine problem in the environment, verifiably
   stubbed out, and *still not the cause*.

Three of the four were real defects worth fixing, which is exactly why each was
convincing. The lesson is not "look harder" but "do not announce a fix from a
single successful run when the failure is intermittent" — a controlled A/B run to
completion settled in 20 minutes what four plausible mechanisms could not.

## 14. Photo changes sign between the two selection rules

This is the single most surprising number in the final results, and as of the
n=10 campaign it appears nowhere in the paper.

Paired OUTPOST − DEMO on Photo, ten seeds, both arms at 400 epochs, same runs
scored two ways:

| selection rule | Δ AUC-ROC | W/T/L | p |
|---|---|---|---|
| oracle (per-metric max over epochs) | **−0.0176** | 3/0/7 | 0.0645 |
| val-selected (peak-validation epoch) | **+0.0170** | 7/0/3 | 0.1934 |

Neither is significant, and the honest summary is "Photo is a tie either way".
But the *direction* reverses, and it reverses because of a protocol choice that
the paper — like every paper in this line — currently makes silently.

The mechanism is visible in §11's table and is now much sharper with the full
seed counts. Oracle inflation, OUTPOST, at 400 epochs:

| dataset | oracle | val-selected | inflation |
|---|---|---|---|
| Photo | 0.8703 | 0.7914 | +0.0789 |
| Computers | 0.8510 | 0.7715 | +0.0795 |
| CS | 0.9842 | 0.9389 | +0.0453 |
| ogbn-mag | 0.5923 | 0.5708 | +0.0215 |
| ogbn-arxiv | 0.6237 | 0.6048 | +0.0188 |
| **Yelp** | **0.7448** | **0.7425** | **+0.0023** |

The three small semi-synthetic graphs inflate by 0.045–0.08. Yelp — a real
anomaly-detection dataset with real anomalies — inflates by 0.0023, roughly
**34× less than Photo**. Oracle selection is not a uniform tax that cancels in a
paired comparison. It is a tax concentrated on exactly the benchmarks the field
reports most, and DEMO on Photo collects more of it than OUTPOST does.

### Why this belongs in the paper rather than in a rebuttal

A reviewer who asks "why report best-over-epochs at all?" is asking the right
question, and the answer is not flattering to anyone: we report it because every
published baseline did, and comparing our val-selected number against their
oracle number would understate them. Both columns are therefore in the main
table. But the Photo reversal shows the choice is not cosmetic — it decides the
sign of a headline cell.

The claim to make is narrow and defensible:

- On Photo the two methods are **tied**, and which one appears ahead depends on
  the selection rule. State it as a tie and show both columns.
- The **gap between the columns** is itself a finding about the benchmarks:
  small semi-synthetic graphs give oracle selection far more room than real ones.
  That is a claim about how this field measures, and it does not depend on
  OUTPOST winning anything.
- The three datasets where OUTPOST wins — Computers +0.0728, CS +0.0196, Yelp
  +0.0138 — win under **both** rules, all at n=10, all p ≤ 0.0195. Those results
  do not depend on the protocol choice at all, which is the strongest thing that
  can be said about them.

### The risk of not saying it

If we report only the oracle column, Photo is a loss and we look like we are
absorbing it quietly. If we report only the val-selected column, Photo is a win
and we look like we picked the protocol that flattered us — and a reviewer who
recomputes under the standard protocol finds the loss. Reporting both, and
naming the reversal, is the only version that survives someone checking.


## 15. "Does the detectability law predict anything out of sample?"

It did not, in the form §6 reported it — and this is known because the test
was registered before the run rather than found afterwards.

Amazon-Fraud was added as a seventh graph on 2026-09-03. Its training-free
diagnostic was computed and two predictions written down
(`analysis/tables/prediction_amazon.md`, P23/P24) before any model trained on
it. The law fitted on the 35 classes of the other six graphs is
`best_auc = 0.5904 + 0.3084 × same_frac`; Amazon's same-class fraction is
0.080, so it predicts **0.615**, 95% band 0.382–0.848.

| | AUC-ROC |
|---|---|
| P23, one-regime law | 0.615 (band to 0.848) |
| Amazon diagnostic ceiling | 0.878 |
| **Amazon, trained OUTPOST, n=10** | **0.9534 ± 0.0048** |

P23 is falsified by 0.105 above the band
ceiling. What went wrong is identifiable and training-free: Amazon's anomalies
are separable from raw features (hop-0 AUC 0.878) and propagation *hurts*
(smoothing gain −0.174). §6 had already named that regime — CS, ogbn-mag class
263 — as an aside; the fitted law ignored it.

### The restatement

Split the 36 classes on the sign of the smoothing gain, itself
computed without training:

- **Propagation regime** (gain > 0, n=21): the
  same-class fraction predicts the ceiling, ρ = +0.896,
  residual sd 0.101. Pooled over all classes ρ is only
  +0.778.
- **Feature-visible regime** (gain ≤ 0, n=15): the
  ceiling is hop-0 by construction and the same-class fraction does not govern
  it. The one-line law's mean absolute error here is
  0.112, against
  0.072 on the propagation regime.

### What must be said, and what must not

1. "Best achievable AUC" in §6 is the *prototype instrument's* ceiling, not a
   trained model's. On Amazon the trained detector beat the instrument by
   +0.075. The paper must not let a reader take
   the law as a bound on learned detectors.
2. The two-regime form was written *after* Amazon forced it. It is a
   restatement the test demanded, not a prediction the test confirmed; the
   propagation-regime fit is within-sample until an eighth graph tests it.
   The honest line: one pre-registered out-of-sample test, one falsification,
   one restatement — and the restatement's own test is still owed.
3. "Best ≈ hop-0" in the feature-visible row is near-definitional (the gain is
   hop-3 minus hop-0) and must not be presented as a finding. The finding is
   that a training-free sign separates the classes the same-class fraction
   can predict from the ones it cannot.

### Why this helps rather than hurts

A reviewer who sees a within-sample ρ = 0.80 called a "law" with no
out-of-sample test will ask for one. Reporting that we ran it, registered the
prediction first, and it failed in a way that names the missing variable is
the strongest available answer — stronger than a second within-sample
correlation would have been.


### 14.1 Addendum (2026-09-04): Amazon flips the other way

Amazon-Fraud, the second real dataset, changes verdict between the two
selection rules as Photo does — in the opposite direction:

| paired OUTPOST − DEMO, n=10 | Photo | Amazon |
|---|---|---|
| oracle ROC | -0.0176 (3/0/7, p=0.0645) | -0.0021 (3/0/7, p=0.1934) |
| val-selected ROC | +0.0170 (7/0/3, p=0.1934) | **-0.0085 (0/0/10, p=0.0020)** |

On Photo the oracle column says loss and the deployable column says win; on
Amazon the oracle column says tie and the deployable column says a clean loss,
unanimous across seeds. The inflation figures explain the geometry: Photo
inflates OUTPOST by +0.0789 and Amazon by only
+0.0114, and DEMO's own inflation on each graph differs
from ours, so the gap between the columns is not a constant that cancels.

**What to say.** Not "we win under the honest rule" — Amazon shows the honest
rule can also be the one that hurts. The defensible statement is that on the
two datasets where the verdict is close, *which* method appears ahead is a
property of the selection rule, and the three datasets where OUTPOST wins
(Computers, CS, Yelp) win under both rules at n=10. Reporting both columns on
every dataset is the only version of the table that a reader who recomputes
either column cannot fault.

**What not to say.** That Amazon is a tie. Under the rule a practitioner
would use it is a loss at p=0.0020, and the paper should print it
as one.

## 16. "Which method is state of the art?" — it depends on the selection rule

The numbers live in `analysis/tables/three_method_ranking.md`, regenerated from
the run record on every refresh so that this section cannot quote a stale cell
while the last NSReg rotations land. The argument does not depend on the
decimals, only on the pattern, which was already decisive with four of five
datasets complete at n=10:

- Three methods — OUTPOST, the full published DEMO, and NSReg from its released
  code — run under one protocol at identical seeds and splits, scored two ways.
- **The ranking changes between the two selection rules on three of the five
  datasets** (Photo, Computers, Amazon) and holds on two (CS, Yelp). No method
  leads under both rules on more than two graphs. The rule, not the method,
  decides who is "state of the art" on the datasets this field reports most.
- **Both published baseline rows are wrong, in opposite directions.** DEMO's
  re-run lands below its paper on Photo and Computers (largely a training-budget
  artefact, `demo_gap_by_budget.csv`); NSReg's re-run lands 0.04–0.10 *above*
  its paper on every dataset. A table that mixes transcribed rows with re-run
  rows is not a ranking of methods.
- **OUTPOST against a fairly-run NSReg** (final, n=10 everywhere): oracle wins
  on Computers and CS; deployable wins on CS (+0.0095, 8/2,
  p=0.027) and Yelp (+0.0178, 9/1, p=0.027);
  a deployable loss on Computers (-0.0140, 2/8); ties on Photo and
  Amazon. On unseen classes the Computers oracle margin is larger (+0.07 AUC-PR)
  and vanishes under the deployable rule.
- **NSReg at its own shipped budget (201 epochs)** reproduces +0.068 above
  its published Computers figure and +0.068 above CS, matches Yelp
  (+0.007) and sits -0.024 on Photo. The
  budget is not what is wrong with the NSReg row; the row is.

**What follows for the paper.** This is not a method paper with a caveat; it is
a measurement paper with a case study. The claim that survives every check is:
*in open-set graph anomaly detection, the selection rule and training budget
decide the published ranking, the published baselines are unreliable in both
directions, and the one component that wins the semi-synthetic benchmarks
contributes nothing on real data.* OUTPOST is the instrument that exposed this,
and its honest scorecard — 3–2 against DEMO, ~1–1 against NSReg — is evidence
for the thesis, not an embarrassment to it.

## 17. "You swept your own method's hyperparameters and ran the baselines at defaults"

This is the fairness objection with the most force, and until 2026-09-07 it was
partly true: OUTPOST had a validation sweep on four datasets, NSReg had only its
released `mag_cs` configuration, DEMO only its published one. The answer is not
an argument, it is the same procedure applied to all three.

**The procedure, fixed before each grid ran.** Mean `val_auc` over seeds 0, 1 and
42, computed on the validation split alone; a 0.002 tie band; ties broken toward
the published configuration. `select_hparams.py` for OUTPOST, `select_nsreg.py`
for NSReg, the same rule for DEMO. Every selector discards test metrics before a
winner is fixed. NSReg's rule was written into `campaigns/nsreg_tune.json`'s
`_comment` before launch; DEMO's grid was registered blind (P36, P37). CS is
excluded from both baseline grids for cost — 77 minutes per rotation across eight
rotations and four configurations — and that exclusion is stated wherever a tuned
number appears.

**What the sweeps did to each method.** Not what a method paper would want.

| | OUTPOST | NSReg |
|---|---|---|
| Photo | default retained (13/13 arms in band) | **`NT_lr0.003_wd0.0`, +0.0106 oracle** |
| Computers | default retained (10/10 in band) | default retained (4/4 in band) |
| Yelp | default retained | default retained (widest spread, 0.0058) |
| Amazon | **`T_hidden16`, −0.0071 oracle** | **`NT_lr0.003_wd0.0`, +0.0047 oracle, +0.0225 deployable** |

Tuning under a common rule made the *baseline* better and the *paper's own
method* worse. A referee who suspected the sweep was flattering OUTPOST can read
the direction off the table.

**The part that is a finding rather than a concession.** Amazon is the only graph
whose validation split discriminates between configurations at all — spread
0.0090, against 0.0016 on Photo and 0.0012 on Computers. On that one graph the
rule selects, for OUTPOST, an arm that is 0.0071 *worse* on test than the default
it displaces, while the best available arm is 0.0034 better. So the paper's
selection-rule finding has two regimes, not one: where validation is saturated it
is useless, and where it is informative it was, here, misleading. This is not an
artefact of the band — no tie band below 0.0057 retains the Amazon default.

**What it cost us to act on it.** `prediction_hparam400.md` had already committed,
in writing and before the sweep ran, to replacing the main-table configuration if
validation picked a non-default arm. It did, so Amazon's OUTPOST row moves to
`T_hidden16` at ten seeds (`campaigns/sel_final.json`), and the paired tests, the
decomposition base and the three-method ranking on Amazon are recomputed there.
The expected effect, pre-registered as P32 and P33, is that OUTPOST's Amazon
deficit against DEMO *widens*. Writing the consequence down first is what stops
a falsified retention clause from quietly becoming a footnote.

## 18. "Your ablation only shows pseudo-labelling matters on datasets you built"

That is what it shows, and it is now measured on every real graph in the study
rather than argued. Removing pseudo-labelling, at 400 epochs, paired by seed
against each dataset's validation-selected arm over its complete rotation set:

| graph | n | `C_nopl` − full | seeds favouring removal | p |
|---|---|---|---|---|
| Computers | 5 | −0.0645 | 0/5 | 0.0625 |
| CS | 5 | −0.0648 | 0/5 | 0.0625 |
| Photo | 5 | −0.0377 | 0/5 | 0.0625 |
| Yelp | 10 | +0.0059 | 9/10 | 0.0273 |
| Amazon | 10 | −0.0018 | 3/10 | 0.4316 |
| T-Finance | 10 | **+0.0097** | 9/10 | **0.0039** |

Three graphs with injected anomalies: the component is worth 0.04 to 0.065 and
every seed agrees. Three graphs with observed anomalies: it is worth nothing, and
on T-Finance — the one real graph where the difference clears noise — removing it
*helps*, significantly. The n=5 rows cannot reach p<0.05 at all, since 0.0625 is
the floor for a five-sample signed-rank test, so they are read from a unanimous
sweep and a large effect rather than from a p-value.

**The obvious rebuttal, and why it fails.** One could say the real graphs are
simply harder, or that their anomalies are scattered so no propagation-based
component can help. T-Finance refutes that: it is *homophilous* (median same-class
neighbour fraction 0.629, in the semi-synthetic range and four times Yelp's) and
pseudo-labelling still fails there. The dividing line is how the labels were
produced, not how the neighbourhood looks — which is a sharper and more
uncomfortable claim than the one the ablation started with.
