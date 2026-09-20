# NSReg as a second reproduced baseline

NSReg (Wang, Pang, Salehi, Xia, Leckie — ICLR 2025, *Open-Set Graph Anomaly
Detection via Normal Structure Regularisation*) is run from its released code
under OUTPOST's protocol. This note records exactly what was held identical,
what is NSReg's own, and how each claim was checked.

## Why NSReg

Every head-to-head in the paper was OUTPOST vs DEMO. NSReg is the paper that
defined the open-set protocol this repository uses — the rotation over anomaly
classes, 5% labelled normals, 50 labelled anomalies of the seen class — and its
released code names the datasets `amz_photo`, `amz_computers`, `mag_cs`, `yelp`.
Its `split_info` dictionary has the same keys as ours. Adding it makes the
ranking-flip claim a three-method claim and gives the reproduction-gap analysis
a second instance.

## Held identical to OUTPOST

| what | how | checked by |
|---|---|---|
| graph tensors | `utils.load_data`, not NSReg's loader | same function object |
| split | `utils.ad_split_num` after `set_seed(seed)`, per rotation — main.py's exact call sequence; handed to NSReg through its own recorded-split path (`split_50.pkl` written per process) | `split_sha` in every shard: photo/s42/rot0 gives `ff3524eb571910b7` from **both** `main.py` and the wrapper |
| metrics | AUC-ROC / AUC-PR over test-all and test-unknown; `best` = per-metric max over epochs; `val_selected` = current-epoch metrics at the validation-AUC peak | matches `trainer.py` lines 660–680 |
| validation set | OUTPOST's `idx_val` (1% normals + 30 seen anomalies). NSReg has none — its split sets `idx_val = idx_train` | wrapper computes `val_auc` on `idx_val` |
| binary graphs | test-unknown has no anomalies on Yelp/Amazon → ROC 0.5, PR 0.0 | same convention as OUTPOST's shards |
| shard / merge | `main.write_rotation(method="nsreg")`; `merge_rotations.py` unchanged | `results.csv` rows appear as method `nsreg` |
| evaluation cadence | every epoch (`eval_every=1`); NSReg's default is every 10 | needed for best-over-epochs parity |

## NSReg's own

Architecture (2-layer GraphSAGE, width 64, [25,10] sampling, batch 512), the
edge labeller, the normal-structure regulariser, losses, optimiser, learning
rates, weight decay 0.0, batch mode. **Only the `mag_cs` configuration was
released**; it is applied to every dataset with `input_dim` changed. Their
per-dataset settings for Photo, Computers and Yelp are not public. Stated as a
limitation wherever NSReg numbers appear.

Two budgets are run:

- `N201_nsreg` — 201 epochs, their config as shipped. The reproduction check
  against their published Photo 0.836 / Computers 0.740 / CS 0.903 / Yelp 0.702.
- `E400_nsreg` — 400 epochs, the paper's uniform budget, the same one DEMO is
  granted.

## Environment fixes, none touching NSReg's source

1. **Sampler.** NSReg uses `torch_geometric.loader.NeighborSampler`, which
   requires `torch-sparse`; this environment stubs that out (the installed wheel
   was built against torch 2.0.1). The module-level name in
   `runners.train_runner` is rebound to OUTPOST's `NeighborSamplerShim`, which
   yields the same `(edge_index, e_id, size)` triples NSReg's encoder unpacks.
   OUTPOST itself trains with that shim.
2. **PyG version skew.** NSReg (PyG 2.0.4) passes numpy index arrays to
   `torch_geometric.utils.subgraph`; PyG 2.7's `index_to_mask` calls `.view(-1)`
   on them. The same arrays go through `torch.from_numpy` elsewhere in their
   runner, so they cannot be converted up front. `subgraph` is wrapped on the
   module object their runner imported, coercing numpy to `LongTensor`.
3. **Namespace collision.** NSReg's `utils/` is a namespace package and
   OUTPOST's `utils.py` is a module; a module anywhere on `sys.path` wins. The
   wrapper binds OUTPOST's names, removes the repository root from `sys.path`,
   drops `utils` from `sys.modules`, then imports NSReg.
4. **Checkpointing** is disabled (`save_ckpt`, `create_ckpt_dir` → no-op).

## Smoke test, 2026-09-04

Photo, seed 42, rotation 0, 3 epochs, GPU: exit 0, 3 evaluations recorded,
118,274 parameters, split hash identical to `main.py`'s. Shard removed after
inspection.

## Initialisation check on all five datasets, 2026-09-04

One full-length rotation per dataset was started under the wrapper (seed 42,
tag `E400_nsreg`, 400 epochs) on a shared card. All five passed the phase
where integration failures live — data load, split injection, the
fully-connected normal-train set on Yelp (≈3.9M pairs), the 6805-dimensional
full-graph pass on CS, sampler and PyG-version shims — and were training:
Amazon epoch 16, Photo 11, Computers 7, Yelp 5, CS 1 at the time they were
stopped. Their per-epoch rates were not used for cost estimates (seven
processes were sharing one GPU); the jobs were handed to the scheduler and
re-run cleanly.

## Published Amazon-Fraud figures

None are transcribed. The cloned GGAD, NSReg and ConsisGAD repositories carry
no Amazon results table in their READMEs, and GGAD's setting (normal-only
supervision) is not this protocol. Rather than quote numbers from memory,
Amazon appears in the paper as a head-to-head only (OUTPOST vs DEMO vs NSReg,
all reproduced under one protocol), with no "published field" column.

## CS memory, measured (2026-09-04)

NSReg embeds the whole graph every epoch, with autograd, in one 50,000-node
batch. On CS (18,333 nodes, 6,805-dim features) that does not fit a 47 GB card:

| train chunk | eval chunk | card state | peak allocated | outcome |
|---|---|---|---|---|
| 50,000 | 50,000 | shared | ~24 GB | OOM (asked 4.2 GB more) |
| 50,000 | 50,000 | **empty** | 39.2 GB | OOM (asked 8.5 GB more) |
| 1,024 | 50,000 | empty | 43.2 GB | OOM in `val()` at epoch 3 |
| **1,024** | **1,024** | empty | **23.3 GB** | **53 s / 5 epochs incl. init** |

What scales with the batch is the *transient* first-layer tensors — ~335k
sampled edges × 6,805 features, ~8.5 GB each, more than one alive at once —
not the retained autograd set. Chunking the training pass to 1,024 targets
brings the peak to 23.3 GB; the evaluation pass has to be chunked too, because
its own gather is the same size and it runs on top of what stays resident.
Both chunk sizes are recorded in every shard's config. Per-target sampling is
unchanged by chunking; only batch boundaries move.

Scheduler budget: CS-NSReg owns a card (`MEM_GB[("cs","nsreg")] = 45`), like
CS-OUTPOST. The other four datasets run at NSReg's native 50,000.

## A silent exclusion, found and fixed (2026-09-05)

NSReg's configuration names the training budget `n_epochs`; OUTPOST's names it
`num_epochs`. `results_writer.append_row` copied `cfg.get("num_epochs")` into
`results.csv`, so every NSReg row carried a blank budget — and every consumer of
that file (`make_paper_tables.py`, `stats.py`, `make_figures.py`) filters
`num_epochs >= 100` to keep smoke runs out of the tables. The 70 NSReg rows
passed through the merge, sat in the file, and were dropped by the guard in
every downstream script. The appendix reads the per-rotation shards directly and
so showed NSReg throughout; the LaTeX tables, `stats.json` and the Yelp figure
did not, for a day.

Fixed at both ends: the row writer falls back to `n_epochs`, the wrapper mirrors
`num_epochs` into its config, and `merge_rotations.py --rewrite` regenerated the
rows. This is the fifth bug of the same shape in this project — a default or a
guard silently winning over data that is present — after the mixup default, the
merge skip, the `--force` resume and the hard-coded label budget.
