# Full DEMO baseline (mixup enabled)

Copy these seven files over the ones in the OUTPOST folder, **including
`config.json`**. `results/`, `data/` and every completed run are untouched, and
**no OUTPOST run is affected** - all changes are in the DEMO path.

## Run

```bash
python main.py --dataset photo     --method demo --seed 42
python main.py --dataset computers --method demo --seed 42
python main.py --dataset cs        --method demo --seed 42
```

Repeat for the seeds OUTPOST used on each dataset, so the comparison stays
paired on the same machine. PPR is computed once per dataset and cached to
`data/<name>/ppr_matrix.npy`.

**`--method demo` now runs the published configuration by default.** Every run
prints which variant it used:

```
==================================================================
DEMO variant: FULL published method (mixup ON)
==================================================================
```

The ablation is still reachable deliberately with `--set mixup=false`, and says
so in its banner.

## Feasibility

DEMO's mixup needs a dense N x N PPR matrix:

| dataset | PPR | full DEMO |
|---|---|---|
| photo | 0.5 GB | yes |
| computers | 1.5 GB | yes |
| cs | 2.7 GB | yes |
| yelp | 17 GB | large-memory node |
| ogbn-arxiv | 229 GB | not computable |
| ogbn-mag | 4.3 TB | not computable |

**Nothing in this code ever downgrades to the ablation.** Where the matrix will
not fit, the run fails with an explicit MemoryError naming the size required. It
never silently falls back to mixup off, because an ablation reported as the
baseline is worse than no baseline at all.

Where full DEMO cannot run, make no paired claim and compare against the
published values instead, stating the reason.

## What was wrong

The authors' released `main.py` defaults `mixup: False`, so our baseline
inherited their ablation setting. Setting it true alone did not work - six
defects had to be fixed:

1. **PPR was never wired in.** Their code computes and caches the matrix and
   passes it to `train()`; ours passed `None`, so `anomaly_mixup()` raised
   `TypeError`.
2. **`compute_ppr` was missing imports** (`to_networkx`, `networkx`, `inv`,
   `fractional_matrix_power`) lost in an earlier refactor.
3. **`args.device` was the literal `0`** from DEMO's config, crashing the mixup
   path on CPU-only machines.
4. **The neighbour sampler indexed CPU tensors with CUDA indices** - DEMO's
   unlabeled dataloader passes a CUDA `node_idx`. Only fails on GPU.
5. **DEMO wrote no rows to `results/results.csv`** - `train()` returned `None`.
6. **DEMO got 200 epochs where OUTPOST got 400** on yelp/ogbn-arxiv/ogbn-mag;
   the config merge dropped `num_epochs`. Under best-over-epochs selection that
   favoured OUTPOST.

## Config verified against the authors' repo

`demo_default` matches their `photo.yaml` key for key. Differences: `input_dim`
(correctly per-dataset), `damping` (1e-3 vs 0.001, identical), and `lbl_lr`,
which is in their YAML but referenced nowhere in their code. `mixup` is the one
deliberate change - set to their *published* configuration rather than their
repo default.

## Verified working

Photo and Computers completed with mixup on, 5 seeds each, on a Kaggle T4. PPR
computed and cached for photo, computers and cs. Both variant banners tested.

## Open issue to be aware of

With mixup on, our DEMO still lands below the published values:

| Photo | AUC-ROC | AUC-PR |
|---|---|---|
| DEMO published | 0.9023 | 0.6330 |
| ours, mixup on (5 seeds) | 0.8304 +- 0.0154 | 0.5262 +- 0.0198 |

Enabling mixup moved Photo AUC-ROC by +0.0004 and Computers by +0.0002, where
the authors' own ablation puts mixup at +0.031 and +0.017 AUC-PR. Since the
config matches their repo, this looks like a reproduction gap in the released
code. Worth reporting both numbers in the paper.
