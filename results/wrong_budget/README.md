# Wrong-budget shards — never merge, never cite

Runs that completed correctly but at a training budget the protocol does not
report, so they are not comparable to the rows they would sit beside.

- `ogbn-arxiv_demo_s*_B_demo_mix_rot*.json` — DEMO on ogbn-arxiv at **200** epochs,
  2026-09-14. `campaigns/arxiv_demo.json` omitted the epoch override. DEMO never
  inherits a dataset's `num_epochs`: `load_config()` takes only `input_dim` and
  `eval_batch_mult` from the dataset block, and `demo_default` sets 200. Every
  main-table DEMO arm reaches 400 by passing `num_epochs: 400` in its job spec.
  OUTPOST on arxiv is 400, so these are half-budget and would understate DEMO.

Quarantined rather than deleted so the mistake stays visible. They carry the
main-table tag `B_demo_mix`, which is exactly why leaving them in
`results/rotations/` was dangerous: nothing downstream filters on budget for that
tag.
