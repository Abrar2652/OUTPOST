# Smoke-test shards — never merge, never cite

Runs made only to prove a code path works. They use truncated budgets and would
be meaningless in any table.

- `ogbn-arxiv_demo_s0_SMOKE_demo_rot4.json` — 3 epochs, 2026-09-13. Written to
  prove DEMO could run on ogbn-arxiv against the newly built 114.7 GB PPR matrix
  after that combination had been recorded as infeasible. It proved the point
  (rotation 4: ROC 0.5184, params 68,097, peak GPU 3.50 GB) and has no other use.

Quarantined here because `results/rotations/` is the merge source: a 3-epoch shard
sitting beside 400-epoch ones is precisely how a truncated run reaches a paper
table. `build_appendix.py` filters `num_epochs >= 100` and `make_paper_tables.py`
matches exact tags, so neither would have picked it up — but relying on two
downstream filters to catch it is weaker than not leaving it there.
