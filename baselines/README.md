# Baselines

The three comparison methods are **not vendored here**, and the reason is licensing
rather than size. `fetch_baselines.sh` clones them into this directory, where the
integration code below expects them.

| baseline | upstream license | redistributable here? |
|---|---|---|
| NSReg | Apache-2.0 | yes in principle, but our working copy had lost the upstream `LICENSE` file, and Apache-2.0 requires it to travel with the code |
| ConsisGAD | MIT | yes in principle, same reason as above |
| GGAD | **none granted** — the upstream README still reads "Pick a licence and describe how to contribute" | no |

Redistributing GGAD would mean redistributing code whose authors never granted a
licence to do so. Rather than ship two of the three and explain the gap, none are
shipped and all three are fetched the same way.

## Fetch them

```bash
bash baselines/fetch_baselines.sh
```

Pin commits before using the results in anything you publish: baselines move, and a
number you cannot re-derive is not a comparison. The script prints the commit it
checked out for each.

## What IS in this repository

* `run_nsreg.py` — our runner. It drives NSReg's own trainer but loads graphs with
  OUTPOST's `utils.load_data`, so both methods see identical tensors, splits and
  seeds. Without this the two would differ by their loaders before any model ran.
* `NSREG_INTEGRATION.md` — what had to be adapted and why, including the arguments
  whose defaults differ between the paper and the released code.

## A measured limit, so you do not repeat it

NSReg does not run on `ogbn-mag`, and not for want of GPU memory. Before its first
epoch it builds the complete graph over the labelled-normal *training* set — once in
`make_fully_connected`, once as a Python list comprehension over every ordered pair.
That is quadratic in the number of labelled normals:

| graph | labelled normals in train | ordered pairs |
|---|---|---|
| Amazon | 556 | 308,580 |
| Yelp | 1,963 | 3,851,406 |
| ogbn-arxiv | 7,075 | 50,048,550 |
| ogbn-mag | 36,662 | **1,344,065,582** |

A probe reached 280 GB of host RAM in thirteen minutes without finishing setup, while
holding 404 MiB on the GPU. See METHODOLOGY 11.5.
