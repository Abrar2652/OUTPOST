"""Generate the GPU-6 queue: cs, which needs a whole card to itself.

cs is the expensive dataset: 6805-dimensional features, 8 rotations, and a
pseudo-label pass over the full unlabelled pool every epoch. Measured at ~12 s
per epoch with the feature table on the device, that is ~40 min per rotation and
~5.3 GPU-hours per seed.

Ordering is seed-major, so an interruption leaves COMPLETE seeds rather than a
scatter of rotations that no row can be built from.

THERE IS NO DEMO ARM HERE, and the reason is a hardware limit rather than a
choice. Both methods retain the autograd graph over the whole unlabelled pool to
take one full-batch step per epoch, and on cs that pool is ~17,000 nodes whose
features are 6805-dimensional. OUTPOST peaks at 18.3 GB and fits a 24 GB A5000;
DEMO needs ~22 GB allocated and does not, failing partway through the first
unlabelled pass. Measured after both memory fixes to the DEMO port (cached
evaluation loader, no-grad weak view) and after tightening the allocator with
`max_split_size_mb:128`, which recovered 5 GB of fragmentation and still left it
short. A card of 32 GB or more would run it unchanged; nothing else would.

The paired cs comparison is therefore unavailable on this hardware and cs uses
the transcribed DEMO constant, flagged as such in METHODOLOGY section 4. The
GPU-hours that would have gone to it go to cs seeds 2 and 3 instead, so the cs
main-table cell is five seeds rather than three.
"""
import json
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROTS = [0, 1, 3, 6, 8, 9, 12, 14]
jobs = []

for s in [42, 0, 1, 2, 3]:
    for c in ROTS:
        jobs.append({"dataset": "cs", "seed": s, "tag": "A_main",
                     "rotations": [c], "priority": 1})

json.dump({"_comment": __doc__, "jobs": jobs},
          open("campaigns/gpu6_cs.json", "w"), indent=1)
print(len(jobs), "cs rotation-jobs")
