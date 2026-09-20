"""Generate the GPU-7 campaign: everything except cs, priority-ordered.

Kept as a script rather than hand-written JSON so the experiment matrix reads as
intent - which arm tests which claim - instead of 200 near-identical objects.

Ordering matters more than it looks. Measured on this hardware, a post-warmup
epoch costs ~18 s on computers and the pseudo-label pass runs over the whole
unlabelled pool, so one computers seed is ~5 GPU-hours and the full queue is
days rather than hours. The bands below are therefore ordered by what each run
BUYS, not by dataset size:

   1  Yelp main + paired DEMO      the headline claim, the only real-fraud graph
   2  Photo main + paired DEMO     cheapest complete dataset + pairing
   3  Yelp component ablations     what the method's parts are worth
   4  DEMO energy term on/off      is our Photo gap a crippled baseline?
   5  Yelp dose-response           makes the SimSample effect causal, not incidental
   6  Photo tuning sweep           validation-only, pre-registered; also sensitivity
   7  Computers main + paired DEMO a further main-table cell with pairing
   8  cross-dataset SimSample      the sign-change prediction on sparse graphs

The two OGB graphs are NOT here - they run on GPU 6 (campaigns/build_gpu6.py)
since cs freed that card.

An interrupted campaign then leaves the top of that list finished rather than
every band half-done.

ONE RUNNER PER GPU. The energy and tuning sweeps live in this queue rather than
in runners of their own, because each runner keeps its own memory budget and
they cannot see each other: three runners on one card believed they had 21 + 9 +
9 GB of a 24 GB device between them, and the driver-level free-memory check is
racy when two of them look at the same moment. Adding a campaign means adding a
band here, not starting a second process on the same device.
"""
import json
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S = [42, 0, 1, 2, 3]

# Yelp gets ten seeds rather than five, and the reason is a power calculation
# that does not depend on any observed effect. The pairing unit for a paired
# test is (seed, rotation); Yelp is a binary fraud graph with ONE rotation, so
# its unit count equals its seed count. At n=5 the smallest p-value the Wilcoxon
# signed-rank test can return is 2^-4 = 0.0625 even when every pair favours the
# same arm - significance is unreachable by construction, not by the data. n=10
# puts the floor at 0.002. Every other dataset already has 2-8 rotations and
# reaches 10-40 pairs at five seeds.
#
# Committed here BEFORE the extra runs were looked at, and fixed at ten: this is
# a power fix, not a licence to keep adding seeds until a threshold is crossed.
S_YELP = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]
jobs = []


def add(ds, tag, sets=None, seeds=S, prio=5, method="outpost", shard=False):
    for s in seeds:
        j = {"dataset": ds, "seed": s, "tag": tag, "priority": prio,
             "method": method}
        if sets:
            j["set"] = sets
        if shard:
            j["shard_rotations"] = True
        jobs.append(j)


# 1-2: main table + the paired DEMO baseline. Same seeds means the same splits,
# which is what makes the comparison paired rather than two independent samples.
#
# num_epochs is set explicitly for the DEMO arm. `--method demo` takes DEMO's own
# hyperparameters, whose num_epochs is 200, but the protocol budget (section 3)
# is 400 on the large-scale graphs. Left alone, DEMO would get 200 epochs on Yelp
# while OUTPOST gets 400 - and under best-over-epochs selection more epochs can
# only help, so the comparison would be rigged in our favour on the one dataset
# carrying the headline claim.
add("yelp",  "A_main", seeds=S_YELP, prio=1)
add("yelp",  "B_demo", {"energy_loss": False, "num_epochs": 400},
    seeds=S_YELP, prio=1, method="demo")
add("photo", "A_main", prio=2)
add("photo", "B_demo", {"energy_loss": False}, prio=2, method="demo")  # 200 both

# 3: Yelp component ablations, one flag each. The A_main rows ARE the full-system
# arm and are not re-run.
#
# Four arms get the full ten seeds, for the same reason A_main does: the pairing
# unit on a one-rotation dataset is the seed, and n=5 puts the Wilcoxon floor at
# 0.0625. Two of them carry the method's causal claim (C_nosim, C_simplacebo);
# the other two carry NULL results at five seeds - removing the atlas gate or
# pseudo-labelling changed nothing - and a null needs more power to assert than
# an effect does, not less. The remaining four arms stay at five seeds: their
# direction is not in doubt and they are not load-bearing.
add("yelp", "C_nosim",       {"sim_topk_frac": 0.0}, seeds=S_YELP, prio=3)
add("yelp", "C_simplacebo",  {"sim_shuffle": True}, seeds=S_YELP, prio=3)
add("yelp", "C_nogate",      {"use_atlas_gate": False}, seeds=S_YELP, prio=3)
add("yelp", "C_nopl",        {"use_pl": False}, seeds=S_YELP, prio=3)
add("yelp", "C_noconformal", {"use_conformal": False}, prio=3)
add("yelp", "C_synth",       {"use_mixup": True, "use_halo": True}, prio=3)
add("yelp", "C_hopmix",      {"use_hybrid": True}, prio=3)
add("yelp", "C_fview",       {"use_fview_gate": True}, prio=3)

# 4: DEMO's energy-gradient term, ON and OFF, run to completion on Photo. Photo
# is the one dataset where its per-training-node second-order backward is
# affordable, and it is the leading candidate explanation for why BOTH our
# OUTPOST and our re-run DEMO land ~0.07 below DEMO's published Photo number:
# we disabled the term, and their number includes it. See campaigns/build_energy.py.
for s in [42, 0, 1]:
    jobs.append({"dataset": "photo", "seed": s, "method": "demo",
                 "tag": "E_demo_energy", "priority": 4,
                 "set": {"energy_loss": True}})
    jobs.append({"dataset": "photo", "seed": s, "method": "demo",
                 "tag": "E_demo_noenergy", "priority": 4,
                 "set": {"energy_loss": False}})

# 5: dose-response. An ablation says a component matters; a monotone curve in the
# component's own strength says the mechanism claimed for it is the one acting.
for f in [0.25, 0.5, 0.75]:
    add("yelp", f"D_sim{f}", {"sim_topk_frac": f}, prio=5)

# 6: validation-only hyperparameter sweep on Photo. Grid and selection rule are
# pre-registered in analysis/tables/prediction_tuning.md; each arm doubles as a
# multi-seed sensitivity point. The default arm is not re-run - A_main is it.
TUNING = {
    "T_lambda_un0.5": {"lambda_un": 0.5},  "T_lambda_un1.5": {"lambda_un": 1.5},
    "T_Kp4": {"K_p": 4},                   "T_Kp12": {"K_p": 12},
    "T_hidden32": {"hidden_dim": 32},      "T_hidden128": {"hidden_dim": 128},
    "T_warmup15": {"warmup_epochs": 15},   "T_drop0.3": {"drop_out": 0.3},
    "T_lean": {"use_mixup": False, "use_halo": False},
    "T_alpha0.01": {"alpha_plus": 0.01},   "T_gateq0.8": {"atlas_gate_q": 0.8},
    "T_lr0.003": {"lr": 0.003},
}
for tag, sets in TUNING.items():
    add("photo", tag, sets, seeds=[42, 0, 1], prio=6)

# Computers moved to campaigns/gpu6.json when arxiv freed that card - see the
# note there. A dataset must live in exactly one queue.

# 8: SimSample is claimed to help only where degree exceeds the sampling budget.
# That predicts a SIGN CHANGE on sparse graphs, so it is tested by switching
# SimSample ON where the claim says it must hurt.
add("photo",     "C_sim",     {"sim_topk_frac": 1.0}, prio=8)
# computers C_sim also lives in the GPU-6 queue, with the rest of computers
add("photo",     "C_nosynth", {"use_mixup": False, "use_halo": False}, prio=8)
add("photo",     "C_nogate",  {"use_atlas_gate": False}, prio=8)

# The two OGB graphs moved to campaigns/gpu6.json when cs freed that card. They
# must not appear in both queues: the shard-resume check skips work that has
# FINISHED, so two runners holding the same dataset can dispatch the same
# rotation simultaneously.

json.dump({"_comment": __doc__, "seeds": S, "jobs": jobs},
          open("campaigns/gpu7.json", "w"), indent=1)
print(len(jobs), "jobs queued")
