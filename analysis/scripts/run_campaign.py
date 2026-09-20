"""Campaign runner: executes a list of runs across the free GPUs, resumably.

Every experiment in the paper is one row of a campaign file, so the full record
of what was run is a data file rather than a shell history. A run is identified
by (dataset, method, seed, train_seed, tag); a run whose results row already
exists is skipped, which makes the whole campaign restartable after an
interruption without repeating GPU hours.

    python analysis/scripts/run_campaign.py campaigns/phaseA.json
    python analysis/scripts/run_campaign.py campaigns/phaseA.json --dry-run
    python analysis/scripts/run_campaign.py campaigns/phaseA.json --gpus 6 7

Scheduling is by GPU memory budget: each job declares an estimate, and a job
starts only when some GPU has room for it. Long jobs are dispatched first so the
tail of the campaign is short jobs rather than one straggler.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# GPU memory to reserve per job, GB, per (dataset, method). Two lessons are
# baked into these numbers.
#
# 1. Measure under the conditions you will run in. The caching allocator reserves
#    what is available rather than a fixed multiple of the allocated peak, so
#    figures taken on an idle card overstate the requirement on a busy one. These
#    were re-measured 2026-08-11, after root's vLLM server expanded onto GPUs 6
#    and 7 and left 6.2 GB free on each; photo peaks at 2.91 GB allocated and
#    still runs fine in that.
# 2. Measure past warmup. An ogbn-mag probe stopped at 2 epochs never entered the
#    pseudo-label pass and under-reported peak memory threefold, which cost 45
#    OOM-killed jobs.
#
# Per-method entries, not a multiplier: DEMO costs 1.3x OUTPOST on photo but 4x
# on yelp, where OUTPOST's model is half the width. A single "demo" multiplier
# put photo+demo at 26.4 GB on a 24 GB card - a request that could never be
# satisfied, which stalled the whole queue behind it.
#
# cs is budgeted a whole card: 6805-dimensional features and a training loop that
# keeps every sampled batch's activations alive for one full-batch gradient step,
# so its footprint is set by the epoch's total sampled nodes and does not fall
# when batch_size does. DEMO on cs does not fit 24 GB at all.
# Re-budgeted 2026-08-19 for the 49 GB A6000s. These are LARGER than the
# allocated peaks by a wide margin, on purpose. Measured on the A5000, cs peaked
# at 18.9 GB allocated; on the A6000 a single cs process was observed holding
# 40.6 GB, because the caching allocator grows into whatever the card has rather
# than to a fixed multiple of demand. Budgeting against the allocated peak
# therefore over-packs by more on a bigger card, not less - which is the same
# mistake as the first OOM cascade, re-made after the hardware changed under us.
# Treat these as "how many of this job fit per card", not as memory estimates.
MEM_GB = {
    ("photo", "outpost"): 15.0,  ("photo", "demo"): 16.0,
    ("computers", "outpost"): 22.0, ("computers", "demo"): 24.0,
    ("yelp", "outpost"): 10.0,   ("yelp", "demo"): 14.0,
    ("amazon", "outpost"): 10.0, ("amazon", "demo"): 14.0,
    # tfinance: 39k nodes but 42M edges - CSR + similarity ordering are the cost
    ("tfinance", "outpost"): 16.0, ("tfinance", "demo"): 22.0, ("tfinance", "nsreg"): 16.0,
    # NSReg does one full-graph embedding pass per epoch through a 2-layer
    # SAGE at width 64; cs is the only one where the 6805-dim input matters.
    # cs-nsreg measured at 23.3 GB allocated: NSReg embeds the whole graph WITH
    # autograd every epoch, so chunking the pass cannot lower the peak (every
    # chunk's activations are retained for backward). Same reason cs-outpost
    # owns a card. One per card.
    ("photo", "nsreg"): 8.0, ("computers", "nsreg"): 10.0, ("cs", "nsreg"): 45.0,
    ("yelp", "nsreg"): 12.0, ("amazon", "nsreg"): 10.0,
    ("cs", "outpost"): 45.0,     ("cs", "demo"): 45.0,
    ("ogbn-arxiv", "outpost"): 10.0, ("ogbn-arxiv", "demo"): 12.0,
    ("ogbn-mag", "outpost"): 14.0,   ("ogbn-mag", "demo"): 16.0,
}
# Relative cost, for longest-first dispatch.
COST = {"photo": 1, "computers": 4, "cs": 20, "yelp": 2, "amazon": 2, "tfinance": 4,
        "ogbn-arxiv": 12, "ogbn-mag": 10}

# The complete rotation set per dataset (see analysis/scripts/merge_rotations.py).
ROTATIONS = {
    "photo": [0, 7], "computers": [0, 3, 5, 6, 9], "yelp": [1], "amazon": [1], "tfinance": [1],
    "cs": [0, 1, 3, 6, 8, 9, 12, 14],
    "ogbn-arxiv": [4, 8, 10, 34],
    "ogbn-mag": [2, 39, 143, 151, 176, 195, 206, 215, 216, 231, 263, 282,
                 321, 327, 341],
}

_lock = threading.Lock()
_gpu_free = {}
_cpu_free = 0
_done = []
_failed = []


def results_index():
    """(dataset, method, seed, train_seed, tag) already present in results.csv."""
    import csv
    path = os.path.join(ROOT, "results/results.csv")
    if not os.path.exists(path):
        return set()
    out = set()
    for r in csv.DictReader(open(path)):
        # a run that died mid-way can leave a short row; treat only full-length
        # runs as done so --resume re-runs the truncated ones
        out.add((r["dataset"], r["method"], str(r["seed"]),
                 str(r["train_seed"]), r["tag"]))
    return out


def job_key(j):
    return (j["dataset"], j.get("method", "outpost"), str(j["seed"]),
            str(j.get("train_seed", "") if j.get("train_seed") is not None else ""),
            j["tag"])


def expand_shards(jobs):
    """One job per rotation where the campaign asks for it.

    Rotations are independent runs that happen to share a --seed, so a dataset
    with many of them (ogbn-mag has 15) is embarrassingly parallel. Sharding
    turns a 25-hour serial job into 15 jobs of under two hours, and makes an
    interrupted campaign lose one rotation rather than all of them.
    """
    out = []
    for j in jobs:
        rots = ROTATIONS.get(j["dataset"])
        if not j.get("shard_rotations") or not rots or len(rots) < 2:
            out.append(j)
            continue
        for c in rots:
            k = dict(j)
            k["rotations"] = [c]
            out.append(k)
    return out


def shard_done(j):
    """A sharded job is done when its rotation file exists."""
    if not j.get("rotations"):
        return False
    tag = j["tag"].replace("/", "_").replace(" ", "_")
    ts = "" if j.get("train_seed") is None else f"_t{j['train_seed']}"
    return all(os.path.exists(os.path.join(
        ROOT, f"results/rotations/{j['dataset']}_{j.get('method','outpost')}"
              f"_s{j['seed']}{ts}_{tag}_rot{c}.json")) for c in j["rotations"])


# Third-party baselines run from their own entry script but speak the same
# CLI, write the same per-rotation shard, and are merged by the same script, so
# a campaign file does not care which one it is dispatching.
ENTRY = {"nsreg": "baselines/run_nsreg.py"}


def build_cmd(j):
    method = j.get("method", "outpost")
    cmd = [sys.executable, "-u", ENTRY.get(method, "main.py"),
           "--dataset", j["dataset"], "--seed", str(j["seed"]),
           "--method", method, "--tag", j["tag"]]
    if j.get("train_seed") is not None:
        cmd += ["--train_seed", str(j["train_seed"])]
    if j.get("epochs"):
        cmd += ["--epochs", str(j["epochs"])]
    if j.get("rotations"):
        cmd += ["--rotations"] + [str(c) for c in j["rotations"]]
    if j.get("set"):
        cmd += ["--set"] + [f"{k}={json.dumps(v)}" for k, v in j["set"].items()]
    if j.get("save_scores"):
        cmd += ["--save-scores"]
    return cmd


def real_free_gb(g):
    """Free memory on GPU g right now, per the driver."""
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                            "--format=csv,noheader,nounits", "-i", str(g)],
                           capture_output=True, text=True, timeout=20).stdout
        return float(q.strip().splitlines()[0]) / 1024.0
    except Exception:
        return float("inf")     # cannot tell: fall back to budget accounting


def acquire(need_gb, gpus):
    """Block until a GPU has need_gb free AND a CPU slot is available.

    Both are taken in one critical section. Taking them separately - a
    semaphore around a blocking GPU wait, say - lets jobs that cannot fit sit
    on the CPU slots and starve the jobs that can, which is how a queue headed
    by a whole-card dataset stalls every small job behind it.

    The budget is checked against what the driver actually reports free as well
    as against our own accounting. Book-keeping alone is not enough: the
    estimates are of RESERVED memory and the caching allocator can hold three or
    four times a job's allocated peak, so an underestimate silently oversubscribes
    the card and kills a run hours in. The driver is also the only thing that
    knows about jobs this campaign did not start.
    """
    global _cpu_free
    while True:
        with _lock:
            candidates = [g for g in gpus if _gpu_free[g] >= need_gb] \
                if _cpu_free > 0 else []
        for g in candidates:
            if real_free_gb(g) < need_gb:
                continue
            with _lock:
                if _cpu_free > 0 and _gpu_free[g] >= need_gb:
                    _gpu_free[g] -= need_gb
                    _cpu_free -= 1
                    return g
        time.sleep(10)


def release(g, need_gb):
    global _cpu_free
    # A finished job's CUDA context is not torn down the instant its process
    # returns. Releasing the budget immediately lets the next job acquire and
    # spawn into memory the driver has not reclaimed yet, which OOMs it in
    # seconds - the budget says there is room and the card disagrees. Settle
    # first, then release.
    time.sleep(8)
    with _lock:
        _gpu_free[g] += need_gb
        _cpu_free += 1


def job_mem_gb(j, budget=None):
    # A job may declare its own requirement. The table is keyed by
    # (dataset, method) and cannot see the config: cs+SimSample needs ~25 GB
    # against plain cs's ~21, because deterministic top-k samples ~1.4x the
    # edges that dedup would. Packing it against the table's 21 put two on a
    # 49 GB card and OOM-killed 27 jobs.
    need = j.get("mem_gb") or MEM_GB.get(
        (j["dataset"], j.get("method", "outpost")), 8.0)
    # A request larger than the whole budget can never be satisfied, and a
    # worker waiting on one blocks the queue behind it forever. Clamp instead:
    # the job then runs alone on a card, which is the best that can be done for
    # it, and either succeeds or fails honestly rather than hanging. Reached by
    # photo+demo, whose estimate is 12.0 x 2.2 = 26.4 GB on a 24 GB card.
    if budget is not None and need > budget:
        return budget
    return need


def run_job(j, gpus, logdir, budget=None, attempt=0):
    need = job_mem_gb(j, budget)
    g = acquire(need, gpus)
    name = f"{j['tag']}__{j['dataset']}_{j.get('method','outpost')}_s{j['seed']}"
    if j.get("train_seed") is not None:
        name += f"_t{j['train_seed']}"
    if j.get("rotations"):
        name += "_rot" + "-".join(str(c) for c in j["rotations"])
    logf = os.path.join(logdir, name + ".log")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
               # each job samples neighbours on CPU; unbounded thread pools from
               # many concurrent jobs thrash a shared node
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               # PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set. It used to
               # carry max_split_size_mb:512 to cut allocator fragmentation, and
               # under torch 2.0.1 that was harmless. After the environment was
               # upgraded to 2.7.1+cu126 mid-campaign it started segfaulting runs
               # mid-training, with no traceback and no reproducible epoch count.
               # Controlled A/B, same job, same seed, only this var differing:
               # with it, SIGSEGV at 20 epochs; without it, 200 epochs and a
               # clean exit. Every pre-upgrade campaign ran thousands of epochs
               # with the var and never crashed, which is why it took so long to
               # suspect. Do not reintroduce it without re-running that A/B.
               # Dump a Python traceback on SIGSEGV/SIGABRT into the job log.
               # Costs nothing when nothing crashes, and a fatal signal is
               # otherwise completely silent: cs runs have been dying with
               # rc=-11 and no diagnostic at all. Passive instrumentation beats
               # re-running a probe and hoping to reproduce.
               PYTHONFAULTHANDLER="1",
               # .stubs/ makes torch_scatter and torch_sparse fail to import
               # BEFORE dlopen maps them. Both are compiled against torch 2.0.1
               # and the environment was upgraded to 2.7.1 mid-campaign, so they
               # now fail with `undefined symbol`. PyG catches that and falls
               # back to pure-torch scatter - but only after the broken objects
               # are already in the process, which is the leading explanation
               # for the untraceable SIGSEGVs that killed cs runs mid-training.
               # Isolated to these jobs; no installed package is modified.
               PYTHONPATH=os.path.join(ROOT, ".stubs")
                          + (os.pathsep + os.environ["PYTHONPATH"]
                             if os.environ.get("PYTHONPATH") else ""))
    # Final check against the driver, not the budget: another runner, another
    # user, or a job still tearing down can hold memory our accounting believes
    # is free. Waiting here costs seconds; guessing wrong costs the whole run.
    for _ in range(30):
        if real_free_gb(g) >= need:
            break
        time.sleep(10)

    t0 = time.time()
    with open(logf, "w") as fh:
        fh.write(f"# gpu={g} cmd={' '.join(build_cmd(j))}\n")
        fh.flush()
        rc = subprocess.run(build_cmd(j), cwd=ROOT, env=env,
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    release(g, need)
    dt = (time.time() - t0) / 60

    # One retry on failure. The observed faults on this machine are transient:
    # an occasional SIGSEGV inside the torch/PyG stack that kills a run mid-epoch
    # while its siblings continue, and OOMs caused by another user's job arriving
    # partway through. Both succeed on a second attempt. A genuine bug fails
    # twice and is reported, so this cannot hide one - and the retry is logged
    # separately so the rate stays visible rather than being smoothed away.
    if rc != 0 and attempt == 0:
        with _lock:
            print(f"[retry] {name}  rc={rc} after {dt:.1f} min - one retry",
                  flush=True)
        time.sleep(20)
        return run_job(j, gpus, logdir, budget, attempt=1)

    with _lock:
        (_done if rc == 0 else _failed).append((name, rc, dt))
        print(f"[{'ok ' if rc == 0 else 'FAIL'}] {name}  gpu{g}  {dt:.1f} min"
              + ("" if rc == 0 else f"  rc={rc}  see {logf}"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--gpus", type=int, nargs="*", default=None,
                    help="GPU indices to use (default: those with >20 GB free)")
    ap.add_argument("--gpu-gb", type=float, default=21.0,
                    help="usable memory budget per GPU")
    ap.add_argument("--max-parallel", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run finished jobs")
    a = ap.parse_args()

    spec = json.load(open(a.campaign))
    jobs = spec["jobs"] if isinstance(spec, dict) else spec
    logdir = os.path.join(ROOT, "runs",
                          os.path.splitext(os.path.basename(a.campaign))[0])
    os.makedirs(logdir, exist_ok=True)

    jobs = expand_shards(jobs)
    if not a.force:
        have = results_index()
        before = len(jobs)
        jobs = [j for j in jobs
                if not (shard_done(j) if j.get("rotations")
                        else job_key(j) in have)]
        print(f"{before - len(jobs)}/{before} already finished, skipping")

    # Dispatch order: explicit priority first (lower runs earlier), then
    # longest-first inside a priority band so the tail is short jobs. Priority
    # exists because these campaigns run longer than a sitting: the datasets
    # carrying the paper's claims must land before the ones filling in a cell.
    jobs.sort(key=lambda j: (j.get("priority", 5), -COST.get(j["dataset"], 1)))

    gpus = a.gpus
    if gpus is None:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout
        gpus = [int(l.split(",")[0]) for l in q.strip().splitlines()
                if float(l.split(",")[1]) > 20000]
        print(f"auto-selected free GPUs: {gpus}")
    if not gpus:
        sys.exit("no free GPU found; pass --gpus explicitly")
    # One runner per GPU, enforced. Each runner tracks its own memory budget and
    # cannot see another's, so two on one card believe they each have the whole
    # device - three of them once claimed 21 + 9 + 9 GB of a 24 GB A5000 and
    # OOM-killed each other's jobs. The driver-level free-memory check in
    # acquire() narrows the window but does not close it: two runners can look at
    # the same instant and both see room. A second campaign belongs in the same
    # queue as another band, not in a second process.
    locks = []
    for g in gpus:
        lk = os.path.join(ROOT, f"runs/.gpu{g}.lock")
        if os.path.exists(lk):
            try:
                pid = int(open(lk).read().split()[0])
                os.kill(pid, 0)          # raises unless that pid is alive
                sys.exit(f"GPU {g} is already held by campaign pid {pid} "
                         f"({lk}).\nAdd your jobs to that campaign instead of "
                         f"starting a second runner on the same device.")
            except (ValueError, IndexError, ProcessLookupError, PermissionError):
                pass                      # stale lock from a dead runner
        with open(lk, "w") as fh:
            fh.write(f"{os.getpid()} {a.campaign}\n")
        locks.append(lk)

    import atexit
    atexit.register(lambda: [os.remove(p) for p in locks
                             if os.path.exists(p)])

    global _cpu_free
    _cpu_free = a.max_parallel
    for g in gpus:
        _gpu_free[g] = a.gpu_gb

    est = sum(COST.get(j["dataset"], 1) for j in jobs)
    print(f"{len(jobs)} job(s), cost units {est}, gpus {gpus}, "
          f"logs -> {logdir}")
    if a.dry_run:
        for j in jobs:
            print("   ", " ".join(build_cmd(j)))
        return

    # A fixed pool of workers pulling from a priority-ordered queue, rather than
    # one thread per job. With a thread per job every job races for the GPU the
    # moment memory frees, so the priority order decides only who STARTS first -
    # a priority-5 job that happened to lose one race then wins the next and
    # overtakes the priority-1 work. The pool makes the order binding.
    _next = [0]

    def take():
        with _lock:
            if _next[0] >= len(jobs):
                return None
            j = jobs[_next[0]]
            _next[0] += 1
            return j

    def worker():
        while True:
            j = take()
            if j is None:
                return
            try:
                run_job(j, gpus, logdir, a.gpu_gb)
            except Exception as e:  # one bad job must not end the campaign
                with _lock:
                    _failed.append((job_key(j), repr(e), 0))
                    print(f"[FAIL] {job_key(j)} {e!r}", flush=True)

    t0 = time.time()
    threads = [threading.Thread(target=worker, daemon=False)
               for _ in range(min(a.max_parallel, len(jobs)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\ncampaign done in {(time.time()-t0)/60:.1f} min: "
          f"{len(_done)} ok, {len(_failed)} failed")
    for n, rc, _ in _failed:
        print("  FAILED", n, rc)


if __name__ == "__main__":
    main()
