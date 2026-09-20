#!/usr/bin/env python3
"""Sequential driver for small jobs that must share a card with a big campaign.

run_campaign.py deliberately refuses to start a second runner on a GPU another
campaign holds, and that refusal is right: two schedulers doing their own budget
accounting will both think there is room.  This script does not do budget
accounting and does not take a lock.  It runs ONE job at a time and, before each
one, waits until the driver itself reports enough free memory, with a headroom
margin on top of the job's declared need.  One job of a few GB against a card
whose big tenant peaks predictably is a different risk from a second scheduler.

It reuses run_campaign's expand/shard_done/build_cmd so the jobs, shard names and
merge behaviour are identical to a normal campaign.
"""
import argparse, json, os, socket, subprocess, sys, time, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "analysis", "scripts"))
import run_campaign as rc


def log(msg, fh):
    line = f"[{datetime.datetime.utcnow():%H:%M:%S}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n"); fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--headroom-gb", type=float, default=4.0,
                    help="free memory required BEYOND the job's declared need")
    ap.add_argument("--threads", type=int, default=4,
                    help="cap OMP/MKL/torch thread pools per job")
    ap.add_argument("--reverse", action="store_true",
                    help="consume the queue from the tail, so a driver started "
                         "later meets a head-first driver in the middle instead "
                         "of duplicating its next job")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    path = a.campaign if os.path.exists(a.campaign) else \
        os.path.join(ROOT, "campaigns", a.campaign + ".json")
    jobs = rc.expand_shards(json.load(open(path))["jobs"])
    if a.force:
        todo = jobs
    else:
        have = rc.results_index()
        todo = [j for j in jobs
                if not (rc.shard_done(j) if j.get("rotations")
                        else rc.job_key(j) in have)]
    todo.sort(key=lambda j: (j.get("priority", 5), -rc.COST.get(j["dataset"], 1)))
    if a.reverse:
        todo.reverse()
    print(f"{len(todo)}/{len(jobs)} job(s) to run on gpu {a.gpu}")
    if a.dry_run:
        for j in todo:
            print("  ", " ".join(rc.build_cmd(j)))
        return

    stem = os.path.basename(path).replace(".json", "")
    logdir = os.path.join(ROOT, "runs", stem)
    os.makedirs(logdir, exist_ok=True)
    claimdir = os.path.join(logdir, "claims")
    os.makedirs(claimdir, exist_ok=True)
    ok = fail = 0
    with open(os.path.join(logdir, "driver.log"), "a") as fh:
        log(f"=== run_small {stem} on gpu {a.gpu}: {len(todo)} job(s) ===", fh)
        for n, j in enumerate(todo, 1):
            need = (j.get("mem_gb") or rc.MEM_GB.get(
                (j["dataset"], j.get("method", "outpost")), 12.0)) + a.headroom_gb
            waited = 0
            while rc.real_free_gb(a.gpu) < need:
                time.sleep(30); waited += 30
                if waited % 600 == 0:
                    log(f"  waiting {waited//60}m for {need:.0f} GB on gpu {a.gpu}", fh)
            name = (f"{j['tag']}__{j['dataset']}_{j.get('method','outpost')}"
                    f"_s{j['seed']}_rot{j.get('rotations',[''])[0]}")
            # Several drivers may share one campaign across cards. Claim a job by
            # creating its marker exclusively; whoever loses the race skips it.
            # O_EXCL on a fresh path is atomic on this filesystem, so two drivers
            # cannot both run the same seed and race to write the same shard.
            claim = os.path.join(claimdir, name + ".claim")
            HOST = socket.gethostname()
            try:
                fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} gpu{a.gpu} {HOST}\n".encode())
                os.close(fd)
            except FileExistsError:
                # PIDs are per-machine. The claims directory lives on the shared
                # NAS, so a claim can belong to a driver on ANOTHER node, and
                # os.kill(pid, 0) here would probe an unrelated or absent local
                # process. Only a same-host claim can be liveness-checked.
                try:
                    fields = open(claim).read().split()
                except OSError:
                    continue
                owner_host = fields[2] if len(fields) > 2 else None
                if owner_host and owner_host != HOST:
                    continue                     # another node's driver: trust it
                if owner_host is None:
                    # written by a pre-host driver; cannot tell where it lives.
                    # Young claims are presumed live (a job runs hours); old ones
                    # with no shard are presumed abandoned.
                    try:
                        age_h = (time.time() - os.path.getmtime(claim)) / 3600
                    except OSError:
                        continue
                    if age_h < 24:
                        continue
                # A claim whose owner is gone is a lie, not a reservation. Drivers
                # die with their session and leave the whole queue claimed; on
                # 2026-09-13 sixty-three stale claims survived a five-day gap and
                # would have blocked every retry if any job had still been missing.
                # Same staleness rule run_campaign uses for its GPU locks.
                try:
                    owner = int(open(claim).read().split()[0])
                    os.kill(owner, 0)            # raises unless that pid is alive
                    continue                     # a live driver really has it
                except PermissionError:
                    # the pid exists but belongs to another user - that is ALIVE,
                    # not stale. Treating it as stale made a stand-down claim
                    # owned by pid 1 get reclaimed immediately on 2026-09-14.
                    continue
                except (ValueError, IndexError, ProcessLookupError, OSError):
                    log(f"  reclaiming stale claim {name}", fh)
                    try:
                        os.remove(claim)
                        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.write(fd, f"{os.getpid()} gpu{a.gpu}\n".encode())
                        os.close(fd)
                    except (FileExistsError, OSError):
                        continue                 # someone beat us to it
            # On a saturated box the default thread pools are the problem, not
            # the fix: torch sizes OMP/MKL to the whole machine, so a dozen
            # co-tenant jobs each spawn 48 workers and the box spends its time
            # in futex_wait rather than in kernels. Cap our own pools so this
            # driver is a good citizen and gets more done, not less.
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu),
                       PYTHONPATH=os.path.join(ROOT, ".stubs"),
                       OMP_NUM_THREADS=str(a.threads),
                       MKL_NUM_THREADS=str(a.threads),
                       OPENBLAS_NUM_THREADS=str(a.threads),
                       NUMEXPR_NUM_THREADS=str(a.threads),
                       TORCH_NUM_THREADS=str(a.threads),
                       # A launch-time free-memory check cannot see co-tenants
                       # that grow AFTER we start. Expandable segments let the
                       # allocator return fragmented blocks instead of holding
                       # them, which is what turned a 16 MiB request into an OOM
                       # on a card with six tenants on 2026-09-08.
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            t0 = time.time()
            with open(os.path.join(logdir, name + ".log"), "w") as jl:
                r = subprocess.run(rc.build_cmd(j), cwd=ROOT, env=env,
                                   stdout=jl, stderr=subprocess.STDOUT)
            mins = (time.time() - t0) / 60
            if r.returncode == 0:
                ok += 1;   log(f"[ok  ] {n}/{len(todo)} {name}  {mins:.1f} min", fh)
            else:
                fail += 1; log(f"[FAIL] {n}/{len(todo)} {name}  rc={r.returncode}", fh)
                # Preserve the evidence. The job log is opened "w", so a retry
                # overwrites it and the reason the first attempt died is gone -
                # on 2026-09-13 eleven recovered failures could no longer be
                # diagnosed because every log had been replaced by its successful
                # rerun. Keep each failure under its own name.
                try:
                    k = 1
                    while os.path.exists(os.path.join(logdir, f"{name}.failed{k}.log")):
                        k += 1
                    os.replace(os.path.join(logdir, name + ".log"),
                               os.path.join(logdir, f"{name}.failed{k}.log"))
                except OSError:
                    pass
                try:
                    os.remove(claim)      # failed: let another driver retry it
                except OSError:
                    pass
        log(f"=== {stem} done: {ok} ok, {fail} failed ===", fh)


if __name__ == "__main__":
    main()
