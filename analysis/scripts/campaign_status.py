#!/usr/bin/env python3
"""Live progress for run_small campaigns, with liveness proven per job.

Written after reading a dead job's log at 02:35 and reporting "epoch 102" as
current progress: the file's last write was 01:53 and it ended in an OOM
traceback. A log says what a process DID, never that it is still running. Every
row here carries the log's age and whether a matching process exists, so a
stalled or dead job cannot be mistaken for a running one.

    python analysis/scripts/campaign_status.py [campaign ...]
"""
import glob, json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
STALE_S = 300          # a live training job writes an epoch line far more often


def running_jobs():
    """(dataset, seed, tag) of every main.py / run_nsreg.py process alive now."""
    out = subprocess.run(["ps", "-eo", "cmd", "--no-headers"],
                         capture_output=True, text=True).stdout
    live = set()
    for line in out.splitlines():
        if "main.py --dataset" not in line and "run_nsreg.py --dataset" not in line:
            continue
        ds = re.search(r"--dataset (\S+)", line)
        sd = re.search(r"--seed (\S+)", line)
        tg = re.search(r"--tag (\S+)", line)
        # The rotation belongs in the key. Without it, one running rotation makes
        # every sibling rotation of the same (dataset, seed, tag) look alive, and
        # the stall check then fires on jobs that simply finished - a false alarm
        # is the same failure as a missed one, pointed the other way.
        rt = re.search(r"--rotations (\S+)", line)
        if ds and sd and tg:
            live.add((ds.group(1), sd.group(1), tg.group(1),
                      rt.group(1) if rt else ""))
    return live


def main():
    camps = sys.argv[1:] or [os.path.basename(d) for d in sorted(glob.glob("runs/*"))
                             if os.path.isdir(d) and os.path.exists(d + "/driver.log")]
    live = running_jobs()
    now = time.time()
    for c in camps:
        logs = [f for f in sorted(glob.glob(f"runs/{c}/*.log"))
                if not f.endswith("driver.log")]
        if not logs:
            continue
        done = fails = 0
        try:
            dl = open(f"runs/{c}/driver.log", encoding="utf-8", errors="replace").read()
            done = dl.count("[ok  ]")
            fails = dl.count("[FAIL]")
        except OSError:
            pass
        print(f"\n=== {c}   {done} ok, {fails} failed ===")
        for f in logs:
            stem = os.path.basename(f)[:-4]
            m = re.match(r"(.+?)__(.+?)_(outpost|demo|nsreg)_s(\S+?)_rot(\S*)$", stem)
            key = (m.group(2), m.group(4), m.group(1), m.group(5)) if m else None
            age = now - os.path.getmtime(f)
            try:
                tail = open(f, encoding="utf-8", errors="replace").read()[-4000:]
            except OSError:
                continue
            ep = re.findall(r"(?:epoch|Epoch:) (\d+)", tail)
            oom = "OutOfMemoryError" in tail
            state = ("LIVE" if key in live else
                     ("OOM/DEAD" if oom else "finished-or-dead"))
            flag = "" if key in live else "   <-- not running"
            print(f"  {stem[:52]:54s} epoch {ep[-1] if ep else '?':>4}  "
                  f"log {age/60:5.1f} min old  {state}{flag}")
        stale = [f for f in logs
                 if (now - os.path.getmtime(f)) > STALE_S
                 and (lambda m: (m.group(2), m.group(4), m.group(1), m.group(5))
                      if m else None)(
                     re.match(r"(.+?)__(.+?)_(outpost|demo|nsreg)_s(\S+?)_rot(\S*)$",
                              os.path.basename(f)[:-4])) in live]
        for f in stale:
            print(f"  STALL: {os.path.basename(f)} process alive but log silent "
                  f"{(now - os.path.getmtime(f))/60:.0f} min")


if __name__ == "__main__":
    main()
