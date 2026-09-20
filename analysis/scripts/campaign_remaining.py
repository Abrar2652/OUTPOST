#!/usr/bin/env python3
"""How many JOBS of a campaign remain. Not shards - jobs.

Written after a monitor reported "DONE sensitivity: 35/35" when the campaign was
at 28 of 35. It counted shard files against a job count, and Photo jobs shard into
two rotations each, so the numerator and denominator were different units. The
same mistake - counting the unit you happen to iterate over rather than the unit
the claim is about - has now appeared three times in this project.

    python analysis/scripts/campaign_remaining.py <campaign> [...]   # prints "done/total"
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
import run_campaign as rc


def status(name):
    path = name if os.path.exists(name) else f"campaigns/{name}.json"
    spec = json.load(open(path))
    jobs = rc.expand_shards(spec["jobs"] if isinstance(spec, dict) else spec)
    have = rc.results_index()
    miss = [j for j in jobs
            if not (rc.shard_done(j) if j.get("rotations") else rc.job_key(j) in have)]
    return len(jobs) - len(miss), len(jobs), miss


if __name__ == "__main__":
    for n in sys.argv[1:]:
        try:
            done, total, _ = status(n)
            print(f"{n} {done}/{total}")
        except Exception as e:
            print(f"{n} ERROR {type(e).__name__}")
