"""Record the exact environment the results were produced in.

A results table without the environment behind it is not reproducible in the
sense a referee means: "we could not reproduce your numbers" and "we ran a
different BLAS" are indistinguishable after the fact. This writes the versions,
the hardware, the protocol constants, and a content hash of every source file
that can change a number, so a later run can be compared against it rather than
guessed at.

    python analysis/scripts/record_environment.py
"""

import hashlib
import json
import os
import platform
import subprocess
import time
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

# the files whose contents can change a reported number
SOURCES = ["main.py", "trainer.py", "model.py", "losses.py", "utils.py",
           "results_writer.py", "config.json"]


def sha(path):
    if not os.path.exists(path):
        return None
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ("torch", "torch_geometric", "numpy", "scipy", "sklearn",
              "pandas", "ogb"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = None
    try:
        import torch
        out["cuda"] = torch.version.cuda
        out["cudnn"] = torch.backends.cudnn.version()
        out["gpus"] = [torch.cuda.get_device_name(i)
                       for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            out["gpu_memory_gb"] = round(p.total_memory / 1e9, 1)
    except Exception as e:
        out["cuda_error"] = repr(e)
    try:
        out["nvidia_driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip().splitlines()[0]
    except Exception:
        pass
    out["cpu_count"] = os.cpu_count()
    return out


def main():
    env = {
        "versions": versions(),
        "source_sha256_16": {f: sha(f) for f in SOURCES},
        "protocol": {
            "train_anomalies_from_one_class": 50,
            "train_normal_ratio": 0.05,
            "val_anomalies": 30,
            "val_normal_ratio": 0.01,
            "epochs_small": 200, "epochs_large": 400,
            "seeds": [42, 0, 1, 2, 3],
            "note": "--seed sets the split AND the initialisation, so the "
                    "spread reported across seeds is split-and-initialisation "
                    "variance. --train_seed varies initialisation at a fixed "
                    "split and reports the narrower of the two.",
        },
        "execution_invariants_checked": [
            "rotation sharding (main.py --rotations) is bit-identical to a "
            "whole run",
            "features_on_gpu is bit-identical to gathering on the host",
        ],
    }
    os.makedirs("results", exist_ok=True)

    # Append to a history as well as overwriting the current snapshot. Writing
    # only the snapshot destroys the evidence it exists to preserve: torch was
    # upgraded from 2.0.1+cu117 to 2.7.1+cu126 partway through this campaign, and
    # because refresh_all.sh re-runs this script, the record of the ORIGINAL
    # environment was silently replaced by the new one. Six result cells have
    # seeds on both sides of that upgrade. A snapshot that overwrites itself
    # cannot tell you that; a history can.
    hist = "results/environment_history.jsonl"
    prev = None
    if os.path.exists(hist):
        lines = [l for l in open(hist) if l.strip()]
        if lines:
            prev = json.loads(lines[-1]).get("versions", {})
    if prev != env["versions"]:
        with open(hist, "a") as fh:
            fh.write(json.dumps({"recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
                                 **env}) + "\n")

    json.dump(env, open("results/environment.json", "w"), indent=1)
    print(json.dumps(env, indent=1))
    print("\n-> results/environment.json")


if __name__ == "__main__":
    main()
