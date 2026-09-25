"""Held-out twins of the two baseline (independent-residual) K = 3 fits, theta and h.

Same held-out rule as the AR(1) plus one-period-noise twins in health_next_queue.py:
each person's last two observations are dropped from the likelihood and scored as a
conditional predictive density, on people observed at least five times. With these,
both specifications have a held-out fit on the same people and rows, which is what
the within-person version of the prediction exercise needs.

Runs after the K = 4/5 queue: the launcher waits for it to exit. Logs to
artifacts/health-next/<tag>/run.log, the same folder as the ssm-ho fits.

    python clustering/runs/health_base_ho_queue.py [--dry-run]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "data" / "processed" / "contracts" / "health_lifecycle_20_89_minobs3_v1"
OUT = ROOT / "artifacts" / "health-next"
HO = ["--holdout-last-k", "2", "--holdout-min-obs", "5"]
JOBS = [("health-theta-base-ho", "health-theta-base", HO), ("health-h-base-ho", "health-h-base", HO)]


def run_job(job, settings):
    tag, model, extra = job
    out_dir = OUT / tag
    if (out_dir / "run_summary.json").exists():
        print(f"[{time.strftime('%H:%M:%S')}] SKIP {tag} (already finished)", flush=True)
        return tag, 0, 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "clustering" / "run_fit.py"), "--model", model,
           "--contract", str(CONTRACT), "--output", str(out_dir)] + settings + extra
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START {tag}", flush=True)
    with open(out_dir / "run.log", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    hours = (time.time() - t0) / 3600
    print(f"[{time.strftime('%H:%M:%S')}] {'DONE' if rc == 0 else f'FAILED rc={rc}'} {tag} ({hours:.2f} h)", flush=True)
    return tag, rc, hours


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--parallel-chains", type=int, default=4)
    ap.add_argument("--threads-per-chain", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--sampling", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    settings = ["--chains", str(a.chains), "--parallel-chains", str(a.parallel_chains),
                "--threads-per-chain", str(a.threads_per_chain), "--warmup", str(a.warmup), "--sampling", str(a.sampling)]
    if a.dry_run:
        for tag, model, extra in JOBS:
            print(tag, model, " ".join(extra))
        return 0
    results = [run_job(j, settings) for j in JOBS]
    failed = [t for t, rc, _ in results if rc != 0]
    print(f"QUEUE COMPLETE; {'all ok' if not failed else 'FAILED: ' + ', '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
