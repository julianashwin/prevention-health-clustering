"""AR(1) plus "spike" with birth-decade shifts, theta and h, first TWO rows held out.

The cohort twins of the first-two-rows held-out fits of frailty_firstheld_queue.py
(people with at least five rows), into artifacts/health-firstheld/, for the
type-from-an-initial-value exercise. Runs alongside that queue. Finished fits are
skipped, so the queue is safe to relaunch.

    python clustering/runs/cohort_first2_queue.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "data" / "processed" / "contracts" / "health_lifecycle_20_89_minobs3_v1"
ART = ROOT / "artifacts"
FIRST1 = ["--holdout-first-k", "1", "--holdout-min-obs", "4"]
FIRST2 = ["--holdout-first-k", "2", "--holdout-min-obs", "5"]

# (output dir, tag, model, extra args)
JOBS = [
    ("health-firstheld", "health-theta-ssm-cohort-first2ho", "health-theta-ssm-cohort", FIRST2),
    ("health-firstheld", "health-h-ssm-cohort-first2ho", "health-h-ssm-cohort", FIRST2),
]


def run_job(job, settings):
    group, tag, model, extra = job
    out_dir = ART / group / tag
    if (out_dir / "run_summary.json").exists():
        print(f"[{time.strftime('%H:%M:%S')}] SKIP {tag} (already finished)", flush=True)
        return tag, 0, 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "clustering" / "run_fit.py"), "--model", model,
           "--contract", str(CONTRACT), "--output", str(out_dir)] + extra + settings
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
        for group, tag, model, extra in JOBS:
            print(f"{group:18s} {tag:28s} {model:24s} {' '.join(extra)}")
        return 0
    t0 = time.time()
    results = [run_job(j, settings) for j in JOBS]
    digest = {"total_hours": (time.time() - t0) / 3600, "jobs": [{"tag": t, "rc": rc, "hours": h} for t, rc, h in results]}
    for group, tag, _, _ in JOBS:
        f = ART / group / tag / "run_summary.json"
        if f.exists():
            r = json.loads(f.read_text())
            digest.setdefault("summaries", {})[tag] = {"max_rhat": r["max_structural_rhat"], "wall_hours": round(r["wall_hours"], 2),
                                                        "theta": [round(r["params"][f"theta[{i}]"]["mean"], 4) for i in range(1, 4)]}
    (ART / "health-firstheld" / "digest_cohort_first2.json").write_text(json.dumps(digest, indent=1))
    failed = [t for t, rc, _ in results if rc != 0]
    print(f"COHORT-FIRST2 QUEUE COMPLETE in {digest['total_hours']:.1f} h; {'all ok' if not failed else 'FAILED: ' + ', '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
