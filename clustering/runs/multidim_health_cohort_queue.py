"""Multidimensional theta fit with birth-cohort shifts, full sample then held out.

AR(1) plus "spike" with Gompertz-Makeham mortality on theta and the mental
GRM, as theta-ssm-mort and theta-ssm-mort-ho, plus birth-decade level shifts
on both Gaussian channels that are common to the classes (the univariate
cohort fits' cohort_by_class 0, prior scale 0.15, oldest decade = 0). The
held-out fit holds each person's last two rows out (people with at least
five rows) and writes about 10 GB of chain csvs.

    python clustering/runs/multidim_health_cohort_queue.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "multidim-health"

JOBS = [
    ("theta-ssm-mort-cohort", "ssm", "theta",
     ["--cohort", "decade"]),
    ("theta-ssm-mort-cohort-ho", "ssm-holdout", "theta",
     ["--cohort", "decade", "--holdout-last-k", "2", "--min-obs", "5"]),
]


def run_job(job, settings):
    tag, variant, physical, extra = job
    out_dir = OUT / tag
    if (out_dir / "run_summary.json").exists():
        print(f"[{time.strftime('%H:%M:%S')}] SKIP {tag} (already finished)", flush=True)
        return tag, 0, 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "clustering" / "run_multidim.py"),
           "--variant", variant, "--physical", physical, "--with-mortality",
           "--output", str(out_dir)] + extra + settings
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
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--sampling", type=int, default=1000)
    ap.add_argument("--threads-per-chain", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    settings = ["--chains", str(a.chains), "--parallel-chains", str(a.parallel_chains),
                "--warmup", str(a.warmup), "--sampling", str(a.sampling),
                "--threads-per-chain", str(a.threads_per_chain)]
    if a.dry_run:
        for tag, variant, physical, extra in JOBS:
            print(f"{tag:26s} variant={variant:12s} physical={physical} {' '.join(extra)}")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = [run_job(j, settings) for j in JOBS]
    digest = {"total_hours": (time.time() - t0) / 3600,
              "jobs": [{"tag": t, "rc": rc, "hours": h} for t, rc, h in results]}
    for t, _, _ in results:
        f = OUT / t / "run_summary.json"
        if f.exists():
            j = json.loads(f.read_text())
            digest[t] = {k: j.get(k) for k in ("max_structural_rhat", "worst_param", "wall_hours", "n_person", "held_rows",
                                               "heldout_ar_conditional_mean", "heldout_class_only_mean")}
    (OUT / "digest_cohort.json").write_text(json.dumps(digest, indent=1))
    ok = all(rc == 0 for _, rc, _ in results)
    print(f"\nCOHORT QUEUE COMPLETE in {digest['total_hours']:.1f} h; {'all ok' if ok else 'FAILURES'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
