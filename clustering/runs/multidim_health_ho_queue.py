"""Held-out multidimensional fits: AR(1) plus "spike" with mortality, theta and h.

The two state-space fits of multidim_health_queue.py refitted with each
person's last two observed rows held out (people with at least five rows),
the Gaussian channels scored on the held rows conditional on the fitted
window (ar_conditional) and on the class alone (class_only). Mortality uses
the full window and is never held out. Person-level quantities are emitted,
so each fit writes about 10 GB of chain csvs.

    python clustering/runs/multidim_health_ho_queue.py [--dry-run]
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
    ("theta-ssm-mort-ho", "ssm-holdout", "theta"),
    ("h-ssm-mort-ho", "ssm-holdout", "h"),
]


def run_job(job, settings):
    tag, variant, physical = job
    out_dir = OUT / tag
    if (out_dir / "run_summary.json").exists():
        print(f"[{time.strftime('%H:%M:%S')}] SKIP {tag} (already finished)", flush=True)
        return tag, 0, 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "clustering" / "run_multidim.py"),
           "--variant", variant, "--physical", physical, "--with-mortality",
           "--holdout-last-k", "2", "--min-obs", "5",
           "--output", str(out_dir)] + settings
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
        for tag, variant, physical in JOBS:
            print(f"{tag:18s} variant={variant:12s} physical={physical}")
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
    (OUT / "digest_ho.json").write_text(json.dumps(digest, indent=1))
    ok = all(rc == 0 for _, rc, _ in results)
    print(f"\nHO QUEUE COMPLETE in {digest['total_hours']:.1f} h; {'all ok' if ok else 'FAILURES'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
