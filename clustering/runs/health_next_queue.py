"""The next batch on the paper's health contract, one fit at a time.

Two things the paper still takes from the archived P-FULL measure:

  AR(1) without a measurement-error term (ar_mode 1). The note's
  recommendation --- estimate persistence with AR(1) plus measurement error,
  not with AR(1) --- rests on a comparison run on P-FULL, where rho went from
  .78/.27/.39 to .95/.90/.86 once the one-period error was allowed. These
  three fits put the same comparison on the paper's own measure, on the same
  people and rows as the fits already in artifacts/health-ssm/.

  Held-out twins of the AR(1) plus measurement error fits: each person's last
  two observations are dropped from the likelihood and scored as a conditional
  predictive density, with the sample restricted to people observed at least
  five times. These replace the held-out table, which is still on P-FULL.

Cheapest and most load-bearing first, so an overnight run lands the AR(1)
comparison by morning. Logs to artifacts/health-next/<tag>/run.log; the queue
writes digest.json.

    python clustering/runs/health_next_queue.py [--dry-run]
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
OUT = ROOT / "artifacts" / "health-next"
HO = ["--holdout-last-k", "2", "--holdout-min-obs", "5"]

# (tag, model, extra args) -- cheapest and most load-bearing first
JOBS = [
    ("health-theta-ar1", "health-theta-ar1", []),
    ("health-h-ar1", "health-h-ar1", []),
    ("health-fi10-ar1", "health-fi10-ar1", []),
    ("health-theta-ssm-ho", "health-theta-ssm", HO),
    ("health-h-ssm-ho", "health-h-ssm", HO),
]


def run_job(job, settings):
    tag, model, extra = job
    out_dir = OUT / tag
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
                "--threads-per-chain", str(a.threads_per_chain),
                "--warmup", str(a.warmup), "--sampling", str(a.sampling)]
    if a.dry_run:
        for tag, model, extra in JOBS:
            print(tag, model, CONTRACT.name, " ".join(extra))
        return 0
    t0 = time.time()
    results = [run_job(j, settings) for j in JOBS]
    digest = {"total_hours": (time.time() - t0) / 3600,
              "jobs": [{"tag": t, "rc": rc, "hours": h} for t, rc, h in results]}
    for tag, rc, hours in results:
        summ = OUT / tag / "run_summary.json"
        if summ.exists():
            r = json.loads(summ.read_text())
            p = r["params"]
            digest.setdefault("summaries", {})[tag] = {
                "max_rhat": r["max_structural_rhat"],
                "theta": [round(p[f"theta[{i}]"]["mean"], 4) for i in range(1, 4)],
                "rho": [round(p[f"rho[{i}]"]["mean"], 3) for i in range(1, 4)] if "rho[1]" in p else None,
                "sigma": round(p["sigma[1,1]"]["mean"], 3),
                "sigma_meas": round(p["sigma_meas[1]"]["mean"], 3) if "sigma_meas[1]" in p else None,
                "heldout_mean_lpd": r.get("heldout_mean_lpd"),
                "wall_hours": round(r["wall_hours"], 2)}
    (OUT / "digest.json").write_text(json.dumps(digest, indent=1))
    failed = [t for t, rc, _ in results if rc != 0]
    print(f"QUEUE COMPLETE in {digest['total_hours']:.1f} h; {'all ok' if not failed else 'FAILED: ' + ', '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
