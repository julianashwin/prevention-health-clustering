"""K = 4 and K = 5 on the paper's health measure, one fit at a time.

theta and h, each under independent residuals (ar_mode 0) and under AR(1)
plus measurement error (ar_mode 2). Every spec is its K = 3 counterpart with
only the class count changed (models/registry.py, HEALTH_K45), on the same
contract, so the comparison across K is clean. The deficit index is left out:
it has tracked h in every fit so far.

Order: all four K = 4 fits, independent residuals first because they are
cheapest, then the same four at K = 5. Warmup is 1,500 rather than 1,000, as
for the earlier K = 4 fits, because more classes means more adjacent pairs the
ordered anchor intercept has to separate.

The queue skips any fit whose run_summary.json already exists, so after a
reboot relaunching it loses only the fit that was running (and that one can be
recovered with clustering/salvage_partial_fit.py).

    python clustering/runs/health_k45_queue.py [--dry-run]
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
OUT = ROOT / "artifacts" / "health-k45"

JOBS = [f"health-{v}-{s}-k{k}" for k in (4, 5) for s in ("base", "ssm") for v in ("theta", "h")]


def run_job(model, settings):
    out_dir = OUT / model
    if (out_dir / "run_summary.json").exists():
        print(f"[{time.strftime('%H:%M:%S')}] SKIP {model} (already finished)", flush=True)
        return model, 0, 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "clustering" / "run_fit.py"), "--model", model,
           "--contract", str(CONTRACT), "--output", str(out_dir)] + settings
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START {model}", flush=True)
    with open(out_dir / "run.log", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    hours = (time.time() - t0) / 3600
    print(f"[{time.strftime('%H:%M:%S')}] {'DONE' if rc == 0 else f'FAILED rc={rc}'} {model} ({hours:.2f} h)", flush=True)
    return model, rc, hours


def summarise(model):
    summ = OUT / model / "run_summary.json"
    if not summ.exists():
        return None
    r = json.loads(summ.read_text())
    p = r["params"]
    k = int(model[-1])
    get = lambda name: [round(p[f"{name}[{i}]"]["mean"], 3) for i in range(1, k + 1)] if f"{name}[1]" in p else None
    return {"max_rhat": r["max_structural_rhat"], "worst": r.get("worst_param"),
            "theta": get("theta"), "rho": get("rho"),
            "sigma_meas": round(p["sigma_meas[1]"]["mean"], 3) if "sigma_meas[1]" in p else None,
            "wall_hours": round(r["wall_hours"], 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--parallel-chains", type=int, default=4)
    ap.add_argument("--threads-per-chain", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--sampling", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    settings = ["--chains", str(a.chains), "--parallel-chains", str(a.parallel_chains),
                "--threads-per-chain", str(a.threads_per_chain),
                "--warmup", str(a.warmup), "--sampling", str(a.sampling)]
    if a.dry_run:
        for m in JOBS:
            print(m, CONTRACT.name, "skip" if (OUT / m / "run_summary.json").exists() else "run")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = []
    for m in JOBS:
        results.append(run_job(m, settings))
        digest = {"total_hours": (time.time() - t0) / 3600,
                  "jobs": [{"model": t, "rc": rc, "hours": h} for t, rc, h in results],
                  "summaries": {t: summarise(t) for t, _, _ in results}}
        (OUT / "digest.json").write_text(json.dumps(digest, indent=1))   # updated after every fit
    failed = [t for t, rc, _ in results if rc != 0]
    print(f"QUEUE COMPLETE in {(time.time() - t0) / 3600:.1f} h; "
          f"{'all ok' if not failed else 'FAILED: ' + ', '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
