"""Multidimensional fits on the paper's measure: physical, mental, mortality.

Four K = 3 fits on the multidim health contract (38,181 people, the health
contract's people with a mental score on every row), one at a time:

  theta-ssm-mort   physical theta + mental GRM theta, AR(1) latent state plus
                   a one-period "spike" on both channels with class- and
                   channel-specific persistence, Gompertz-Makeham mortality
  h-ssm-mort       the same with h as the physical channel
  theta-base-mort  independent residuals, mortality on: the no-persistence
                   contrast
  h-base-mort      the same on h

The chronic count is left out: it is already an item of the physical measure.
Warmup is 1,500 as for the K = 4 and 5 fits. The queue skips any fit whose
run_summary.json exists, so relaunching after a reboot loses only the fit
that was running. Check a running state-space fit for a mode split with
clustering/runs/multidim_mode_check.py.

    python clustering/runs/multidim_health_queue.py [--dry-run]
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

# (tag, variant, physical channel)
JOBS = [
    ("theta-ssm-mort", "ssm", "theta"),
    ("h-ssm-mort", "ssm", "h"),
    ("theta-base-mort", "baseline", "theta"),
    ("h-base-mort", "baseline", "h"),
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
           "--output", str(out_dir)] + settings
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START {tag}", flush=True)
    with open(out_dir / "run.log", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    hours = (time.time() - t0) / 3600
    print(f"[{time.strftime('%H:%M:%S')}] {'DONE' if rc == 0 else f'FAILED rc={rc}'} {tag} ({hours:.2f} h)", flush=True)
    return tag, rc, hours


def summarise(tag):
    f = OUT / tag / "run_summary.json"
    if not f.exists():
        return {}
    j = json.loads(f.read_text())
    par = j.get("params", {})
    get = lambda n: par[n]["mean"] if n in par else None  # noqa: E731
    return {"max_rhat": j.get("max_structural_rhat"), "worst": j.get("worst_param"),
            "wall_hours": j.get("wall_hours"), "n_person": j.get("n_person"),
            "theta": [get(f"theta[{k}]") for k in (1, 2, 3)],
            "rho_phys": [get(f"rho[{k},1]") for k in (1, 2, 3)],
            "rho_ment": [get(f"rho[{k},2]") for k in (1, 2, 3)],
            "sigma_meas": [get(f"sigma_meas[{c}]") for c in (1, 2)],
            "log_b": [get(f"log_b[{k}]") for k in (1, 2, 3)],
            "gomp_slope": [get(f"gomp_slope[{k}]") for k in (1, 2, 3)],
            "makeham": get("makeham[1]"), "hazard_by_age": j.get("hazard_by_age")}


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
            print(f"{tag:16s} variant={variant:9s} physical={physical}")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = [run_job(j, settings) for j in JOBS]
    digest = {"total_hours": (time.time() - t0) / 3600,
              "jobs": [{"tag": t, "rc": rc, "hours": h} for t, rc, h in results],
              "summaries": {t: summarise(t) for t, _, _ in results}}
    (OUT / "digest.json").write_text(json.dumps(digest, indent=1))
    ok = all(rc == 0 for _, rc, _ in results)
    print(f"\nQUEUE COMPLETE in {digest['total_hours']:.1f} h; {'all ok' if ok else 'FAILURES'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
