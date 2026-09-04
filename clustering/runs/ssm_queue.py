"""Batch: three univariate K=3 fits under AR(1) state plus measurement error.

The original GRM theta, P-FUNC theta and P-FULL theta, each with ar_mode 2:
a latent health state following an AR(1) and an independent one-period
observation error on top. Two at a time, longest first.

The samples are NOT common across the three. P-FULL needs the diagnosed
condition inventory, which BHPS entrants never receive, so it runs on 38,963
people against roughly 50,000 for the other two. That is a property of the
measures, not of this batch, and it is why the three fits are compared on
parameters rather than on likelihoods.

    python clustering/runs/ssm_queue.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "data" / "processed" / "contracts"
OUT = ROOT / "artifacts" / "ssm"

# (tag, model, contract) - longest expected first
JOBS = [
    ("grm-ssm", "grm-ssm", CONTRACTS / "grm_lifecycle_20_89_minobs3_v1"),
    ("physfunc-ssm", "physgrm-func-ssm",
     CONTRACTS / "physfunc_lifecycle_20_89_minobs3_v1"),
    ("physfull-ssm", "physgrm-full-ssm",
     CONTRACTS / "physgrm_lifecycle_20_89_minobs3_v1"),
]


def run_job(job, settings):
    tag, model, contract = job
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "clustering" / "run_fit.py"),
           "--model", model, "--contract", str(contract),
           "--output", str(out_dir)] + settings
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START {tag}", flush=True)
    with open(out_dir / "run.log", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    hours = (time.time() - t0) / 3600
    print(f"[{time.strftime('%H:%M:%S')}] "
          f"{'DONE' if rc == 0 else f'FAILED rc={rc}'} {tag} ({hours:.2f} h)",
          flush=True)
    return tag, rc, hours


def summarise(tag):
    """Pull the parameters this batch exists to compare out of the summary."""
    f = OUT / tag / "run_summary.json"
    if not f.exists():
        return {}
    j = json.loads(f.read_text())
    par = j.get("params", {})
    get = lambda n: par[n]["mean"] if n in par else None
    out = {"max_rhat": j.get("max_structural_rhat"),
           "wall_hours": j.get("wall_hours"),
           "n_person": j.get("n_person"), "n_obs": j.get("n_obs"),
           "rho": [get(f"rho[{k}]") for k in (1, 2, 3)],
           "sigma": get("sigma[1,1]"),
           "sigma_meas": get("sigma_meas[1]"),
           "theta": [get(f"theta[{k}]") for k in (1, 2, 3)]}
    if out["sigma"] and out["sigma_meas"] and out["rho"][0]:
        # share of observation variance that is persistent signal, per class
        out["signal_share"] = [
            (out["sigma"] ** 2 / (1 - r ** 2))
            / (out["sigma"] ** 2 / (1 - r ** 2) + out["sigma_meas"] ** 2)
            for r in out["rho"] if r is not None]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--sampling", type=int, default=1000)
    ap.add_argument("--threads-per-chain", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    settings = ["--chains", str(a.chains), "--warmup", str(a.warmup),
                "--sampling", str(a.sampling),
                "--threads-per-chain", str(a.threads_per_chain)]
    if a.dry_run:
        for tag, model, contract in JOBS:
            ok = (contract / "long.csv").exists()
            print(f"{tag:16s} {model:20s} {contract.name:40s} "
                  f"{'contract OK' if ok else 'CONTRACT MISSING'}")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(lambda j: run_job(j, settings), JOBS))
    digest = {"total_hours": (time.time() - t0) / 3600,
              "jobs": [{"tag": t, "rc": rc, "hours": h} for t, rc, h in results],
              "summaries": {t: summarise(t) for t, _, _ in results}}
    (OUT / "digest.json").write_text(json.dumps(digest, indent=1))
    print(f"\ntotal {digest['total_hours']:.2f} h -> {OUT / 'digest.json'}")
    for t, s in digest["summaries"].items():
        if s:
            print(f"  {t:16s} rho {s['rho']}  sigma {s['sigma']:.4f}  "
                  f"sigma_meas {s['sigma_meas']:.4f}  Rhat {s['max_rhat']:.4f}")
    return 0 if all(rc == 0 for _, rc, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
