"""K=4 under the state-space specification, for P-FUNC and P-FULL.

Everything matches the K=3 state-space fits except the number of classes, so
the comparison is a clean one: does a fourth class earn its place once
measurement error has been separated from persistence?

More classes means more adjacent pairs the ordered anchor intercept has to
keep apart, and a pair pinned against that constraint is exactly the mode
split found in the multidim smoke. So these are watched rather than assumed:
run k4_mode_check on a live run to see the per-chain intercept gaps.

Runs one fit at a time with a capped CPU footprint, alongside the multidim
batch in its own screen session.

    python clustering/runs/k4_queue.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "data" / "processed" / "contracts"
OUT = ROOT / "artifacts" / "k4"

JOBS = [
    ("physfull-ssm-k4", "physgrm-full-ssm-k4",
     CONTRACTS / "physgrm_lifecycle_20_89_minobs3_v1"),
    ("physfunc-ssm-k4", "physgrm-func-ssm-k4",
     CONTRACTS / "physfunc_lifecycle_20_89_minobs3_v1"),
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


def summarise(tag, k=4):
    f = OUT / tag / "run_summary.json"
    if not f.exists():
        return {}
    j = json.loads(f.read_text())
    par = j.get("params", {})
    get = lambda n: par[n]["mean"] if n in par else None
    return {"max_rhat": j.get("max_structural_rhat"),
            "worst_param": j.get("worst_param"),
            "wall_hours": j.get("wall_hours"), "n_person": j.get("n_person"),
            "theta": [get(f"theta[{i}]") for i in range(1, k + 1)],
            "rho": [get(f"rho[{i}]") for i in range(1, k + 1)],
            "alpha": [get(f"coef[1,{i},1]") for i in range(1, k + 1)],
            "sigma": get("sigma[1,1]"), "sigma_meas": get("sigma_meas[1]")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--parallel-chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--sampling", type=int, default=1000)
    ap.add_argument("--threads-per-chain", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    settings = ["--chains", str(a.chains),
                "--parallel-chains", str(a.parallel_chains),
                "--warmup", str(a.warmup), "--sampling", str(a.sampling),
                "--threads-per-chain", str(a.threads_per_chain)]
    if a.dry_run:
        for tag, model, contract in JOBS:
            ok = (contract / "long.csv").exists()
            print(f"{tag:18s} {model:22s} {contract.name:36s} "
                  f"{'OK' if ok else 'CONTRACT MISSING'}")
        print(f"peak cores: {a.parallel_chains} x {a.threads_per_chain} = "
              f"{a.parallel_chains * a.threads_per_chain}")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = [run_job(j, settings) for j in JOBS]   # strictly one at a time
    digest = {"total_hours": (time.time() - t0) / 3600,
              "jobs": [{"tag": t, "rc": rc, "hours": h} for t, rc, h in results],
              "summaries": {t: summarise(t) for t, _, _ in results}}
    (OUT / "digest.json").write_text(json.dumps(digest, indent=1))
    print(f"\ntotal {digest['total_hours']:.2f} h -> {OUT / 'digest.json'}")
    for t, s in digest["summaries"].items():
        if s:
            print(f"  {t:18s} Rhat {s['max_rhat']:.4f} ({s['worst_param']})")
            print(f"      theta {[round(x, 3) for x in s['theta']]}")
            print(f"      rho   {[round(x, 3) for x in s['rho']]}")
            print(f"      alpha {[round(x, 3) for x in s['alpha']]}  "
                  f"sigma_meas {s['sigma_meas']:.4f}")
    return 0 if all(rc == 0 for _, rc, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
