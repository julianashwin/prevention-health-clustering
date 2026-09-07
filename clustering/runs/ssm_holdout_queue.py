"""Second state-space batch: the instrument control, and three holdout twins.

Four fits, two at a time, longest first.

  physfunc-control   P-FUNC on the P-FULL ROSTER, no holdout. The control for
                     section 9's instrument comparison. Its contract carries
                     the exact person-age cells of the P-FULL contract
                     (38,963 people, 334,194 rows), so the only difference
                     from physfull-ssm is which measure sits on the channel
                     and sigma_meas is directly comparable.

  *-ssm-ho           the three AR(1)+measurement-error fits with each person's
                     last two observations held out, sample restricted to
                     people observed at least five times. These answer whether
                     separating the noise actually improves prediction, which
                     the first batch could not.

Held-out scoring under ar_mode 2 filters over the whole fitted history rather
than conditioning on one noisy last value; that path was validated against an
offline Kalman filter to a correlation of 0.99995
(clustering/validation/ssm_holdout_check.py).

    python clustering/runs/ssm_holdout_queue.py [--dry-run]
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
OUT = ROOT / "artifacts" / "ssm2"
HO = ["--holdout-last-k", "2", "--holdout-min-obs", "5"]

# (tag, model, contract, extra args) - longest expected first
JOBS = [
    ("grm-ssm-ho", "grm-ssm", CONTRACTS / "grm_lifecycle_20_89_minobs3_v1", HO),
    ("physfunc-ssm-ho", "physgrm-func-ssm",
     CONTRACTS / "physfunc_lifecycle_20_89_minobs3_v1", HO),
    ("physfunc-control", "physgrm-func-ssm",
     CONTRACTS / "physfunc_on_full_roster_v1", []),
    ("physfull-ssm-ho", "physgrm-full-ssm",
     CONTRACTS / "physgrm_lifecycle_20_89_minobs3_v1", HO),
]


def run_job(job, settings):
    tag, model, contract, extra = job
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "clustering" / "run_fit.py"),
           "--model", model, "--contract", str(contract),
           "--output", str(out_dir)] + settings + extra
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
    f = OUT / tag / "run_summary.json"
    if not f.exists():
        return {}
    j = json.loads(f.read_text())
    par = j.get("params", {})
    get = lambda n: par[n]["mean"] if n in par else None
    rho = [get(f"rho[{k}]") for k in (1, 2, 3)]
    sig, mea = get("sigma[1,1]"), get("sigma_meas[1]")
    out = {"max_rhat": j.get("max_structural_rhat"),
           "wall_hours": j.get("wall_hours"), "n_person": j.get("n_person"),
           "n_obs": j.get("n_obs"), "rho": rho, "sigma": sig,
           "sigma_meas": mea,
           "theta": [get(f"theta[{k}]") for k in (1, 2, 3)],
           "heldout_mean_lpd": j.get("heldout_mean_lpd")}
    if sig and mea and all(rho):
        out["signal_share"] = [(sig**2 / (1 - r**2))
                               / (sig**2 / (1 - r**2) + mea**2) for r in rho]
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
        for tag, model, contract, extra in JOBS:
            ok = (contract / "long.csv").exists()
            print(f"{tag:18s} {model:20s} {contract.name:36s} "
                  f"{'holdout' if extra else 'full   '} "
                  f"{'OK' if ok else 'CONTRACT MISSING'}")
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
            lpd = (f"  heldout LPD {s['heldout_mean_lpd']:.4f}"
                   if s.get("heldout_mean_lpd") is not None else "")
            print(f"  {t:18s} sigma_meas {s['sigma_meas']:.4f}  "
                  f"Rhat {s['max_rhat']:.4f}{lpd}")
    return 0 if all(rc == 0 for _, rc, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
