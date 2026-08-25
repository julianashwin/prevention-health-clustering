"""Overnight batch: eight univariate K=3 fits, two at a time.

PCS: AR(1), AR(1)+holdout. Physical GRM (P-FULL theta) and combined GRM
theta: baseline, AR(1), AR(1)+holdout. Holdout = each person's last two
observations, sample restricted to people with at least five. Jobs run
longest-first on two workers; each job logs to its own artifacts directory
and the queue writes a digest when done.

    python clustering/runs/overnight_queue.py [--dry-run]
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
OUT = ROOT / "artifacts" / "overnight"
PCS = CONTRACTS / "pcs_lifecycle_20_89_minobs3_v1"
PHYS = CONTRACTS / "physgrm_lifecycle_20_89_minobs3_v1"
COMB = CONTRACTS / "combgrm_lifecycle_20_89_minobs3_v1"
HO = ["--holdout-last-k", "2", "--holdout-min-obs", "5"]

# (tag, model, contract, extra args) - longest expected first
JOBS = [
    ("pcs-ar1", "pcs-ar1", PCS, []),
    ("physgrm-ar1", "physgrm-ar1", PHYS, []),
    ("combgrm-ar1", "combgrm-ar1", COMB, []),
    ("pcs-ar1-ho", "pcs-ar1", PCS, HO),
    ("physgrm-ar1-ho", "physgrm-ar1", PHYS, HO),
    ("combgrm-ar1-ho", "combgrm-ar1", COMB, HO),
    ("physgrm-base", "physgrm-headline", PHYS, []),
    ("combgrm-base", "combgrm-headline", COMB, []),
]


def run_job(job, settings):
    tag, model, contract, extra = job
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "run.log"
    cmd = [sys.executable, str(ROOT / "clustering" / "run_fit.py"),
           "--model", model, "--contract", str(contract),
           "--output", str(out_dir)] + settings + extra
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START {tag}", flush=True)
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    hours = (time.time() - t0) / 3600
    status = "DONE" if rc == 0 else f"FAILED rc={rc}"
    print(f"[{time.strftime('%H:%M:%S')}] {status} {tag} ({hours:.2f} h)",
          flush=True)
    return tag, rc, hours


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--sampling", type=int, default=1000)
    parser.add_argument("--threads-per-chain", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    settings = ["--chains", str(args.chains), "--warmup", str(args.warmup),
                "--sampling", str(args.sampling),
                "--threads-per-chain", str(args.threads_per_chain)]
    if args.dry_run:
        for job in JOBS:
            print(job[0], job[1], job[2].name, job[3])
        return 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda j: run_job(j, settings), JOBS))
    digest = {
        "total_hours": (time.time() - t0) / 3600,
        "jobs": [{"tag": t, "rc": rc, "hours": h} for t, rc, h in results],
    }
    for tag, rc, hours in results:
        summ = OUT / tag / "run_summary.json"
        if summ.exists():
            r = json.loads(summ.read_text())
            digest.setdefault("summaries", {})[tag] = {
                "max_rhat": r["max_structural_rhat"],
                "theta": [round(r["params"][f"theta[{i}]"]["mean"], 4)
                          for i in range(1, 4)],
                "heldout_mean_lpd": r.get("heldout_mean_lpd"),
                "wall_hours": round(r["wall_hours"], 2),
            }
    (OUT / "digest.json").write_text(json.dumps(digest, indent=1))
    failed = [t for t, rc, _ in results if rc != 0]
    print(f"QUEUE COMPLETE in {digest['total_hours']:.1f} h; "
          f"{'all ok' if not failed else 'FAILED: ' + ', '.join(failed)}",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
