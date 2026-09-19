"""Baseline K=3 fits on the paper's health measure: theta, h and the deficit index.

Three registry models on the one contract that carries all three variants
(data_cleaning/07_build_health_contract.py), run one after another so that,
with the K=5 fit still sampling, the machine stays within its cores. Each job
logs to artifacts/health-base/<tag>/run.log; the queue writes digest.json.

    python clustering/runs/health_base_queue.py [--workers 1] [--dry-run]
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
CONTRACT = ROOT / "data" / "processed" / "contracts" / "health_lifecycle_20_89_minobs3_v1"
OUT = ROOT / "artifacts" / "health-base"
JOBS = [("health-theta-base", "health-theta-base"),
        ("health-h-base", "health-h-base"),
        ("health-fi10-base", "health-fi10-base")]


def run_job(job, settings):
    tag, model = job
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "clustering" / "run_fit.py"), "--model", model,
           "--contract", str(CONTRACT), "--output", str(out_dir)] + settings
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START {tag}", flush=True)
    with open(out_dir / "run.log", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    hours = (time.time() - t0) / 3600
    print(f"[{time.strftime('%H:%M:%S')}] {'DONE' if rc == 0 else f'FAILED rc={rc}'} {tag} ({hours:.2f} h)", flush=True)
    return tag, rc, hours


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--parallel-chains", type=int, default=4)
    parser.add_argument("--threads-per-chain", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--sampling", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    settings = ["--chains", str(args.chains), "--parallel-chains", str(args.parallel_chains),
                "--threads-per-chain", str(args.threads_per_chain),
                "--warmup", str(args.warmup), "--sampling", str(args.sampling)]
    if args.dry_run:
        for tag, model in JOBS:
            print(tag, model, CONTRACT.name)
        return 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda j: run_job(j, settings), JOBS))
    digest = {"total_hours": (time.time() - t0) / 3600,
              "jobs": [{"tag": t, "rc": rc, "hours": h} for t, rc, h in results]}
    for tag, rc, hours in results:
        summ = OUT / tag / "run_summary.json"
        if summ.exists():
            r = json.loads(summ.read_text())
            digest.setdefault("summaries", {})[tag] = {
                "max_rhat": r["max_structural_rhat"],
                "theta": [round(r["params"][f"theta[{i}]"]["mean"], 4) for i in range(1, 4)],
                "wall_hours": round(r["wall_hours"], 2)}
    (OUT / "digest.json").write_text(json.dumps(digest, indent=1))
    failed = [t for t, rc, _ in results if rc != 0]
    print(f"QUEUE COMPLETE in {digest['total_hours']:.1f} h; {'all ok' if not failed else 'FAILED: ' + ', '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
