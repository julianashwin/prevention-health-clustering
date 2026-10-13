"""Bayesian version of the windows figure: theta AR(1) + "spike" refitted on age windows.

For each sliding twenty-year window and each window expanding backwards from 90
(the windows of descriptives/22_paper_kmeans.py), a contract is cut from the health
contract -- the rows with age inside the window, people with at least three such
rows -- and health-theta-ssm is fitted on it. Each window contract is written to
data/processed/contracts/health_window_{lo}_{hi}_v1/ (long.csv plus a manifest whose
metric moments are the window's own, so descriptives/_paper_common.bayes_fit reads it
like any contract); fits go to artifacts/health-windows/theta-ssm-{lo}-{hi}/
(--k 4: the K = 4 spec, tags theta-ssm-k4-{lo}-{hi}, log queue_k4.log).
Finished fits are skipped.

    python clustering/runs/windows_queue.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "data" / "processed" / "contracts" / "health_lifecycle_20_89_minobs3_v1"
CONTRACTS = ROOT / "data" / "processed" / "contracts"
OUT = ROOT / "artifacts" / "health-windows"
SLIDING = [(20, 40), (30, 50), (40, 60), (50, 70), (60, 80), (70, 90)]
BACKWARD = [(60, 90), (50, 90), (40, 90), (30, 90)]
WINDOWS = SLIDING + [w for w in BACKWARD if w not in SLIDING]
MIN_OBS = 3


def window_contract(lo: int, hi: int) -> Path:
    out = CONTRACTS / f"health_window_{lo}_{hi}_v1"
    if (out / "long.csv").exists():
        return out
    long = pd.read_csv(CONTRACT / "long.csv")
    w = long[long["age"].between(lo, hi)]
    n = w.groupby("pidp")["age"].transform("size")
    w = w[n >= MIN_OBS].sort_values(["pidp", "age"]).reset_index(drop=True)
    base = json.loads((CONTRACT / "manifest.json").read_text())
    manifest = {"schema_version": base.get("schema_version"), "parent_contract": CONTRACT.name,
                "spec": {**base.get("spec", {}), "contract_id": out.name, "age_min_inclusive": lo, "age_max_inclusive": hi},
                "counts": {"retained_persons": int(w["pidp"].nunique()), "retained_rows": int(len(w))},
                "metric_moments": {m: {"mean": float(w[m].mean()), "sd": float(w[m].std(ddof=1))} for m in ("theta", "h")}}
    out.mkdir(parents=True, exist_ok=True)
    w.to_csv(out / "long.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[{time.strftime('%H:%M:%S')}] contract {out.name}: {w['pidp'].nunique():,} people, {len(w):,} rows", flush=True)
    return out


def run_job(lo: int, hi: int, settings: list[str], k: int = 3) -> tuple[str, int, float]:
    tag = f"theta-ssm-{lo}-{hi}" if k == 3 else f"theta-ssm-k{k}-{lo}-{hi}"
    out_dir = OUT / tag
    if (out_dir / "run_summary.json").exists():
        print(f"[{time.strftime('%H:%M:%S')}] SKIP {tag} (already finished)", flush=True)
        return tag, 0, 0.0
    contract = window_contract(lo, hi)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = "health-theta-ssm" if k == 3 else f"health-theta-ssm-k{k}"       # models/registry.py HEALTH_K45
    cmd = [sys.executable, str(ROOT / "clustering" / "run_fit.py"), "--model", model,
           "--contract", str(contract), "--output", str(out_dir)] + settings
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
    ap.add_argument("--k", type=int, default=3, help="class count: 3 (health-theta-ssm) or 4 / 5 (health-theta-ssm-k4/5), tags theta-ssm-k{K}-{lo}-{hi}")
    a = ap.parse_args(argv)
    settings = ["--chains", str(a.chains), "--parallel-chains", str(a.parallel_chains),
                "--threads-per-chain", str(a.threads_per_chain), "--warmup", str(a.warmup), "--sampling", str(a.sampling)]
    ksfx = "" if a.k == 3 else f"-k{a.k}"
    if a.dry_run:
        for lo, hi in WINDOWS:
            print(f"theta-ssm{ksfx}-{lo}-{hi}  contract health_window_{lo}_{hi}_v1")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = [run_job(lo, hi, settings, a.k) for lo, hi in WINDOWS]
    digest = {"total_hours": (time.time() - t0) / 3600, "jobs": [{"tag": t, "rc": rc, "hours": h} for t, rc, h in results]}
    (OUT / f"digest{ksfx}.json").write_text(json.dumps(digest, indent=1))
    failed = [t for t, rc, _ in results if rc != 0]
    print(f"WINDOWS{ksfx.upper()} QUEUE COMPLETE in {digest['total_hours']:.1f} h; {'all ok' if not failed else 'FAILED: ' + ', '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
