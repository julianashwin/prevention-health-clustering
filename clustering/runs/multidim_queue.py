"""The four multidimensional variants, two at a time.

baseline | holdout | ar1 | ar1-holdout, longest first. Each writes to
artifacts/multidim/<variant>/ with its own run.log; a digest lands at
artifacts/multidim/digest.json.
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
OUT = ROOT / "artifacts" / "multidim"
ORDER = ["ar1", "ar1-holdout", "baseline", "holdout"]


def run_job(variant, settings):
    out_dir = OUT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "clustering" / "run_multidim.py"),
           "--variant", variant, "--output", str(out_dir)] + settings
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START {variant}", flush=True)
    with open(out_dir / "run.log", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    hours = (time.time() - t0) / 3600
    print(f"[{time.strftime('%H:%M:%S')}] "
          f"{'DONE' if rc == 0 else f'FAILED rc={rc}'} {variant} "
          f"({hours:.2f} h)", flush=True)
    return variant, rc, hours


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
        for v in ORDER:
            print(v, settings)
        return 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda v: run_job(v, settings), ORDER))
    digest = {"total_hours": (time.time() - t0) / 3600, "jobs": {}}
    for v, rc, h in results:
        entry = {"rc": rc, "hours": round(h, 2)}
        summ = OUT / v / "run_summary.json"
        if summ.exists():
            r = json.loads(summ.read_text())
            entry.update({
                "max_rhat": r["max_structural_rhat"],
                "theta": [round(r["params"][f"theta[{k}]"]["mean"], 4)
                          for k in (1, 2, 3)],
                "heldout_ar": r.get("heldout_ar_conditional_mean"),
                "heldout_class_only": r.get("heldout_class_only_mean"),
            })
        digest["jobs"][v] = entry
    (OUT / "digest.json").write_text(json.dumps(digest, indent=1))
    failed = [v for v, rc, _ in results if rc != 0]
    print(f"MULTIDIM QUEUE COMPLETE in {digest['total_hours']:.1f} h; "
          f"{'all ok' if not failed else 'FAILED: ' + ', '.join(failed)}",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
