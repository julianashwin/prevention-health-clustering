"""Multidimensional batch: measurement error, and the mortality channel.

Three fits, two at a time, longest first.

  ssm            AR(1) latent state plus one-period measurement error on the
                 two Gaussian channels. The multidimensional counterpart of
                 the univariate state-space fits: does separating transient
                 noise from persistence change the joint typology the way it
                 changed the univariate one?
  ssm-mort       the same, with the discrete-time mortality hazard switched
                 on. The mortality channel has been built and toggleable
                 since the multidim model was written but has never been
                 fitted.
  baseline-mort  mortality on top of the conditionally independent baseline,
                 so the mortality channel's contribution can be read against
                 a specification with no persistence at all.

Measurement error applies to the GAUSSIAN channels only. A chronic-condition
count is a report of an accumulated stock rather than a noisy read of a
continuous state, and mortality is administrative; neither has a
measurement-error analogue here.

RUNTIME. The multidim AR(1) fits took 19.0 h and 18.3 h; the baselines 8.0 h
and 6.9 h. The Kalman filter is somewhat dearer per gradient evaluation than
the AR(1) recursion, so expect the two state-space fits to be the long poles.

    python clustering/runs/multidim_ssm_queue.py [--dry-run]
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
OUT = ROOT / "artifacts" / "multidim-ssm"

# (tag, variant, extra args). Run one at a time, in the order the results
# are wanted: the state-space spec first, then the mortality channel on top
# of it, then mortality on the iid baseline as the no-persistence contrast.
JOBS = [
    ("ssm", "ssm", []),
    ("ssm-mort", "ssm", ["--with-mortality"]),
    ("baseline-mort", "baseline", ["--with-mortality"]),
]


def run_job(job, settings):
    tag, variant, extra = job
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "clustering" / "run_multidim.py"),
           "--variant", variant, "--output", str(out_dir)] + settings + extra
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
    return {"max_rhat": j.get("max_structural_rhat"),
            "wall_hours": j.get("wall_hours"),
            "n_person": j.get("n_person"), "n_obs": j.get("n_obs"),
            "use_mortality": j.get("use_mortality"),
            "theta": [get(f"theta[{k}]") for k in (1, 2, 3)],
            "rho": [get(f"rho[{k}]") for k in (1, 2, 3)],
            "sigma": [get(f"sigma[1,{c}]") for c in (1, 2)],
            "sigma_meas": [get(f"sigma_meas[{c}]") for c in (1, 2)],
            "mort_intercept": [get(f"coef_mort[{k},1]") for k in (1, 2, 3)],
            "heldout_mean_lpd": j.get("heldout_mean_lpd")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--sampling", type=int, default=1000)
    ap.add_argument("--threads-per-chain", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    settings = ["--chains", str(a.chains), "--warmup", str(a.warmup),
                "--sampling", str(a.sampling),
                "--threads-per-chain", str(a.threads_per_chain)]
    if a.dry_run:
        for tag, variant, extra in JOBS:
            print(f"{tag:16s} variant={variant:10s} "
                  f"{'mortality ON ' if extra else 'mortality off'}")
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
            print(f"  {t:16s} Rhat {s['max_rhat']:.4f}  theta {s['theta']}")
            if s.get("sigma_meas") and s["sigma_meas"][0]:
                print(f"      sigma {s['sigma']}  sigma_meas {s['sigma_meas']}")
            if s.get("use_mortality"):
                print(f"      mortality logit intercepts {s['mort_intercept']}")
    return 0 if all(rc == 0 for _, rc, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
