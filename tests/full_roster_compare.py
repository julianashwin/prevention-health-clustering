"""Full-roster fits compared against the published six-fit-v1 bundle.

Person-level generated quantities stay OFF, so the draws file holds the ~20
parameters that matter rather than 301,200 columns per draw.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from prevention_health_clustering.models.registry import get_model
from prevention_health_clustering.runner.fit import (
    assignment_inits,
    build_payload,
    fit_model,
)

# Published posterior means, standardised scale, native class order.
REFERENCE = {
    "pcs-headline": {
        "theta": [0.11084, 0.27541, 0.61375],
        "alpha": [-1.90684, -0.45486, 0.47856],
        "beta": [-0.24883, -0.26819, -0.13542],
        "gamma": [0.03484, -0.01953, -0.02512],
        "sigma": 0.60906,
        "y_mean": 49.1684050688419,
        "y_sd": 11.1724567013148,
    },
    "mcs-headline": {
        "theta": [0.10548, 0.34094, 0.55358],
        "alpha": [-1.59735, -0.41358, 0.51184],
        "beta": [0.08897, 0.05817, 0.06583],
        "gamma": [0.03108, 0.01496, 0.00543],
        "sigma": 0.74103,
        "y_mean": 48.8403342536121,
        "y_sd": 10.2822508549754,
    },
    # No bivariate fit exists in the export; only the report's modal shares.
    "joint-headline": None,
}

REPORT_MODAL_SHARES = {"joint-headline": [13.4, 27.5, 59.2]}  # native order 1,2,3


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--sampling", type=int, default=1000)
    parser.add_argument("--threads-per-chain", type=int, default=3)
    parser.add_argument("--assignment-init", action="store_true")
    parser.add_argument("--init-jitter", type=float, default=0.15)
    args = parser.parse_args(argv)

    spec = get_model(args.model)
    long = pd.read_csv(args.contract / "long.csv")
    payload = build_payload(spec, long)
    print(
        f"{args.model}: {payload.data['N_person']:,} persons, "
        f"{payload.data['N_obs']:,} rows, C={payload.data['C']}, "
        f"K={payload.data['K']}, emit_person_quantities="
        f"{payload.data['emit_person_quantities']}"
    )
    for channel, m in payload.channel_moments.items():
        print(f"  {channel:14s} mean {m['mean']!r}  sd {m['sd']!r}")

    inits = None
    if args.assignment_init:
        inits = assignment_inits(spec, payload, jitter=args.init_jitter)
        print(f"  partition init, jitter={args.init_jitter}, "
              f"theta start {[round(v, 3) for v in inits[0]['theta']]}")

    fit = fit_model(
        spec, payload,
        inits=inits,
        output_dir=args.output,
        chains=args.chains,
        iter_warmup=args.warmup,
        iter_sampling=args.sampling,
        threads_per_chain=args.threads_per_chain,
    )

    draws = fit.draws_pd()
    k, c = spec.n_classes, spec.n_channels
    got = {
        "theta": [float(draws[f"theta[{i}]"].mean()) for i in range(1, k + 1)],
        "alpha": [float(draws[f"coef[1,{i},1]"].mean()) for i in range(1, k + 1)],
        "beta": [float(draws[f"coef[1,{i},2]"].mean()) for i in range(1, k + 1)],
        "gamma": [float(draws[f"coef[1,{i},3]"].mean()) for i in range(1, k + 1)],
        "sigma": float(draws["sigma[1,1]"].mean()),
    }
    sd = {
        "theta": [float(draws[f"theta[{i}]"].std()) for i in range(1, k + 1)],
        "alpha": [float(draws[f"coef[1,{i},1]"].std()) for i in range(1, k + 1)],
        "beta": [float(draws[f"coef[1,{i},2]"].std()) for i in range(1, k + 1)],
        "gamma": [float(draws[f"coef[1,{i},3]"].std()) for i in range(1, k + 1)],
        "sigma": float(draws["sigma[1,1]"].std()),
    }

    summary = fit.summary()
    rhat = float(summary["R_hat"].dropna().max())
    ess = float(summary["ESS_bulk"].dropna().min())
    print(f"\ndiagnostics: max R-hat {rhat:.5f}, min ESS bulk {ess:.0f}")

    ref = REFERENCE.get(args.model)
    if ref is None:
        print("\nNo published parameter reference for this model.")
        print("Posterior means (channel 1):")
        for key in ("theta", "alpha", "beta", "gamma"):
            print(f"  {key:6s} " + ", ".join(f"{v:8.5f}" for v in got[key]))
        print(f"  {'sigma':6s} {got['sigma']:8.5f}")
        if c > 1:
            print("Channel 2:")
            for p, name in ((1, "alpha"), (2, "beta"), (3, "gamma")):
                vals = [float(draws[f"coef[2,{i},{p}]"].mean()) for i in range(1, k + 1)]
                print(f"  {name:6s} " + ", ".join(f"{v:8.5f}" for v in vals))
            print(f"  sigma2 {float(draws['sigma[1,2]'].mean()):8.5f}")
        shares = REPORT_MODAL_SHARES.get(args.model)
        if shares:
            got_pct = [v * 100 for v in got["theta"]]
            print("\nmixture weights vs the report's modal shares (weak check):")
            print(f"  theta%    " + ", ".join(f"{v:6.2f}" for v in got_pct))
            print(f"  report%   " + ", ".join(f"{v:6.2f}" for v in shares))
            print("  (theta is the mixture weight; the report quotes modal "
                  "assignment shares, so these differ slightly by construction)")
        return 0

    print(f"\n{'param':8s} {'rebuilt':>11s} {'published':>11s} {'delta':>10s} "
          f"{'post.sd':>9s} {'|d|/sd':>8s}")
    worst = 0.0
    for key in ("theta", "alpha", "beta", "gamma"):
        for i in range(k):
            delta = got[key][i] - ref[key][i]
            z = abs(delta) / sd[key][i]
            worst = max(worst, z)
            print(f"{key + str(i + 1):8s} {got[key][i]:11.5f} {ref[key][i]:11.5f} "
                  f"{delta:10.5f} {sd[key][i]:9.5f} {z:8.2f}")
    delta = got["sigma"] - ref["sigma"]
    z = abs(delta) / sd["sigma"]
    worst = max(worst, z)
    print(f"{'sigma':8s} {got['sigma']:11.5f} {ref['sigma']:11.5f} {delta:10.5f} "
          f"{sd['sigma']:9.5f} {z:8.2f}")
    print(f"\nlargest discrepancy: {worst:.2f} posterior SDs")

    (args.output / "comparison.json").write_text(
        json.dumps({"model": args.model, "rebuilt": got, "published": ref,
                    "max_z": worst, "max_rhat": rhat, "min_ess": ess}, indent=2)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
