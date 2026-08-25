"""Compare a rebuilt PCS fit against the published six-fit-v1 bundle.

The published fit used the same roster, the same quadratic design in
(age - 55)/10, the same standardisation, and the same K. Its posterior means
are therefore a direct target for the rebuilt model.

Reference values are taken from
``artifacts/exports/model-parameters/six-fit-v1/fits/pcs_headline`` in the
predecessor repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from prevention_health_clustering.models.registry import get_model
from prevention_health_clustering.runner.fit import build_payload, fit_model

REFERENCE = {
    "theta": [0.11084, 0.27541, 0.61375],
    "alpha": [-1.90684, -0.45486, 0.47856],
    "beta": [-0.24883, -0.26819, -0.13542],
    "gamma": [0.03484, -0.01953, -0.02512],
    "sigma": [0.60906],
    "y_mean": 49.1684050688419,
    "y_sd": 11.1724567013148,
}

# Published response-scale class trajectories, native class order.
REFERENCE_TRAJECTORY = {
    20: [42.36, 51.90, 56.37],
    50: [29.35, 45.53, 55.20],
    55: [27.86, 44.09, 54.52],
    60: [26.57, 42.53, 53.69],
    89: [22.91, 31.38, 46.13],
}


def summarise(fit, payload, spec) -> dict:
    draws = fit.draws_pd()
    out: dict[str, list[float]] = {}
    k = spec.n_classes
    out["theta"] = [float(draws[f"theta[{i}]"].mean()) for i in range(1, k + 1)]
    out["alpha"] = [float(draws[f"coef[1,{i},1]"].mean()) for i in range(1, k + 1)]
    out["beta"] = [float(draws[f"coef[1,{i},2]"].mean()) for i in range(1, k + 1)]
    out["gamma"] = [float(draws[f"coef[1,{i},3]"].mean()) for i in range(1, k + 1)]
    out["sigma"] = [float(draws["sigma[1,1]"].mean())]
    return out


def response_scale_trajectory(params, moments, ages, age_center=55, age_scale=10.0):
    mean, sd = moments["mean"], moments["sd"]
    rows = {}
    for age in ages:
        a = (age - age_center) / age_scale
        rows[age] = [
            (params["alpha"][k] + params["beta"][k] * a + params["gamma"][k] * a * a)
            * sd
            + mean
            for k in range(len(params["alpha"]))
        ]
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-persons", type=int, default=0)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--sampling", type=int, default=300)
    parser.add_argument("--threads-per-chain", type=int, default=2)
    args = parser.parse_args(argv)

    spec = get_model("pcs-headline")
    long = pd.read_csv(args.contract / "long.csv")
    if args.max_persons:
        keep = long["pidp"].drop_duplicates().head(args.max_persons)
        long = long[long["pidp"].isin(keep)]

    payload = build_payload(spec, long)
    print(
        f"payload: {payload.data['N_person']:,} persons, "
        f"{payload.data['N_obs']:,} rows, K={payload.data['K']}, "
        f"C={payload.data['C']}, P={payload.data['P']}, "
        f"ar_mode={payload.data['ar_mode']}, N_cohort={payload.data['N_cohort']}"
    )
    moments = payload.channel_moments[spec.channels[0]]
    print(f"moments: mean={moments['mean']!r} sd={moments['sd']!r}")
    print(
        f"  vs published mean={REFERENCE['y_mean']!r} sd={REFERENCE['y_sd']!r}"
    )

    fit = fit_model(
        spec,
        payload,
        output_dir=args.output,
        threads_per_chain=args.threads_per_chain,
        chains=args.chains,
        iter_warmup=args.warmup,
        iter_sampling=args.sampling,
    )

    params = summarise(fit, payload, spec)
    print("\n=== posterior means: rebuilt vs published ===")
    print(f"{'param':10s} {'rebuilt':>12s} {'published':>12s} {'delta':>10s}")
    for key in ("theta", "alpha", "beta", "gamma", "sigma"):
        for i, value in enumerate(params[key]):
            ref = REFERENCE[key][i]
            print(f"{key + str(i + 1):10s} {value:12.5f} {ref:12.5f} {value - ref:10.5f}")

    print("\n=== response-scale class trajectories ===")
    traj = response_scale_trajectory(params, moments, sorted(REFERENCE_TRAJECTORY))
    print(f"{'age':>5s}  {'rebuilt (c1,c2,c3)':>28s}   {'published':>28s}")
    max_abs = 0.0
    for age in sorted(REFERENCE_TRAJECTORY):
        got = traj[age]
        ref = REFERENCE_TRAJECTORY[age]
        max_abs = max(max_abs, max(abs(g - r) for g, r in zip(got, ref)))
        got_s = ", ".join(f"{v:7.2f}" for v in got)
        ref_s = ", ".join(f"{v:7.2f}" for v in ref)
        print(f"{age:5d}  {got_s:>28s}   {ref_s:>28s}")
    print(f"\nmax absolute trajectory difference: {max_abs:.3f} PCS points")

    diagnostics = fit.diagnose()
    print("\n=== sampler diagnostics ===")
    summary = fit.summary()
    rhat = summary["R_hat"].dropna()
    print(f"  max R-hat        {rhat.max():.5f}")
    print(f"  min ESS bulk     {summary['ESS_bulk'].dropna().min():.0f}")
    print(f"  divergences      {'yes' if 'divergent' in diagnostics.lower() else 'none reported'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
