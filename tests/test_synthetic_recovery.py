"""Recover known parameters from simulated data.

This is the test that establishes the Stan program is correctly specified,
independently of whether any particular empirical fit has converged. Truth is
taken to be the published six-fit-v1 posterior means, so the simulated panel
sits in the same region of parameter space as the real data.

Run directly:

    PYTHONPATH=src .venv/bin/python tests/test_synthetic_recovery.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from prevention_health_clustering.models.registry import get_model
from prevention_health_clustering.runner.fit import build_payload, fit_model

# Published posterior means, on the standardised scale, native class order.
TRUE_THETA = np.array([0.11084, 0.27541, 0.61375])
TRUE_ALPHA = np.array([-1.90684, -0.45486, 0.47856])
TRUE_BETA = np.array([-0.24883, -0.26819, -0.13542])
TRUE_GAMMA = np.array([0.03484, -0.01953, -0.02512])
TRUE_SIGMA = 0.60906

AGE_CENTER = 55
AGE_SCALE = 10.0


def simulate(
    n_persons: int = 3000,
    *,
    seed: int = 20260823,
    min_age: int = 20,
    max_age: int = 89,
    mean_obs: int = 9,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Simulate an unbalanced panel with the real data's rough age structure."""
    rng = np.random.default_rng(seed)
    classes = rng.choice(len(TRUE_THETA), size=n_persons, p=TRUE_THETA)

    rows = []
    for person, k in enumerate(classes):
        n_obs = int(np.clip(rng.poisson(mean_obs - 3) + 3, 3, 15))
        start = rng.integers(min_age, max_age - n_obs)
        ages = np.arange(start, start + n_obs)
        a = (ages - AGE_CENTER) / AGE_SCALE
        mu = TRUE_ALPHA[k] + TRUE_BETA[k] * a + TRUE_GAMMA[k] * a**2
        y = mu + rng.normal(0.0, TRUE_SIGMA, size=n_obs)
        for age, value in zip(ages, y):
            rows.append((person, age, age - AGE_CENTER, 1, 1970, value))

    frame = pd.DataFrame(
        rows, columns=["pidp", "age", "age_c", "wave", "birthy", "sf12pcs_dv"]
    )
    # The runner standardises internally; simulated values are already on the
    # standardised scale, so undo that here to keep the round trip honest.
    frame["sf12pcs_dv"] = frame["sf12pcs_dv"] * 11.1724567 + 49.1684051
    return frame, classes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persons", type=int, default=3000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--sampling", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("artifacts/synthetic"))
    args = parser.parse_args(argv)

    frame, true_classes = simulate(args.persons)
    print(f"simulated {frame['pidp'].nunique():,} persons, {len(frame):,} rows")

    spec = get_model("pcs-headline")
    payload = build_payload(spec, frame)
    fit = fit_model(
        spec,
        payload,
        output_dir=args.output,
        chains=args.chains,
        iter_warmup=args.warmup,
        iter_sampling=args.sampling,
    )

    draws = fit.draws_pd()
    k = spec.n_classes
    got = {
        "theta": np.array([draws[f"theta[{i}]"].mean() for i in range(1, k + 1)]),
        "alpha": np.array([draws[f"coef[1,{i},1]"].mean() for i in range(1, k + 1)]),
        "beta": np.array([draws[f"coef[1,{i},2]"].mean() for i in range(1, k + 1)]),
        "gamma": np.array([draws[f"coef[1,{i},3]"].mean() for i in range(1, k + 1)]),
        "sigma": float(draws["sigma[1,1]"].mean()),
    }
    truth = {
        "theta": TRUE_THETA,
        "alpha": TRUE_ALPHA,
        "beta": TRUE_BETA,
        "gamma": TRUE_GAMMA,
        "sigma": TRUE_SIGMA,
    }

    print("\n=== parameter recovery ===")
    print(f"{'param':10s} {'recovered':>12s} {'truth':>12s} {'delta':>10s}")
    worst = {}
    for key in ("theta", "alpha", "beta", "gamma"):
        for i in range(k):
            delta = got[key][i] - truth[key][i]
            worst[key] = max(worst.get(key, 0.0), abs(delta))
            print(
                f"{key + str(i + 1):10s} {got[key][i]:12.5f} "
                f"{truth[key][i]:12.5f} {delta:10.5f}"
            )
    delta_sigma = got["sigma"] - truth["sigma"]
    worst["sigma"] = abs(delta_sigma)
    print(f"{'sigma':10s} {got['sigma']:12.5f} {truth['sigma']:12.5f} {delta_sigma:10.5f}")


    summary = fit.summary()
    rhat = summary["R_hat"].dropna().max()
    ess = summary["ESS_bulk"].dropna().min()
    print("\n=== diagnostics ===")
    print(f"  max R-hat     {rhat:.5f}")
    print(f"  min ESS bulk  {ess:.0f}")

    print("\n=== verdict ===")
    tolerances = {"theta": 0.05, "alpha": 0.15, "beta": 0.10, "gamma": 0.03, "sigma": 0.05}
    ok = True
    for key, tol in tolerances.items():
        status = "PASS" if worst[key] <= tol else "FAIL"
        ok &= worst[key] <= tol
        print(f"  {key:8s} max |delta| {worst[key]:.5f}  tol {tol:.3f}  {status}")
    converged = rhat < 1.05
    print(f"  {'R-hat':8s} {rhat:.5f} < 1.05  {'PASS' if converged else 'FAIL'}")
    ok &= converged
    print(f"\n{'RECOVERY OK' if ok else 'RECOVERY FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
