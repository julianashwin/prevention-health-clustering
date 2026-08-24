"""Parameter recovery from simulated data, for any registered Gaussian model.

Establishes that a model is correctly specified, independently of whether any
particular empirical fit converged. Truth is taken from the published
six-fit-v1 posterior means so the simulated panel sits in the same region of
parameter space as the real data.

The bivariate case matters because no joint PCS+MCS fit exists in the published
export — it is one of the artifacts the evidence registry pins but that is not
recoverable from the checkout. Synthetic recovery is therefore the only
available validation for C=2.

    PYTHONPATH=src .venv/bin/python tests/test_recovery.py --model joint-headline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from prevention_health_clustering.models.registry import get_model
from prevention_health_clustering.runner.fit import (
    assignment_inits,
    build_payload,
    fit_model,
    pathfinder_inits,
)

AGE_CENTER, AGE_SCALE = 55, 10.0

# Per-channel truth on the standardised scale, native class order.
TRUTH = {
    "sf12pcs_dv": {
        "alpha": [-1.90684, -0.45486, 0.47856],
        "beta": [-0.24883, -0.26819, -0.13542],
        "gamma": [0.03484, -0.01953, -0.02512],
        "sigma": 0.60906,
        "mean": 49.1684051,
        "sd": 11.1724567,
    },
    "sf12mcs_dv": {
        "alpha": [-1.59735, -0.41358, 0.51184],
        "beta": [0.08897, 0.05817, 0.06583],
        "gamma": [0.03108, 0.01496, 0.00543],
        "sigma": 0.74103,
        "mean": 48.8403343,
        "sd": 10.2822509,
    },
}
TRUE_THETA = np.array([0.11084, 0.27541, 0.61375])

TOLERANCES = {"theta": 0.05, "alpha": 0.20, "beta": 0.12, "gamma": 0.04, "sigma": 0.06}


def simulate(channels, n_persons: int, seed: int = 20260824) -> pd.DataFrame:
    """Unbalanced panel; channels share one latent class per person."""
    rng = np.random.default_rng(seed)
    classes = rng.choice(len(TRUE_THETA), size=n_persons, p=TRUE_THETA)
    records = []
    for person, k in enumerate(classes):
        n_obs = int(np.clip(rng.poisson(6) + 3, 3, 15))
        start = rng.integers(20, 89 - n_obs)
        ages = np.arange(start, start + n_obs)
        a = (ages - AGE_CENTER) / AGE_SCALE
        row = {"pidp": person, "age": ages, "wave": 1, "birthy": 1970}
        values = {}
        for channel in channels:
            t = TRUTH[channel]
            mu = t["alpha"][k] + t["beta"][k] * a + t["gamma"][k] * a**2
            z = mu + rng.normal(0.0, t["sigma"], size=n_obs)
            values[channel] = z * t["sd"] + t["mean"]
        for i, age in enumerate(ages):
            rec = {"pidp": person, "age": int(age), "age_c": int(age) - AGE_CENTER,
                   "wave": 1, "birthy": 1970}
            rec.update({ch: values[ch][i] for ch in channels})
            records.append(rec)
    return pd.DataFrame.from_records(records)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pcs-headline")
    parser.add_argument("--persons", type=int, default=4000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--sampling", type=int, default=500)
    parser.add_argument("--threads-per-chain", type=int, default=2)
    parser.add_argument("--pathfinder", action="store_true",
                        help="initialise from Pathfinder instead of the static ladder")
    parser.add_argument("--assignment-init", action="store_true",
                        help="k-means partition + per-class OLS init (the predecessor recipe)")
    parser.add_argument("--init-jitter", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/recovery"))
    args = parser.parse_args(argv)

    spec = get_model(args.model)
    frame = simulate(spec.channels, args.persons)
    print(f"{args.model}: simulated {frame['pidp'].nunique():,} persons, "
          f"{len(frame):,} rows, C={spec.n_channels}, K={spec.n_classes}")

    out = args.output / args.model
    payload = build_payload(spec, frame)

    inits = None
    if args.assignment_init:
        inits = assignment_inits(spec, payload, jitter=args.init_jitter)
        print(f"  assignment init (jitter={args.init_jitter}); "
              f"theta start {np.round(inits[0]['theta'], 3)}")
    elif args.pathfinder:
        inits, seconds = pathfinder_inits(spec, payload, output_dir=out / "pathfinder")
        print(f"  pathfinder init: {seconds:.1f}s")

    fit = fit_model(
        spec, payload, output_dir=out, inits=inits,
        chains=args.chains, iter_warmup=args.warmup,
        iter_sampling=args.sampling, threads_per_chain=args.threads_per_chain,
    )
    draws = fit.draws_pd()
    k = spec.n_classes

    print(f"\n{'channel':12s} {'param':8s} {'recovered':>11s} {'truth':>11s} {'delta':>10s}")
    worst: dict[str, float] = {}
    theta = np.array([draws[f"theta[{i}]"].mean() for i in range(1, k + 1)])
    for i in range(k):
        d = theta[i] - TRUE_THETA[i]
        worst["theta"] = max(worst.get("theta", 0), abs(d))
        print(f"{'-':12s} {'theta' + str(i+1):8s} {theta[i]:11.5f} "
              f"{TRUE_THETA[i]:11.5f} {d:10.5f}")

    for ci, channel in enumerate(spec.channels, start=1):
        t = TRUTH[channel]
        for col, name in ((1, "alpha"), (2, "beta"), (3, "gamma")):
            for i in range(1, k + 1):
                got = float(draws[f"coef[{ci},{i},{col}]"].mean())
                truth = t[name][i - 1]
                d = got - truth
                worst[name] = max(worst.get(name, 0), abs(d))
                print(f"{channel:12s} {name + str(i):8s} {got:11.5f} {truth:11.5f} {d:10.5f}")
        row = 1 if spec.homosigma else 1
        got = float(draws[f"sigma[{row},{ci}]"].mean())
        d = got - t["sigma"]
        worst["sigma"] = max(worst.get("sigma", 0), abs(d))
        print(f"{channel:12s} {'sigma':8s} {got:11.5f} {t['sigma']:11.5f} {d:10.5f}")

    summary = fit.summary()
    rhat = float(summary["R_hat"].dropna().max())
    ess = float(summary["ESS_bulk"].dropna().min())
    print(f"\ndiagnostics: max R-hat {rhat:.5f}, min ESS bulk {ess:.0f}")

    print("\n=== verdict ===")
    ok = True
    for key, tol in TOLERANCES.items():
        good = worst.get(key, 0.0) <= tol
        ok &= good
        print(f"  {key:8s} max |delta| {worst.get(key, 0):.5f}  tol {tol:.3f}  "
              f"{'PASS' if good else 'FAIL'}")
    conv = rhat < 1.05
    ok &= conv
    print(f"  {'R-hat':8s} {rhat:.5f} < 1.05  {'PASS' if conv else 'FAIL'}")
    print(f"\n{'RECOVERY OK' if ok else 'RECOVERY FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
