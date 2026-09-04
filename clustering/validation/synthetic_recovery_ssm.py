"""Recover known parameters from an AR(1)-plus-measurement-error panel.

Establishes that ar_mode 2 is correctly specified before any real fit is run.
The identification at stake is between persistent variation and transient
noise: both inflate the residual, and only the autocorrelation structure tells
them apart. Simulating from known values is the only way to check that the
Kalman filter actually separates them rather than trading one off against the
other along a ridge.

Two fits on the SAME simulated panel:

  ar_mode 2   should recover rho, the state scale and the measurement scale
  ar_mode 1   should recover a rho biased DOWNWARDS, because treating the
              observation as the state forces transient noise into the
              persistence parameter. This is the misspecification the new
              model exists to fix, so seeing it here is a positive result.

Run:
    PYTHONPATH=src .venv/bin/python \
        clustering/validation/synthetic_recovery_ssm.py --persons 2000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from prevention_health_clustering.models.registry import get_model
from prevention_health_clustering.runner.fit import build_payload, fit_model

TRUE_THETA = np.array([0.20, 0.35, 0.45])
TRUE_ALPHA = np.array([-1.60, -0.30, 0.75])
TRUE_BETA = np.array([-0.30, -0.22, -0.12])
TRUE_GAMMA = np.array([0.03, -0.02, -0.02])
TRUE_RHO = 0.75          # persistence of the latent state
TRUE_SIGMA_STATE = 0.45  # innovation sd of the state
TRUE_SIGMA_MEAS = 0.35   # i.i.d. observation noise

AGE_CENTER, AGE_SCALE = 55, 10.0
CHANNEL = "theta_phys_full"


def simulate(n_persons=2000, *, seed=20260904, min_age=25, max_age=85,
             mean_obs=9):
    rng = np.random.default_rng(seed)
    classes = rng.choice(len(TRUE_THETA), size=n_persons, p=TRUE_THETA)
    v_stat = TRUE_SIGMA_STATE**2 / (1 - TRUE_RHO**2)
    rows = []
    for person, k in enumerate(classes):
        n_obs = int(np.clip(rng.poisson(mean_obs - 5) + 5, 5, 15))
        start = rng.integers(min_age, max(min_age + 1, max_age - n_obs))
        ages = np.arange(start, start + n_obs)
        a = (ages - AGE_CENTER) / AGE_SCALE
        mu = TRUE_ALPHA[k] + TRUE_BETA[k] * a + TRUE_GAMMA[k] * a**2
        # latent AR(1) state, opened at its stationary variance
        u = np.empty(n_obs)
        u[0] = rng.normal(0.0, np.sqrt(v_stat))
        for t in range(1, n_obs):
            u[t] = TRUE_RHO * u[t - 1] + rng.normal(0.0, TRUE_SIGMA_STATE)
        y = mu + u + rng.normal(0.0, TRUE_SIGMA_MEAS, size=n_obs)
        for age, value in zip(ages, y):
            rows.append((person, age, age - AGE_CENTER, 1, 1970, value))
    frame = pd.DataFrame(rows, columns=["pidp", "age", "age_c", "wave",
                                        "birthy", CHANNEL])
    return frame, classes


def posterior(fit, k, has_meas):
    d = fit.draws_pd()
    out = {
        "theta": np.array([d[f"theta[{i}]"].mean() for i in range(1, k + 1)]),
        "alpha": np.array([d[f"coef[1,{i},1]"].mean() for i in range(1, k + 1)]),
        "rho": np.array([d[f"rho[{i}]"].mean() for i in range(1, k + 1)]),
        "sigma": float(d["sigma[1,1]"].mean()),
    }
    out["sigma_meas"] = float(d["sigma_meas[1]"].mean()) if has_meas else np.nan
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persons", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--sampling", type=int, default=400)
    ap.add_argument("--output", type=Path, default=Path("artifacts/recovery-ssm"))
    a = ap.parse_args(argv)

    frame, _ = simulate(a.persons)
    print(f"simulated {frame['pidp'].nunique():,} persons, {len(frame):,} rows")

    # The runner standardises internally, so put truth on the same scale.
    v = frame[CHANNEL].to_numpy()
    m, sd = float(np.mean(v)), float(np.std(v, ddof=1))
    truth = {"theta": TRUE_THETA,
             "alpha": (TRUE_ALPHA - m) / sd,
             "rho": np.full(3, TRUE_RHO),
             "sigma": TRUE_SIGMA_STATE / sd,
             "sigma_meas": TRUE_SIGMA_MEAS / sd}
    print(f"standardisation: mean {m:.4f}, sd {sd:.4f}")
    print(f"truth on that scale: rho {TRUE_RHO:.3f}, sigma_state "
          f"{truth['sigma']:.4f}, sigma_meas {truth['sigma_meas']:.4f}")
    signal = TRUE_SIGMA_STATE**2 / (1 - TRUE_RHO**2)
    print(f"signal share of observation variance: "
          f"{signal / (signal + TRUE_SIGMA_MEAS**2):.3f}")

    base = get_model("physgrm-full-ssm")
    results = {}
    for tag, ar in (("ar_mode 2 (state + measurement error)", 2),
                    ("ar_mode 1 (observation is the state)", 1)):
        spec = replace(base, ar_mode=ar, name=f"{base.name}-ar{ar}")
        payload = build_payload(spec, frame)
        fit = fit_model(spec, payload, output_dir=a.output / f"ar{ar}",
                        chains=a.chains, iter_warmup=a.warmup,
                        iter_sampling=a.sampling)
        results[ar] = posterior(fit, spec.n_classes, ar == 2)
        s = fit.summary()
        results[ar]["max_rhat"] = float(s["R_hat"].max())
        print(f"\n--- {tag}: max R-hat {results[ar]['max_rhat']:.4f} ---")

    print("\n=== recovery ===")
    print(f"{'param':14s} {'truth':>10s} {'ar_mode 2':>12s} {'ar_mode 1':>12s}")
    for key in ("rho", "sigma", "sigma_meas"):
        tv = truth[key]
        tv_s = np.mean(tv) if np.ndim(tv) else tv
        g2 = results[2][key]
        g1 = results[1][key]
        g2 = np.mean(g2) if np.ndim(g2) else g2
        g1 = np.mean(g1) if np.ndim(g1) else g1
        print(f"{key:14s} {tv_s:10.4f} {g2:12.4f} "
              f"{g1:12.4f}" if not np.isnan(g1) else
              f"{key:14s} {tv_s:10.4f} {g2:12.4f} {'n/a':>12s}")

    rho2, rho1 = np.mean(results[2]["rho"]), np.mean(results[1]["rho"])
    err = abs(rho2 - TRUE_RHO)
    print(f"\nrho: truth {TRUE_RHO:.3f}, ar_mode 2 recovers {rho2:.3f} "
          f"(error {err:.3f}), ar_mode 1 gives {rho1:.3f}")
    print(f"attenuation under the old spec: {(TRUE_RHO - rho1) / TRUE_RHO:+.1%}")
    ok = err < 0.06 and abs(results[2]["sigma_meas"] - truth["sigma_meas"]) < 0.06
    print(f"\nRECOVERY {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
