"""The held-out generated quantities, checked against an independent
recomputation.

The Stan block emits two conditional predictive densities for each person's
held-out window:

  log_lik_heldout            AR-conditional: the window opens by carrying
                             the last FITTED observation forward through rho
  log_lik_heldout_marginal   the window opens at the stationary variance, so
                             the fitted rows reach it only through the class
                             posterior

Both are recomputed here in numpy from the same payload and the same
posterior draws. With ar_mode == 0 they must coincide; with ar_mode == 1 the
AR-conditional one must be the larger on average (it conditions on strictly
more information).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from prevention_health_clustering.models.registry import get_model
from prevention_health_clustering.runner.fit import (
    build_payload,
    compile_model,
    fit_model,
)

LOG2PI = float(np.log(2 * np.pi))
K = 3


def norm_lpdf(y, mu, sd):
    return -0.5 * LOG2PI - np.log(sd) - 0.5 * ((y - mu) / sd) ** 2


def simulate(n_persons=400, seed=99):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_persons):
        n_obs = int(rng.integers(5, 9))
        start = int(rng.integers(25, 70))
        ages = start + np.cumsum(rng.integers(1, 3, n_obs))
        level = rng.normal(0, 1)
        for age in ages:
            rows.append({"pidp": pid, "age": int(age), "wave": 1,
                         "birthy": 1960,
                         "sf12pcs_dv": 50 + 8 * level
                         - 0.15 * (age - 55) + rng.normal(0, 4)})
    return pd.DataFrame(rows)


def window_lp(y, X, age_gap, coef, sigma, rho, lo, hi, prev, ar):
    """Log density of rows [lo, hi] (0-based, inclusive) for every class."""
    idx = np.arange(lo, hi + 1)
    mu = X[idx] @ coef.T                       # (rows, K)
    out = np.zeros(K)
    for j, n in enumerate(idx):
        if j == 0 and (prev is None or not ar):
            if ar:
                out += norm_lpdf(y[n], mu[j], sigma / np.sqrt(1 - rho**2))
            else:
                out += norm_lpdf(y[n], mu[j], sigma)
        elif not ar:
            out += norm_lpdf(y[n], mu[j], sigma)
        else:
            anchor = prev if j == 0 else n - 1
            mu_prev = X[anchor] @ coef.T
            gap = age_gap[n]
            rho_gap = rho**gap
            var_mult = (1 - rho ** (2 * gap)) / (1 - rho**2)
            out += norm_lpdf(y[n], mu[j] + rho_gap * (y[anchor] - mu_prev),
                             sigma * np.sqrt(np.maximum(var_mult, 1e-9)))
    return out


def check(model_name: str, ar: bool, tmp_dir="artifacts/test-gq"):
    spec = get_model(model_name)
    frame = simulate()
    payload = build_payload(spec, frame, holdout_last_k=2,
                            holdout_min_person_obs=5,
                            emit_person_quantities=True)
    fit = fit_model(spec, payload, output_dir=__import__("pathlib").Path(tmp_dir),
                    chains=2, iter_warmup=100, iter_sampling=100,
                    threads_per_chain=2)
    d = payload.data
    y = np.asarray(d["y"][0], dtype=float)
    X = np.asarray(d["X"], dtype=float)
    age_gap = np.asarray(d["age_gap"], dtype=float)
    fs, fe = np.asarray(d["fit_start"]), np.asarray(d["fit_end"])
    hs, he = np.asarray(d["hold_start"]), np.asarray(d["hold_end"])

    got_ar = fit.stan_variable("log_lik_heldout")
    got_mg = fit.stan_variable("log_lik_heldout_marginal")
    draws = fit.draws_pd(vars=["theta", "coef", "sigma"]
                         + (["rho"] if ar else []))

    rng = np.random.default_rng(0)
    people = rng.choice(len(fs), 8, replace=False)
    for d_i in (0, 37, 91):
        row = draws.iloc[d_i]
        theta = np.array([row[f"theta[{k}]"] for k in range(1, K + 1)])
        coef = np.array([[row[f"coef[1,{k},{p}]"] for p in range(1, 4)]
                         for k in range(1, K + 1)])
        sigma = float(row["sigma[1,1]"])
        rho = (np.array([row[f"rho[{k}]"] for k in range(1, K + 1)])
               if ar else np.zeros(K))
        for i in people:
            f_lp = np.log(theta) + window_lp(
                y, X, age_gap, coef, sigma, rho,
                fs[i] - 1, fe[i] - 1, None, ar)
            base = logsumexp(f_lp)
            h_ar = window_lp(y, X, age_gap, coef, sigma, rho,
                             hs[i] - 1, he[i] - 1, fe[i] - 1, ar)
            h_mg = window_lp(y, X, age_gap, coef, sigma, rho,
                             hs[i] - 1, he[i] - 1, None, ar)
            want_ar = logsumexp(f_lp + h_ar) - base
            want_mg = logsumexp(f_lp + h_mg) - base
            assert abs(want_ar - got_ar[d_i, i]) < 1e-5, (
                model_name, d_i, i, want_ar, got_ar[d_i, i])
            assert abs(want_mg - got_mg[d_i, i]) < 1e-5, (
                model_name, d_i, i, want_mg, got_mg[d_i, i])
    return got_ar, got_mg


def test_ar_conditional_gq_matches_numpy():
    got_ar, got_mg = check("pcs-ar1", ar=True)
    # conditioning on the last fitted value can only help on average
    assert got_ar.mean() > got_mg.mean(), (got_ar.mean(), got_mg.mean())


def test_without_ar_the_two_estimands_coincide():
    got_ar, got_mg = check("pcs-headline", ar=False,
                           tmp_dir="artifacts/test-gq-noar")
    assert np.abs(got_ar - got_mg).max() < 1e-9


if __name__ == "__main__":
    import sys, traceback

    failures = 0
    for name, fn in sorted(
        (k, v) for k, v in globals().items()
        if k.startswith("test_") and callable(v)
    ):
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failures += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print("GQ TESTS OK" if failures == 0 else f"GQ TESTS FAILED ({failures})")
    sys.exit(1 if failures else 0)
