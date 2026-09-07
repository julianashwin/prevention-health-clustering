"""Verify the ar_mode 2 held-out generated quantities against an offline filter.

The three state-space fits ran without holdout, so the `filter_start` path in
person_class_loglik -- the one that filters over the whole fitted history
before scoring held-out rows -- has never been exercised on real output. This
is the same validation done for the AR(1) decomposition: recompute the
estimand offline and check it reproduces what Stan stored.

Conditioning matters here in a way it does not under ar_mode 1. With
measurement error the last fitted observation is a noisy read of the state,
so a forecast must condition on the filtered state carried through the whole
fitted window, not on that one value.

    PYTHONPATH=src .venv/bin/python clustering/validation/ssm_holdout_check.py
"""

from __future__ import annotations

import sys

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from prevention_health_clustering.config import PROCESSED_DATA_DIR
from prevention_health_clustering.models.registry import get_model
from prevention_health_clustering.runner.fit import build_payload, fit_model

K = 3


def offline(d, p, n_person):
    """Recompute fitted and held-out class log densities with a Kalman filter."""
    X, Y = np.asarray(d["X"]), np.asarray(d["y"][0])
    gap = np.asarray(d["age_gap"])
    fs, fe = np.asarray(d["fit_start"]), np.asarray(d["fit_end"])
    hs, he = np.asarray(d["hold_start"]), np.asarray(d["hold_end"])
    theta = np.array([p[f"theta[{k}]"]["mean"] for k in range(1, K + 1)])
    coef = np.array([[p[f"coef[1,{k},{j}]"]["mean"] for j in (1, 2, 3)]
                     for k in range(1, K + 1)])
    rho = np.array([p[f"rho[{k}]"]["mean"] for k in range(1, K + 1)])
    sig, mea = p["sigma[1,1]"]["mean"], p["sigma_meas[1]"]["mean"]
    mu_all = X @ coef.T
    v_stat = sig**2 / (1 - rho**2)

    fitted = np.zeros((n_person, K))
    held = np.zeros((n_person, K))
    for i in range(n_person):
        a = np.zeros(K)
        Pv = v_stat.copy()
        last = he[i] if hs[i] > 0 else fe[i]
        for n in range(fs[i] - 1, last):          # zero-based row index
            if n > fs[i] - 1:
                g = gap[n]
                rg = rho ** g
                a = rg * a
                Pv = rg**2 * Pv + v_stat * np.maximum(1 - rho ** (2 * g), 1e-9)
            F = Pv + mea**2
            v = Y[n] - mu_all[n] - a
            lp = -0.5 * (np.log(2 * np.pi * F) + v**2 / F)
            if n < fe[i]:
                fitted[i] += lp
            else:
                held[i] += lp
            Kg = Pv / F
            a = a + Kg * v
            Pv = Pv - Kg * Pv
    lw = np.log(theta)[None, :]
    ll_fit = logsumexp(lw + fitted, axis=1)
    ll_hold = logsumexp(lw + fitted + held, axis=1) - ll_fit
    return ll_hold


def main() -> int:
    spec = get_model("physgrm-full-ssm")
    long = pd.read_csv(PROCESSED_DATA_DIR / "contracts"
                       / "physgrm_lifecycle_20_89_minobs3_v1" / "long.csv")
    keep = long["pidp"].drop_duplicates().sample(2500, random_state=11)
    long = long[long["pidp"].isin(set(keep))]
    pl = build_payload(spec, long, holdout_last_k=2, holdout_min_person_obs=5,
                       emit_person_quantities=True)
    print(f"smoke sample: {pl.data['N_person']:,} people, "
          f"{pl.data['N_obs']:,} rows, "
          f"{sum(1 for h in pl.data['hold_start'] if h > 0):,} with held-out rows")

    fit = fit_model(spec, pl, output_dir=Path("artifacts/ssm-holdout-check"),
                    chains=2, iter_warmup=250, iter_sampling=250)
    s = fit.summary()
    print(f"max R-hat {s['R_hat'].max():.4f} (a smoke fit, not a converged one)")

    stan = fit.stan_variable("log_lik_heldout")
    lpd = logsumexp(stan, axis=0) - np.log(stan.shape[0])

    draws = fit.draws_pd()
    p = {n: {"mean": float(draws[n].mean())} for n in draws.columns
         if n.startswith(("theta[", "coef[1,", "rho[", "sigma[",
                           "sigma_meas["))}
    mine = offline(pl.data, p, pl.data["N_person"])

    m = np.array([h > 0 for h in pl.data["hold_start"]])
    r = np.corrcoef(lpd[m], mine[m])[0, 1]
    md = float(np.max(np.abs(lpd[m] - mine[m])))
    print(f"\nStan held-out LPD vs offline recomputation, {m.sum():,} people:")
    print(f"  correlation {r:.7f}   max abs diff {md:.4f}")
    print(f"  means: Stan {lpd[m].mean():.4f}, offline {mine[m].mean():.4f}")
    print("\n(The offline value uses posterior-MEAN parameters while Stan "
          "averages the\n density over draws, so a small gap is expected; a "
          "correlation below ~0.99\n or a sign disagreement would indicate the "
          "filter_start path is wrong.)")
    ok = r > 0.99
    print(f"\nHELD-OUT PATH {'LOOKS CORRECT' if ok else 'SUSPECT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
