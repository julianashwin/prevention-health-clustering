"""Out-of-sample assessment of the holdout fits: density, accuracy, calibration.

The Stan generated quantities report a CLASS-CONDITIONAL held-out density:
``person_class_loglik`` restarts the AR(1) recursion at the first held-out
row with the stationary variance, so the person's own last fitted value is
not carried forward through rho. The fitted rows inform the held-out rows
only through the class posterior (the second held-out row does condition on
the first). That is a legitimate estimand -- what the latent class structure
alone predicts -- but it is not the forecast a persistence model can make.

Both are reported here on the same rows:

  class-only        the model's own estimand, reproduced offline and
                    validated against the stored heldout_lpd.parquet
  AR-conditional    the honest forecast: same class posterior, but the last
                    fitted deviation carried forward,
                    mu_k(a_h) + rho_k^G (y_last - mu_k(a_last)), with the
                    gap-adjusted predictive variance

scored against two benchmarks fitted on the fitted rows only: an
age-quadratic Gaussian, and last-observation-carried-forward. Metrics are
log predictive density per held-out observation, RMSE/MAE/R-squared on the
standardised channel scale, and central-interval coverage.

    python clustering/oos_assessment.py --draws-dir <dir> [--draws 200]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR

JOBS = [
    ("pcs-ar1-ho", "pcs_lifecycle_20_89_minobs3_v1", "sf12pcs_dv", "PCS"),
    ("physgrm-ar1-ho", "physgrm_lifecycle_20_89_minobs3_v1",
     "theta_phys_full", "GRM physical"),
    ("combgrm-ar1-ho", "combgrm_lifecycle_20_89_minobs3_v1",
     "theta_combined", "GRM combined"),
]
HOLD_K, MIN_OBS, K = 2, 5, 3
AGE_CENTER, AGE_SCALE = 55, 10.0
LOG2PI = float(np.log(2 * np.pi))


def norm_lpdf(y, mu, sd):
    return -0.5 * LOG2PI - np.log(sd) - 0.5 * ((y - mu) / sd) ** 2


def prepare(contract: str, channel: str):
    long = pd.read_csv(
        PROCESSED_DATA_DIR / "contracts" / contract / "long.csv"
    ).sort_values(["pidp", "age"]).reset_index(drop=True)
    n = long.groupby("pidp")["age"].transform("size")
    frame = long[n >= MIN_OBS].reset_index(drop=True)
    mean, sd = frame[channel].mean(), frame[channel].std(ddof=1)
    frame["z"] = (frame[channel] - mean) / sd
    frame["a"] = (frame["age"] - AGE_CENTER) / AGE_SCALE
    frame["rank_desc"] = frame.groupby("pidp").cumcount(ascending=False)
    frame["is_held"] = frame["rank_desc"] < HOLD_K
    frame["new_person"] = frame["pidp"].ne(frame["pidp"].shift())
    frame["gap"] = frame["age"].diff().clip(lower=1.0).fillna(1.0)
    return frame


def class_loglik_block(z, a, gap, window_start, coef, sigma, rho):
    """(rows, K) AR(1) log density, restarting at each window_start row."""
    design = np.column_stack([np.ones_like(a), a, a**2])
    mu = design @ coef.T
    lp = np.empty_like(mu)
    stat_sd = sigma / np.sqrt(1 - rho**2)
    lp[window_start] = norm_lpdf(z[window_start, None], mu[window_start],
                                 stat_sd[None, :])
    rest = ~window_start
    g = gap[rest][:, None]
    rho_gap = rho[None, :] ** g
    var_mult = (1 - rho[None, :] ** (2 * g)) / (1 - rho[None, :] ** 2)
    prev_mu = np.roll(mu, 1, axis=0)[rest]
    prev_z = np.roll(z, 1)[rest][:, None]
    lp[rest] = norm_lpdf(z[rest, None], mu[rest] + rho_gap * (prev_z - prev_mu),
                         sigma[None, :] * np.sqrt(np.maximum(var_mult, 1e-9)))
    return lp, mu


def metrics(name, tag, label, y, pred, lpd, sd_pred):
    err = y - pred
    return {"fit": tag, "channel": label, "predictor": name, "n_rows": len(y),
            "lpd_per_obs": float(np.mean(lpd)),
            "rmse": float(np.sqrt((err**2).mean())),
            "mae": float(np.abs(err).mean()),
            "r2": float(1 - (err**2).sum() / ((y - y.mean())**2).sum()),
            "cover50": float((np.abs(err / sd_pred) < 0.6745).mean()),
            "cover90": float((np.abs(err / sd_pred) < 1.6449).mean())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--draws-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    out_rows, checks = [], []
    for tag, contract, channel, label in JOBS:
        frame = prepare(contract, channel)
        held = frame["is_held"].to_numpy()
        z, a = frame["z"].to_numpy(), frame["a"].to_numpy()
        gap = frame["gap"].to_numpy()
        person = pd.factorize(frame["pidp"])[0]
        n_person = person.max() + 1
        window_start = (frame["new_person"].to_numpy()
                        | (frame["rank_desc"].to_numpy() == HOLD_K - 1))
        held_idx = np.flatnonzero(held)
        held_person = person[held_idx]
        last_fit_idx = np.flatnonzero(frame["rank_desc"].to_numpy() == HOLD_K)
        anchor = last_fit_idx[held_person]
        age = frame["age"].to_numpy()
        big_gap = np.maximum(age[held_idx] - age[anchor], 1.0)[:, None]

        draws = pd.read_csv(args.draws_dir / f"{tag}.csv")
        draws = draws.iloc[:: max(1, len(draws) // args.draws)].reset_index(drop=True)
        D = len(draws)
        n_held = len(held_idx)

        ll_stan = np.zeros((D, n_person))
        lp_ar = np.zeros((D, n_held))
        lp_cls = np.zeros((D, n_held))
        pred_ar = np.zeros((D, n_held))
        var_ar = np.zeros((D, n_held))
        pred_cls = np.zeros((D, n_held))

        for d in range(D):
            r = draws.iloc[d]
            theta = np.array([r[f"theta.{k}"] for k in range(1, K + 1)])
            coef = np.array([[r[f"coef.1.{k}.{p}"] for p in range(1, 4)]
                             for k in range(1, K + 1)])
            sigma = np.repeat(float(r["sigma.1.1"]), K)
            rho = np.array([r[f"rho.{k}"] for k in range(1, K + 1)])

            lp_all, mu = class_loglik_block(z, a, gap, window_start,
                                            coef, sigma, rho)
            fit_lp = np.zeros((n_person, K))
            np.add.at(fit_lp, person[~held], lp_all[~held])
            unnorm = np.log(theta)[None, :] + fit_lp
            w = np.exp(unnorm - logsumexp(unnorm, axis=1, keepdims=True))

            hold_lp = np.zeros((n_person, K))
            np.add.at(hold_lp, person[held], lp_all[held])
            ll_stan[d] = (logsumexp(unnorm + hold_lp, axis=1)
                          - logsumexp(unnorm, axis=1))

            ww = w[held_person]
            stat_sd = sigma / np.sqrt(1 - rho**2)
            mu_h = mu[held_idx]
            lp_cls[d] = logsumexp(np.log(ww) + norm_lpdf(
                z[held_idx, None], mu_h, stat_sd[None, :]), axis=1)
            pred_cls[d] = (ww * mu_h).sum(axis=1)

            rho_gap = rho[None, :] ** big_gap
            pred_k = mu_h + rho_gap * (z[anchor, None] - mu[anchor])
            var_k = (sigma[None, :] ** 2
                     * (1 - rho[None, :] ** (2 * big_gap))
                     / (1 - rho[None, :] ** 2))
            m = (ww * pred_k).sum(axis=1)
            pred_ar[d] = m
            var_ar[d] = (ww * (var_k + pred_k**2)).sum(axis=1) - m**2
            lp_ar[d] = logsumexp(np.log(ww) + norm_lpdf(
                z[held_idx, None], pred_k, np.sqrt(var_k)), axis=1)

        mine = logsumexp(ll_stan, axis=0) - np.log(D)
        stan = pd.read_parquet(
            ARTIFACTS_DIR / "overnight" / tag / "heldout_lpd.parquet"
        )["lpd_heldout"].to_numpy()
        checks.append((tag, float(np.abs(mine - stan).max()),
                       float(np.corrcoef(mine, stan)[0, 1])))

        y = z[held_idx]
        fit_rows = ~held
        X = np.column_stack([np.ones(fit_rows.sum()), a[fit_rows],
                             a[fit_rows] ** 2])
        beta, *_ = np.linalg.lstsq(X, z[fit_rows], rcond=None)
        sd_age = (z[fit_rows] - X @ beta).std(ddof=3)
        Xh = np.column_stack([np.ones(n_held), a[held_idx], a[held_idx] ** 2])

        out_rows += [
            metrics("model, AR-conditional forecast", tag, label, y,
                    pred_ar.mean(axis=0),
                    logsumexp(lp_ar, axis=0) - np.log(D),
                    np.sqrt(var_ar.mean(axis=0) + pred_ar.var(axis=0))),
            metrics("model, class-only (the Stan estimand)", tag, label, y,
                    pred_cls.mean(axis=0),
                    logsumexp(lp_cls, axis=0) - np.log(D),
                    np.full(n_held, np.sqrt(
                        np.mean((y - pred_cls.mean(axis=0)) ** 2)))),
            metrics("age-quadratic null", tag, label, y, Xh @ beta,
                    norm_lpdf(y, Xh @ beta, sd_age), np.full(n_held, sd_age)),
            metrics("last observation carried forward", tag, label, y,
                    z[anchor],
                    norm_lpdf(y, z[anchor], (y - z[anchor]).std(ddof=1)),
                    np.full(n_held, (y - z[anchor]).std(ddof=1))),
        ]

    print("offline reproduction of the Stan held-out density:")
    for tag, dev, corr in checks:
        print(f"  {tag:16s} max |dev| {dev:.2e}  corr {corr:.8f}")
    tab = pd.DataFrame(out_rows)
    out = ARTIFACTS_DIR / "descriptives" / "oos_assessment.csv"
    tab.to_csv(out, index=False)
    print()
    print(tab.round(4).to_string(index=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
