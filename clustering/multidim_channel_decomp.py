"""Per-channel held-out decomposition for the multidimensional holdout fits.

The model reports one held-out density per person over a window that mixes
three channels. Chronic is nearly free to predict -- carry-forward alone gets
93.4% of held-out values exactly right -- so a single headline number is
flattered by it. This script recomputes the held-out density offline from the
posterior draws, channel by channel, and scores each against its own naive
benchmark on the same rows.

Validated the same way as the univariate assessment: the sum of the channel
pieces must reproduce the model's own stored per-person density.

    python clustering/multidim_channel_decomp.py --draws-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR
from prevention_health_clustering.runner.multidim import (
    GAUSS_CHANNELS, build_multidim_payload)

K, HOLD_K, MIN_OBS = 3, 2, 5
LOG2PI = float(np.log(2 * np.pi))


def norm_lpdf(y, mu, sd):
    return -0.5 * LOG2PI - np.log(sd) - 0.5 * ((y - mu) / sd) ** 2


def nb2_log_lpmf(y, log_mu, phi):
    mu = np.exp(log_mu)
    return (gammaln(y + phi) - gammaln(phi) - gammaln(y + 1)
            + phi * (np.log(phi) - np.log(phi + mu))
            + y * (log_mu - np.log(phi + mu)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws-dir", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=200)
    args = ap.parse_args(argv)

    long = pd.read_csv(PROCESSED_DATA_DIR / "contracts"
                       / "multidim_lifecycle_20_89_minobs3_v1" / "long.csv")
    rows = []
    for tag, ar in (("holdout", 0), ("ar1-holdout", 1)):
        pl = build_multidim_payload(long, ar_mode=ar, holdout_last_k=HOLD_K,
                                    min_person_obs=MIN_OBS, use_mortality=False)
        d = pl.data
        X = np.asarray(d["X"]); Y = np.asarray(d["y"])
        gap = np.asarray(d["age_gap"]) if ar else np.zeros(d["N_obs"])
        cobs = np.asarray(d["chronic_obs"]) == 1
        cy = np.asarray(d["chronic_y"], float)
        fs = np.asarray(d["fit_start"]); fe = np.asarray(d["fit_end"])
        hs = np.asarray(d["hold_start"]); he = np.asarray(d["hold_end"])
        n_person = d["N_person"]
        person = np.zeros(d["N_obs"], int)
        for i in range(n_person):
            person[fs[i] - 1:he[i]] = i
        held = np.zeros(d["N_obs"], bool)
        for i in range(n_person):
            held[hs[i] - 1:he[i]] = True
        # Two openings for the held-out window. Marking hold_start as a fresh
        # window start reproduces log_lik_heldout_marginal (the class-only
        # estimand); leaving it unmarked lets the AR recursion run on from the
        # last fitted row, which is log_lik_heldout (AR-conditional). Both are
        # computed so the channel split is available for each.
        wstart_marg = np.zeros(d["N_obs"], bool)
        wstart_marg[fs - 1] = True
        wstart_marg[hs - 1] = True
        wstart_ar = np.zeros(d["N_obs"], bool)
        wstart_ar[fs - 1] = True

        draws = pd.read_csv(args.draws_dir / f"{tag}.csv")
        draws = draws.iloc[:: max(1, len(draws) // args.draws)].reset_index(drop=True)
        D = len(draws)
        acc = {(m, c): np.zeros((D, n_person))
               for m in ("ar", "marg")
               for c in ("gauss1", "gauss2", "chronic", "all")}

        for di in range(D):
            r = draws.iloc[di]
            theta = np.array([r[f"theta.{k}"] for k in range(1, K + 1)])
            coef = np.array([[[r[f"coef.{c}.{k}.{p}"] for p in (1, 2, 3)]
                              for k in range(1, K + 1)] for c in (1, 2)])
            sigma = np.array([r[f"sigma.1.{c}"] for c in (1, 2)])
            rho = (np.array([r[f"rho.{k}"] for k in range(1, K + 1)])
                   if ar else np.zeros(K))
            cch = np.array([[r[f"coef_chronic.{k}.{p}"] for p in (1, 2, 3)]
                            for k in range(1, K + 1)])
            phi = float(r["phi_chronic"])

            # per-row, per-class log density per channel, under both openings
            for mode, wstart in (("ar", wstart_ar), ("marg", wstart_marg)):
                lp = {}
                for ci in range(2):
                    mu = X @ coef[ci].T
                    out = np.empty_like(mu)
                    if ar:
                        out[wstart] = norm_lpdf(Y[ci][wstart, None], mu[wstart],
                                                (sigma[ci] / np.sqrt(1 - rho**2))[None, :])
                        rest = ~wstart
                        g = gap[rest][:, None]
                        vm = (1 - rho[None, :] ** (2 * g)) / (1 - rho[None, :] ** 2)
                        out[rest] = norm_lpdf(
                            Y[ci][rest, None],
                            mu[rest] + rho[None, :] ** g
                            * (np.roll(Y[ci], 1)[rest][:, None] - np.roll(mu, 1, 0)[rest]),
                            sigma[ci] * np.sqrt(np.maximum(vm, 1e-9)))
                    else:
                        out = norm_lpdf(Y[ci][:, None], mu, sigma[ci])
                    lp[f"gauss{ci+1}"] = out
                lmu = X @ cch.T
                lpc = np.zeros_like(lmu)
                lpc[cobs] = nb2_log_lpmf(cy[cobs, None], lmu[cobs], phi)
                lp["chronic"] = lpc

                fit_lp = np.zeros((n_person, K))
                for c in lp:
                    np.add.at(fit_lp, person[~held], lp[c][~held])
                unnorm = np.log(theta)[None, :] + fit_lp
                base = logsumexp(unnorm, axis=1)
                for name in ("gauss1", "gauss2", "chronic"):
                    h = np.zeros((n_person, K))
                    np.add.at(h, person[held], lp[name][held])
                    acc[(mode, name)][di] = logsumexp(unnorm + h, axis=1) - base
                hall = np.zeros((n_person, K))
                for c in lp:
                    np.add.at(hall, person[held], lp[c][held])
                acc[(mode, "all")][di] = logsumexp(unnorm + hall, axis=1) - base

        stored = pd.read_parquet(ARTIFACTS_DIR / "multidim" / tag
                                 / "heldout_lpd.parquet")["lpd_heldout"].to_numpy()
        mine = logsumexp(acc[("ar", "all")], axis=0) - np.log(D)
        print(f"[{tag}] offline AR-conditional vs stored: corr "
              f"{np.corrcoef(mine, stored)[0,1]:.6f}, "
              f"max |dev| {np.abs(mine - stored).max():.3f}")

        # carry-forward benchmark on the held-out chronic rows: does the
        # channel's near-monotonicity make it free to predict?
        cf_hits, cf_n = 0, 0
        for i in range(n_person):
            anchor = fe[i] - 1
            if not cobs[anchor]:
                continue
            for n in range(hs[i] - 1, he[i]):
                if cobs[n]:
                    cf_n += 1
                    cf_hits += int(cy[n] == cy[anchor])
        print(f"        carry-forward exactly right on {cf_hits/cf_n:.1%} of "
              f"{cf_n:,} held-out chronic rows")
        n_held_rows = int(held.sum())
        n_chron_held = int((held & cobs).sum())
        for mode in ("ar", "marg"):
            for name, nrows in (("theta physical", n_held_rows),
                                ("theta mental", n_held_rows),
                                ("chronic count", n_chron_held),
                                ("ALL THREE", n_held_rows * 2 + n_chron_held)):
                key = {"theta physical": "gauss1", "theta mental": "gauss2",
                       "chronic count": "chronic", "ALL THREE": "all"}[name]
                lpd = logsumexp(acc[(mode, key)], axis=0) - np.log(D)
                rows.append({"fit": tag, "estimand": mode, "channel": name,
                             "held_rows": nrows,
                             "lpd_per_person": float(lpd.mean()),
                             "lpd_per_held_row": float(lpd.sum() / nrows)})

    tab = pd.DataFrame(rows)
    out = ARTIFACTS_DIR / "descriptives" / "multidim_channel_decomp.csv"
    tab.to_csv(out, index=False)
    print()
    print(tab.round(4).to_string(index=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
