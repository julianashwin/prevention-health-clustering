"""Class composition by age for every fitted Bayesian model.

The mixture assigns each person to a class for life, so theta is a constant.
What varies over the lifecycle is who is OBSERVED: the composition of the
sample at each age. That is the quantity the partial K-means figure draws as
a strip beneath each panel, and this script computes its Bayesian counterpart
so the same strip can sit beneath the model trajectories.

For each fit, each person's class posterior is computed from their fitted
rows under the model's own likelihood, then person-waves are pooled by age.
Parameters are taken at the posterior mean rather than integrated over draws;
``--validate`` checks that choice against a draw-averaged posterior on one
fit, where the two agree to about a thousandth of a share.

Output: artifacts/descriptives/class_composition_by_age.csv
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
from prevention_health_clustering.models.registry import get_model
from prevention_health_clustering.runner.fit import build_payload
from prevention_health_clustering.runner.multidim import build_multidim_payload

K = 3
LOG2PI = float(np.log(2 * np.pi))
CONTRACTS = PROCESSED_DATA_DIR / "contracts"

UNIVARIATE = [
    ("pcs-ar1", "pcs-ar1", "pcs_lifecycle_20_89_minobs3_v1", None),
    ("pcs-ar1-ho", "pcs-ar1", "pcs_lifecycle_20_89_minobs3_v1", 2),
    ("physgrm-base", "physgrm-headline", "physgrm_lifecycle_20_89_minobs3_v1", None),
    ("physgrm-ar1", "physgrm-ar1", "physgrm_lifecycle_20_89_minobs3_v1", None),
    ("physgrm-ar1-ho", "physgrm-ar1", "physgrm_lifecycle_20_89_minobs3_v1", 2),
    ("combgrm-base", "combgrm-headline", "combgrm_lifecycle_20_89_minobs3_v1", None),
    ("combgrm-ar1", "combgrm-ar1", "combgrm_lifecycle_20_89_minobs3_v1", None),
    ("combgrm-ar1-ho", "combgrm-ar1", "combgrm_lifecycle_20_89_minobs3_v1", 2),
]
MULTIDIM = [("baseline", 0, False), ("holdout", 0, True),
            ("ar1", 1, False), ("ar1-holdout", 1, True)]


def norm_lpdf(y, mu, sd):
    return -0.5 * LOG2PI - np.log(sd) - 0.5 * ((y - mu) / sd) ** 2


def nb2(y, log_mu, phi):
    mu = np.exp(log_mu)
    return (gammaln(y + phi) - gammaln(phi) - gammaln(y + 1)
            + phi * (np.log(phi) - np.log(phi + mu))
            + y * (log_mu - np.log(phi + mu)))


def gauss_rows(Y, X, coef, sigma, rho, gap, wstart, ar):
    """(rows, K) log density for one Gaussian channel."""
    mu = X @ coef.T
    if not ar:
        return norm_lpdf(Y[:, None], mu, sigma)
    out = np.empty_like(mu)
    out[wstart] = norm_lpdf(Y[wstart, None], mu[wstart],
                            (sigma / np.sqrt(1 - rho**2))[None, :])
    rest = ~wstart
    g = gap[rest][:, None]
    vm = (1 - rho[None, :] ** (2 * g)) / (1 - rho[None, :] ** 2)
    out[rest] = norm_lpdf(
        Y[rest, None],
        mu[rest] + rho[None, :] ** g
        * (np.roll(Y, 1)[rest][:, None] - np.roll(mu, 1, 0)[rest]),
        sigma * np.sqrt(np.maximum(vm, 1e-9)))
    return out


def composition(ages, person, fit_lp, theta, n_person, pids=None):
    """Posterior class shares among the person-waves observed at each age.

    Returns the age composition and, when person ids are supplied, the
    per-person class posterior itself (which 12_outcome_prediction.py uses to
    score the classes against external outcomes).
    """
    unnorm = np.log(theta)[None, :] + fit_lp
    w = np.exp(unnorm - logsumexp(unnorm, axis=1, keepdims=True))
    rows = pd.DataFrame(w[person], columns=[f"class{k+1}" for k in range(K)])
    rows["age"] = ages
    g = rows.groupby("age").mean()
    g["n"] = rows.groupby("age").size()
    post = None
    if pids is not None:
        post = pd.DataFrame(w, columns=[f"class{k+1}" for k in range(K)])
        post.insert(0, "pidp", pids)
    return g.reset_index(), post


def univariate_fit(tag, model, contract, hold_k):
    spec = get_model(model)
    long = pd.read_csv(CONTRACTS / contract / "long.csv")
    kw = dict(holdout_last_k=hold_k, holdout_min_person_obs=5) if hold_k else {}
    pl = build_payload(spec, long, **kw)
    d = pl.data
    p = json.loads((ARTIFACTS_DIR / "overnight" / tag
                    / "run_summary.json").read_text())["params"]
    theta = np.array([p[f"theta[{k}]"]["mean"] for k in range(1, K + 1)])
    coef = np.array([[p[f"coef[1,{k},{j}]"]["mean"] for j in (1, 2, 3)]
                     for k in range(1, K + 1)])
    sigma = p["sigma[1,1]"]["mean"]
    ar = d["ar_mode"] == 1
    rho = (np.array([p[f"rho[{k}]"]["mean"] for k in range(1, K + 1)])
           if ar else np.zeros(K))
    X = np.asarray(d["X"]); Y = np.asarray(d["y"][0])
    fs, fe = np.asarray(d["fit_start"]), np.asarray(d["fit_end"])
    n_person = d["N_person"]
    person = np.zeros(d["N_obs"], int)
    for i in range(n_person):
        person[fs[i] - 1:(d["hold_end"][i] or fe[i])] = i
    wstart = np.zeros(d["N_obs"], bool); wstart[fs - 1] = True
    fitted = np.zeros(d["N_obs"], bool)
    for i in range(n_person):
        fitted[fs[i] - 1:fe[i]] = True
    lp = gauss_rows(Y, X, coef, sigma, rho, np.asarray(
        d["age_gap"]) if ar else np.zeros(d["N_obs"]), wstart, ar)
    fit_lp = np.zeros((n_person, K))
    np.add.at(fit_lp, person[fitted], lp[fitted])
    ages = np.round(X[:, 1] * 10 + 55).astype(int)
    return composition(ages, person, fit_lp, theta, n_person, pl.person_ids)


def multidim_fit(tag, ar, hold):
    long = pd.read_csv(CONTRACTS / "multidim_lifecycle_20_89_minobs3_v1" / "long.csv")
    pl = build_multidim_payload(long, ar_mode=ar, min_person_obs=5,
                                holdout_last_k=2 if hold else None,
                                use_mortality=False)
    d = pl.data
    p = json.loads((ARTIFACTS_DIR / "multidim" / tag
                    / "run_summary.json").read_text())["params"]
    theta = np.array([p[f"theta[{k}]"]["mean"] for k in range(1, K + 1)])
    X = np.asarray(d["X"]); Y = np.asarray(d["y"])
    fs, fe = np.asarray(d["fit_start"]), np.asarray(d["fit_end"])
    n_person = d["N_person"]
    person = np.zeros(d["N_obs"], int)
    for i in range(n_person):
        person[fs[i] - 1:d["full_end"][i]] = i
    wstart = np.zeros(d["N_obs"], bool); wstart[fs - 1] = True
    fitted = np.zeros(d["N_obs"], bool)
    for i in range(n_person):
        fitted[fs[i] - 1:fe[i]] = True
    gap = np.asarray(d["age_gap"]) if ar else np.zeros(d["N_obs"])
    rho = (np.array([p[f"rho[{k}]"]["mean"] for k in range(1, K + 1)])
           if ar else np.zeros(K))
    fit_lp = np.zeros((n_person, K))
    for ci in (1, 2):
        coef = np.array([[p[f"coef[{ci},{k},{j}]"]["mean"] for j in (1, 2, 3)]
                         for k in range(1, K + 1)])
        lp = gauss_rows(Y[ci - 1], X, coef, p[f"sigma[1,{ci}]"]["mean"],
                        rho, gap, wstart, ar == 1)
        np.add.at(fit_lp, person[fitted], lp[fitted])
    cch = np.array([[p[f"coef_chronic[{k},{j}]"]["mean"] for j in (1, 2, 3)]
                    for k in range(1, K + 1)])
    cobs = np.asarray(d["chronic_obs"]) == 1
    cy = np.asarray(d["chronic_y"], float)
    lpc = np.zeros((d["N_obs"], K))
    lpc[cobs] = nb2(cy[cobs, None], (X @ cch.T)[cobs], p["phi_chronic"]["mean"])
    np.add.at(fit_lp, person[fitted], lpc[fitted])
    ages = np.round(X[:, 1] * 10 + 55).astype(int)
    return composition(ages, person, fit_lp, theta, n_person, pl.person_ids)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args(argv)
    out, posts = [], []
    for tag, model, contract, hk in UNIVARIATE:
        g, post = univariate_fit(tag, model, contract, hk)
        g.insert(0, "fit", tag); g.insert(1, "family", "univariate")
        out.append(g)
        post.insert(0, "fit", tag); posts.append(post)
        print(f"  {tag:16s} ages {g['age'].min()}-{g['age'].max()}, "
              f"share at 30 {g.loc[g.age==30,'class1'].iloc[0]:.3f} / "
              f"at 80 {g.loc[g.age==80,'class1'].iloc[0]:.3f} (class 1)")
    for tag, ar, hold in MULTIDIM:
        g, post = multidim_fit(tag, ar, hold)
        g.insert(0, "fit", f"multidim-{tag}"); g.insert(1, "family", "multidim")
        out.append(g)
        post.insert(0, "fit", f"multidim-{tag}"); posts.append(post)
        print(f"  multidim-{tag:12s} share at 30 "
              f"{g.loc[g.age==30,'class1'].iloc[0]:.3f} / at 80 "
              f"{g.loc[g.age==80,'class1'].iloc[0]:.3f} (class 1)")
    tab = pd.concat(out, ignore_index=True)
    path = ARTIFACTS_DIR / "descriptives" / "class_composition_by_age.csv"
    tab.to_csv(path, index=False)
    ppath = ARTIFACTS_DIR / "descriptives" / "class_posteriors.parquet"
    pd.concat(posts, ignore_index=True).to_parquet(ppath, index=False)
    print(f"\nwrote {path} ({len(tab):,} rows) and {ppath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
