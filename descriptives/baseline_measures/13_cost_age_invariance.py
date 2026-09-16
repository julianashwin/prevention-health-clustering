"""Is the health-cost relationship age-invariant, measure by measure?

The proportional gradient — log points of cost per standard deviation sicker —
asks whether health discriminates more at older ages; the absolute gradient in
pounds almost has to rise with age because the base level of spending does.
Specification and standard errors as in descriptives/17_cost_age_interaction.py:
band-by-band gradients, then one interaction test with age in decades centred
at 50, health and its square, and person-clustered standard errors. Each
measure uses its own cost sample.

Writes data/processed/baseline_measures/cost_age_invariance.csv.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from prevention_health_clustering.config import PROCESSED_DATA_DIR

OUT = PROCESSED_DATA_DIR / "baseline_measures"
COST = "flat_cost_total"
BANDS = [(20, 34), (35, 44), (45, 54), (55, 64), (65, 74), (75, 90)]
LAB = ["20-34", "35-44", "45-54", "55-64", "65-74", "75-90"]
SHOW = [f"{b} {s}" for b in ("P-FUNC", "P-LIM", "P-LIM3", "P-LIM3+CC") for s in ("h", "theta", "FS", "sum")] + \
    ["P-FULL h", "P-FULL theta", "PHYS-4", "GRM h", "GRM theta"]   # printed; every measure is computed


def ols_cluster(X, y, groups):
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    u = y - X @ b
    order = np.argsort(groups, kind="stable")
    Xs, us, gs = X[order], u[order], groups[order]
    edges = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    meat = np.zeros((X.shape[1], X.shape[1]))
    for a, z in zip(edges[:-1], edges[1:]):
        s = Xs[a:z].T @ us[a:z]
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv
    return b, np.sqrt(np.diag(V))


def main() -> int:
    cost = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "cost_index.parquet",
                           columns=["pidp", "wave", COST])
    meas = pd.read_parquet(OUT / "master_measures.parquet")
    d0 = meas.merge(cost, on=["pidp", "wave"], how="inner")
    d0 = d0[d0[COST].notna() & d0["age"].between(20, 90)]
    keys = [c for c in meas.columns if c not in ("pidp", "wave", "age")]
    rows = []
    for key in keys:
        d = d0[d0[key].notna()].copy()
        d["z"] = (d[key] - d[key].mean()) / d[key].std()
        d["logc"] = np.log(d[COST] + 1.0)
        d["band"] = pd.cut(d["age"], [b[0] - 1 for b in BANDS] + [90], labels=LAB)
        r = {"key": key, "n": len(d), "people": d["pidp"].nunique()}
        for lab in LAB:
            s = d[d["band"] == lab]
            X = np.column_stack([np.ones(len(s)), s["z"]])
            b_log, _ = ols_cluster(X, s["logc"].to_numpy(float), s["pidp"].to_numpy())
            b_abs, _ = ols_cluster(X, s[COST].to_numpy(float), s["pidp"].to_numpy())
            r[f"log_{lab}"] = -b_log[1]
            r[f"abs_{lab}"] = -b_abs[1]
        a = (d["age"].to_numpy(float) - 50) / 10.0
        z = d["z"].to_numpy()
        X = np.column_stack([np.ones(len(d)), a, a**2, z, z**2, z * a])
        for name, y in (("abs", d[COST].to_numpy(float)), ("log", d["logc"].to_numpy(float))):
            b, se = ols_cluster(X, y, d["pidp"].to_numpy())
            r[f"{name}_interaction"], r[f"{name}_interaction_t"] = b[5], b[5] / se[5]
            r[f"{name}_curv"], r[f"{name}_curv_t"] = b[4], b[4] / se[4]
        r["log_ratio_old_young"] = r["log_75-90"] / r["log_20-34"]
        rows.append(r)
        if key in SHOW:
            print(f"{key:14s} n {len(d):7,}  log gradient per sd sicker: "
              + " ".join(f"{r[f'log_{lab}']:5.2f}" for lab in LAB)
              + f"  | old/young {r['log_ratio_old_young']:4.2f}"
                  + f"  | z x age {r['log_interaction']:+.4f} (t {r['log_interaction_t']:5.1f})", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "cost_age_invariance.csv", index=False)
    print(f"\nwrote cost_age_invariance.csv, {len(t)} measures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
