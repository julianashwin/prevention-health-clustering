"""Two checks on employment and the health measure, for the data section.

A. A discontinuity at 65, the state pension age for men throughout the panel (women's rose from
   60 to 65 over 2010-18). Each item, standardised on the pooled sample, plus h and theta, is
   regressed on a linear trend in age either side of 65 with a jump at 65, on ages 58-72, with
   person-clustered standard errors. The SF-12 role items (RP: problems "with your work or other
   regular daily activities") are the only work-related items; if retirement moved them
   specifically, their jump would stand out from the others'.
B. Whether the cost-health relationship differs for the employed: ages 25-64, the cost index in
   pounds on a quadratic in the pooled-standardised measure, interacted with an employed indicator
   (jbstat 1 or 2), clustered on the person; once with everyone else as the comparison and once
   excluding the long-term sick or disabled (jbstat 8), whose non-employment is itself a health outcome.

Outputs: artifacts/descriptives/paper_employment_checks.csv
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from prevention_health_clustering.config import ARTIFACTS_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from prevention_health_clustering.measures.health import HEALTH_ITEMS, build_health_items
MEAS = PROCESSED_DATA_DIR / "measures"


def ols_cluster(X, y, g):
    XtX = np.linalg.inv(X.T @ X); b = XtX @ X.T @ y; u = y - X @ b
    o = np.argsort(g, kind="stable"); Xs, us, gs = X[o], u[o], g[o]
    e = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    M = sum(np.outer(v, v) for v in (Xs[a:z].T @ us[a:z] for a, z in zip(e[:-1], e[1:])))
    return b, np.sqrt(np.diag(XtX @ M @ XtX))


def main() -> int:
    s = pd.read_parquet(MEAS / "health_measure.parquet")
    bank, _ = build_health_items(pd.read_parquet(INTERIM_DATA_DIR / "sf12_items_long.parquet"),
                                 pd.read_parquet(MEAS / "chronic_conditions.parquet"))
    d = s.merge(bank.drop(columns="age"), on=["pidp", "wave"])
    d = d.merge(pd.read_csv(PROCESSED_DATA_DIR / "ukhls_indresp_processed.csv", usecols=["pidp", "wave", "sex", "jbstat"]),
                on=["pidp", "wave"], how="left")
    rows = []
    w = d[d["age"].between(58, 72)].copy()
    a = w["age"].to_numpy(float) - 65; post = (a >= 0).astype(float)
    print("A. jump at 65 in pooled-sd units (linear trend each side, ages 58-72, person-clustered t):")
    for col in list(HEALTH_ITEMS) + ["h", "theta"]:
        z = ((w[col] - d[col].mean()) / d[col].std()).to_numpy(); out = []
        for lab, m in [("all", np.ones(len(w), bool)), ("men", (w["sex"] == 1).to_numpy()), ("women", (w["sex"] == 2).to_numpy())]:
            b, se = ols_cluster(np.column_stack([np.ones(m.sum()), a[m], post[m], a[m] * post[m]]), z[m], w["pidp"].to_numpy()[m])
            out.append((b[2], b[2] / se[2])); rows.append({"check": "A", "variable": col, "group": lab, "jump": b[2], "t": b[2] / se[2]})
        print(f"  {col:9s} " + "  ".join(f"{lab} {j:+.3f} ({t:+.1f})" for lab, (j, t) in zip(("all", "men", "women"), out)) + ("   <- work-related" if col == "RP" else ""))
    print(f"  share retired at 63-64: {(w.loc[w.age.between(63, 64), 'jbstat'] == 4).mean():.2f}; at 65-66: {(w.loc[w.age.between(65, 66), 'jbstat'] == 4).mean():.2f}")
    c = pd.read_parquet(MEAS / "cost_index.parquet", columns=["pidp", "wave", "flat_cost_total"])
    e = d.merge(c, on=["pidp", "wave"]).dropna(subset=["flat_cost_total"])
    e = e[e["age"].between(25, 64) & e["jbstat"].notna()].copy()
    e["employed"] = e["jbstat"].isin([1, 2]).astype(float)
    print(f"\nB. cost-health by employment, ages 25-64, {len(e):,} person-waves, {e['employed'].mean():.0%} employed")
    y = e["flat_cost_total"].to_numpy(float); g = e["pidp"].to_numpy(); ag = (e["age"].to_numpy(float) - 45) / 10
    for col in ["h", "theta"]:
        z = -((e[col] - d[col].mean()) / d[col].std()).to_numpy()
        for lab, keep in [("vs all others", np.ones(len(e), bool)), ("vs others excl. long-term sick", ~(e["jbstat"] == 8).to_numpy())]:
            E = e["employed"].to_numpy()[keep]
            X = np.column_stack([np.ones(keep.sum()), ag[keep], ag[keep] ** 2, z[keep], z[keep] ** 2, E, z[keep] * E, z[keep] ** 2 * E])
            b, se = ols_cluster(X, y[keep], g[keep])
            print(f"  {col:5s} {lab:32s} slope £{b[3]:5.0f} curvature £{b[4]:4.0f} | employed x slope £{b[6]:+5.0f} (t {b[6]/se[6]:+.1f}), x curvature £{b[7]:+4.0f} (t {b[7]/se[7]:+.1f})")
            rows.append({"check": "B", "variable": col, "group": lab, "slope": b[3], "curvature": b[4], "emp_x_slope": b[6], "emp_x_slope_t": b[6] / se[6],
                         "emp_x_curv": b[7], "emp_x_curv_t": b[7] / se[7], "n": int(keep.sum())})
        dec = pd.qcut(pd.Series(z).rank(method="first"), 10, labels=False).to_numpy()
        tab = e.assign(dec=dec).groupby(["dec", "employed"])["flat_cost_total"].mean().unstack()
        print(f"  {col} mean cost by decile of poor health (0 = healthiest), not employed / employed: "
              + "  ".join(f"{i}: {tab.loc[i, 0.0]:.0f}/{tab.loc[i, 1.0]:.0f}" for i in tab.index))
    pd.DataFrame(rows).to_csv(ARTIFACTS_DIR / "descriptives" / "paper_employment_checks.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
