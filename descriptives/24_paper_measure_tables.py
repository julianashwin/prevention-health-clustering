"""The measure section's small tables: comparison metrics, mortality, cost by age.

1. Correlations of the three variants (theta, h, fi10) with the measures the
   paper compares against: the UKHLS PCS and MCS, and the chronic-condition
   count (negated so higher = better), on the person-waves where both exist.
2. Mortality: the log-odds of dying before the next wave per standard
   deviation of each variant, overall and within age bands, and the death rate
   in the least healthy tenth over the healthiest. Death before the next wave
   is read from the cross-wave death record (dcsedw_dv, UKHLS wave = code - 18);
   person-waves at wave 15 have no next wave and are excluded.
3. Cost: the proportional gradient, log points of cost per standard deviation
   sicker, within the same age bands, with the variant x age interaction and
   person-clustered standard errors, as in 17_cost_age_interaction.py.

Outputs: paper/tables/tab_measure_correlations.tex, tab_measure_mortality.tex,
         tab_measure_cost_age.tex; artifacts/descriptives/paper_measure_tables.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, ROOT_DIR

MEAS = PROCESSED_DATA_DIR / "measures"
TAB = ROOT_DIR / "paper" / "tables"
VARIANTS = [("theta", r"$\theta$"), ("h", "$h$"), ("fi10", "deficit index")]
BANDS = [(20, 44), (45, 64), (65, 90)]
COST_BANDS = [(20, 34), (35, 44), (45, 54), (55, 64), (65, 74), (75, 90)]


def logit(X, y, iters=25):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        b = b + np.linalg.solve(H, X.T @ (y - p))
    p = 1 / (1 + np.exp(-X @ b))
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))))
    return b, se


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
    return b, np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))


def z(s):
    return (s - s.mean()) / s.std()


def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True)
    hm = pd.read_parquet(MEAS / "health_measure.parquet")
    sf = pd.read_parquet(MEAS / "sf12_measures.parquet", columns=["pidp", "wave", "sf12pcs_dv", "sf12mcs_dv"])
    ch = pd.read_parquet(MEAS / "chronic_conditions.parquet")
    ncols = [c for c in ch.columns if c.startswith("n_")]
    ch = ch.assign(n_chronic=ch[ncols].sum(axis=1))[["pidp", "wave", "n_chronic"]]
    d = hm.merge(sf, on=["pidp", "wave"], how="left").merge(ch, on=["pidp", "wave"], how="left")
    d["neg_chronic"] = -d["n_chronic"]
    rows = []

    # 1. correlations
    comp = [("sf12pcs_dv", "UKHLS PCS"), ("sf12mcs_dv", "UKHLS MCS"), ("neg_chronic", "chronic count (negated)")]
    corr = pd.DataFrame({lab: [d[[v, c]].corr().iloc[0, 1] for c, _ in comp] for v, lab in VARIANTS},
                        index=[n for _, n in comp])
    corr.loc["Spearman, vs $h$"] = [d[[v, "h"]].corr(method="spearman").iloc[0, 1] for v, _ in VARIANTS]
    n_corr = d[["sf12pcs_dv", "n_chronic"]].notna().all(axis=1).sum()
    with open(TAB / "tab_measure_correlations.tex", "w") as f:
        f.write("\\begin{tabular}{l" + "r" * len(VARIANTS) + "}\n\\toprule\n")
        f.write(" & " + " & ".join(lab for _, lab in VARIANTS) + " \\\\\n\\midrule\n")
        for r, row in corr.iterrows():
            f.write(f"{r} & " + " & ".join(f"{x:.3f}" for x in row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    for r, row in corr.iterrows():
        for v, _ in VARIANTS:
            rows.append({"table": "corr", "row": r, "variant": v, "value": row[[l for vv, l in VARIANTS if vv == v][0]]})
    print(f"correlations on {n_corr:,} person-waves\n", corr.round(3).to_string())

    # 2. mortality
    panel = pd.read_csv(PROCESSED_DATA_DIR / "ukhls_indresp_processed.csv", usecols=["pidp", "wave", "dcsedw_dv"])
    death_wave = (panel.groupby("pidp")["dcsedw_dv"].max() - 18)
    m = hm[hm["wave"] <= 14].merge(death_wave.rename("death_wave"), left_on="pidp", right_index=True, how="left")
    m["died_next"] = (m["death_wave"] == m["wave"] + 1).astype(int)
    m = m[~(m["death_wave"] <= m["wave"])]        # recorded dead before this wave: not at risk
    m["band"] = pd.cut(m["age"], [b[0] - 1 for b in BANDS] + [90], labels=[f"{a}--{b}" for a, b in BANDS])
    mort = []
    for v, lab in VARIANTS:
        rec = {"variant": lab}
        for name, s in [("all ages", m)] + [(str(b), m[m["band"] == b]) for b in m["band"].cat.categories]:
            X = np.column_stack([np.ones(len(s)), z(s[v]), z(s["age"]), z(s["age"]) ** 2])
            b, se = logit(X, s["died_next"].to_numpy(float))
            rec[name] = b[1]
            rec[f"{name} se"] = se[1]
        dec = pd.qcut(m[v], 10, labels=False, duplicates="drop")
        rate = m.groupby(dec)["died_next"].mean()
        rec["decile ratio"] = rate.iloc[0] / rate.iloc[-1]
        mort.append(rec)
    mort = pd.DataFrame(mort).set_index("variant")
    print(f"\nmortality: {len(m):,} person-waves at risk, {m['died_next'].sum():,} deaths before the next wave\n",
          mort.round(3).to_string())
    with open(TAB / "tab_measure_mortality.tex", "w") as f:
        f.write("\\begin{tabular}{l" + "r" * (len(BANDS) + 2) + "}\n\\toprule\n")
        f.write(" & all ages & " + " & ".join(f"{a}--{b}" for a, b in BANDS) + " & decile ratio \\\\\n\\midrule\n")
        for lab, r in mort.iterrows():
            cells = [f"{r['all ages']:.2f} ({r['all ages se']:.2f})"] + \
                [f"{r[str(b)]:.2f} ({r[str(b) + ' se']:.2f})" for b in m["band"].cat.categories]
            f.write(f"{lab} & " + " & ".join(cells) + f" & {r['decile ratio']:.0f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    for lab, r in mort.iterrows():
        for c in r.index:
            rows.append({"table": "mortality", "row": c, "variant": lab, "value": r[c]})

    # 3. cost gradient by age band
    cost = pd.read_parquet(MEAS / "cost_index.parquet", columns=["pidp", "wave", "flat_cost_total"])
    c = hm.merge(cost, on=["pidp", "wave"], how="inner").dropna(subset=["flat_cost_total"])
    c["logc"] = np.log(c["flat_cost_total"] + 1.0)
    c["band"] = pd.cut(c["age"], [b[0] - 1 for b in COST_BANDS] + [90], labels=[f"{a}--{b}" for a, b in COST_BANDS])
    ct = []
    for v, lab in VARIANTS:
        rec = {"variant": lab}
        for bnd, s in c.groupby("band", observed=True):
            zz = z(c[v]).loc[s.index].to_numpy()         # pooled standardisation, as in the note
            X = np.column_stack([np.ones(len(s)), zz, zz ** 2])
            b, se = ols_cluster(X, s["logc"].to_numpy(float), s["pidp"].to_numpy())
            rec[str(bnd)] = -b[1]
        zz = z(c[v]).to_numpy()
        dec = (c["age"].to_numpy() - 50) / 10
        X = np.column_stack([np.ones(len(c)), zz, zz ** 2, dec, dec ** 2, zz * dec])
        b, se = ols_cluster(X, c["logc"].to_numpy(float), c["pidp"].to_numpy())
        rec["interaction"], rec["interaction t"] = b[5], b[5] / se[5]
        ct.append(rec)
    ct = pd.DataFrame(ct).set_index("variant")
    print(f"\ncost: {len(c):,} person-waves; log points of cost per SD sicker, by age band\n", ct.round(3).to_string())
    with open(TAB / "tab_measure_cost_age.tex", "w") as f:
        f.write("\\begin{tabular}{l" + "r" * (len(COST_BANDS) + 1) + "}\n\\toprule\n")
        f.write(" & " + " & ".join(f"{a}--{b}" for a, b in COST_BANDS) + " & $z\\times$age ($t$) \\\\\n\\midrule\n")
        for lab, r in ct.iterrows():
            f.write(f"{lab} & " + " & ".join(f"{r[str(b)]:.2f}" for b in c["band"].cat.categories)
                    + f" & {r['interaction']:+.3f} ({r['interaction t']:.1f}) \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    for lab, r in ct.iterrows():
        for col in r.index:
            rows.append({"table": "cost", "row": col, "variant": lab, "value": r[col]})
    pd.DataFrame(rows).to_csv(ARTIFACTS_DIR / "descriptives" / "paper_measure_tables.csv", index=False)
    print(f"\nwrote three tables to {TAB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
