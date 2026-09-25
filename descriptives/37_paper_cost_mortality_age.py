"""Convexity of cost in health, mortality in health, and whether either changes with age (UKHLS).

The UKHLS twin of PSID-playground/analysis/descriptives/24b_cost_mortality_age.py: same bins, age
bands, knot, logits and pooled tests, so the two panels can be read side by side.

Top row, cost (the cost index in pounds, uncapped, waves 7-15):
  (a) mean annual cost in equal-width bins of h within three age bands (bin means, 95% intervals)
  (b) the same on theta
  (c) convexity by age band, quadratic: cost on the pooled-standardised measure, a + b z + c z^2,
      within each of six bands, person-clustered. The quadratic term c is the convexity the
      framework's cost function is written in; age-invariant means the series is flat. The pooled
      test interacts z and z^2 with age. Panel (c) uses costs capped at the 99th percentile
      (--uncapped-c to switch): a quadratic coefficient is driven by the top 1% of costs, and the
      binned means in (a)-(b) are not.
Bottom row, mortality (death before the next wave, one year on in UKHLS; two in PSID):
  (d) death rate in equal-width bins of h within three age bands (log scale, Wilson 95% intervals,
      bins with at least 10 deaths), with the within-band logit (linear in the measure, quadratic in
      age, evaluated at the band's mean age) drawn through them
  (e) the same on theta
  (f) log-odds of death per pooled SD sicker within six age bands, h and theta
The measure is standardised on the pooled sample, never within band.

Differences from the PSID version that come from the data, not the method: UKHLS mortality is one
year ahead, not two; the UKHLS index prices in-patient, out-patient and GP contacts, where the PSID
one prices hospital nights. `--inpatient` runs the in-patient tier alone, the like-for-like comparison.

The paper's figure has the four binned panels only (a, b, d, e); the by-band panels (c, f) and the
pooled tests are printed and written to the csv, and quoted in the text. --panels6 draws all six.

Flags: --inpatient (in-patient tier only), --cap-ab (cap (a)-(b) too), --uncapped-c (no cap in (c)),
       --panels6 (the six-panel version, to artifacts/scratch).

Outputs: paper/figures/fig_cost_mortality_age.png (the full index),
         artifacts/scratch/fig_cost_mortality_age_inpatient.png (with --inpatient),
         artifacts/descriptives/cost_mortality_age{,_inpatient}.csv and _tests.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import BLUE, INK2, VERM, apply_style  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, ROOT_DIR  # noqa: E402

MEAS = PROCESSED_DATA_DIR / "measures"
DESC = ARTIFACTS_DIR / "descriptives"
BANDS3 = [(20, 44, "20-44"), (45, 64, "45-64"), (65, 90, "65-90")]
BANDS6 = [(20, 34), (35, 44), (45, 54), (55, 64), (65, 74), (75, 90)]
BAND_COL = ["#9ECAE1", "#4292C6", "#08306B"]
EDGES_COST = {"h": np.arange(0.3, 1.0001, 0.05), "theta": np.arange(-3.0, 2.0001, 0.25)}
EDGES_MORT = {"h": np.arange(0.3, 1.0001, 0.1), "theta": np.arange(-3.0, 2.0001, 0.5)}
LAB = {"h": "$h$", "theta": r"$\theta$"}
CAP_Q = 0.99


def ols_cluster(X, y, g):
    XtX = np.linalg.inv(X.T @ X)
    b = XtX @ X.T @ y
    u = y - X @ b
    o = np.argsort(g, kind="stable")
    Xs, us, gs = X[o], u[o], g[o]
    e = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    M = np.zeros((X.shape[1],) * 2)
    for a, z in zip(e[:-1], e[1:]):
        s = Xs[a:z].T @ us[a:z]
        M += np.outer(s, s)
    return b, XtX @ M @ XtX


def logit(X, y, iters=60):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.maximum(p * (1 - p), 1e-9)
        step = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-9:
            break
    p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
    return b, np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def main() -> int:
    inpatient = "--inpatient" in sys.argv
    cap_ab, cap_c = "--cap-ab" in sys.argv, "--uncapped-c" not in sys.argv
    six = "--panels6" in sys.argv
    cost_col = "flat_cost_inpatient" if inpatient else "flat_cost_total"
    cost_lab = "in-patient cost, £ (uncapped)" if inpatient else "cost index, £ (uncapped)"
    apply_style()
    hm = pd.read_parquet(MEAS / "health_measure.parquet", columns=["pidp", "wave", "age", "h", "theta"])
    hm = hm[hm["age"].between(20, 90)]
    c = hm.merge(pd.read_parquet(MEAS / "cost_index.parquet", columns=["pidp", "wave", cost_col]),
                 on=["pidp", "wave"]).dropna(subset=[cost_col]).rename(columns={cost_col: "cost"}).reset_index(drop=True)
    panel = pd.read_csv(PROCESSED_DATA_DIR / "ukhls_indresp_processed.csv", usecols=["pidp", "wave", "dcsedw_dv"])
    death_wave = panel.groupby("pidp")["dcsedw_dv"].max() - 18          # UKHLS wave of death
    m = hm[hm["wave"] <= 14].merge(death_wave.rename("dw"), left_on="pidp", right_index=True, how="left")
    m = m[~(m["dw"] <= m["wave"])].assign(died=lambda x: (x["dw"] == x["wave"] + 1).astype(float)).reset_index(drop=True)
    cap = c["cost"].quantile(CAP_Q)
    c["cost_c"] = c["cost"].clip(upper=cap) if cap_c else c["cost"]
    if cap_ab:
        c["cost"] = c["cost"].clip(upper=cap)
    print(f"cost: {len(c):,} person-waves ({cost_col}), p99 £{cap:,.0f}; mortality: {len(m):,} at risk, {int(m['died'].sum()):,} deaths")

    if six:
        fig, ax = plt.subplots(2, 3, figsize=(14.5, 8.4), gridspec_kw={"hspace": 0.45, "wspace": 0.32})
    else:
        fig, ax4 = plt.subplots(2, 2, figsize=(10.2, 8.4), gridspec_kw={"hspace": 0.45, "wspace": 0.28})
        hidden, hax = plt.subplots(2, 1)          # the by-band panels are still computed, into a throwaway figure
        ax = np.array([[ax4[0, 0], ax4[0, 1], hax[0]], [ax4[1, 0], ax4[1, 1], hax[1]]])
    rows = []
    for j, v in enumerate(["h", "theta"]):
        mu, sd = m[v].mean(), m[v].std()
        for (lo, hi, name), col in zip(BANDS3, BAND_COL):
            s = c[c["age"].between(lo, hi)].assign(bin=lambda x: pd.cut(x[v], EDGES_COST[v]))
            g = s.groupby("bin", observed=True).agg(x=(v, "mean"), mean=("cost", "mean"), se=("cost", "sem"), n=("cost", "size"))
            g = g[g["n"] >= 150]
            ax[0, j].errorbar(g["x"], g["mean"], yerr=1.96 * g["se"], fmt="o-", ms=3.5, lw=1.1, color=col,
                              elinewidth=0.7, capsize=0, label=f"ages {name}")
            rows += [{"panel": "cost", "variant": v, "band": name, **r} for r in g.reset_index(drop=True).to_dict("records")]
            s = m[m["age"].between(lo, hi)].assign(bin=lambda x: pd.cut(x[v], EDGES_MORT[v]))
            g = s.groupby("bin", observed=True).agg(x=(v, "mean"), k=("died", "sum"), n=("died", "size"))
            g = g[g["k"] >= 10]
            lo_ci, hi_ci = wilson(g["k"], g["n"])
            rate = g["k"] / g["n"]
            ax[1, j].errorbar(g["x"], rate, yerr=[rate - lo_ci, hi_ci - rate], fmt="o", ms=4, color=col,
                              elinewidth=0.8, capsize=0, label=f"ages {name}")
            z = ((s[v] - mu) / sd).to_numpy()
            a = (s["age"].to_numpy() - 55) / 10
            b, _ = logit(np.column_stack([np.ones(len(s)), z, a, a ** 2]), s["died"].to_numpy(float))
            if len(g):
                xs = np.linspace(g["x"].min(), g["x"].max(), 60)
                am = (s["age"].mean() - 55) / 10
                ax[1, j].plot(xs, 1 / (1 + np.exp(-(b[0] + b[1] * (xs - mu) / sd + b[2] * am + b[3] * am ** 2))), color=col, lw=1.4)
            rows += [{"panel": "mortality", "variant": v, "band": name, "x": x, "rate": r, "deaths": k, "n": n}
                     for x, r, k, n in zip(g["x"], rate, g["k"], g["n"])]
        ax[0, j].set_title(f"({'ab'[j]}) annual cost against {LAB[v]}, by age band", loc="left")
        ax[0, j].set_ylabel(f"mean {cost_lab}".replace("uncapped", "capped at p99" if cap_ab else "uncapped"))
        ax[0, j].set_ylim(bottom=0)
        ax[0, j].set_xlabel(f"{LAB[v]}, equal-width bins (right = healthier)")
        ax[0, j].legend(loc="upper right")
        ax[1, j].set_yscale("log")
        ax[1, j].set_title(f"({('de' if six else 'cd')[j]}) one-year mortality against {LAB[v]}, by age band", loc="left")
        ax[1, j].set_ylabel("death before next wave (log scale)")
        ax[1, j].set_xlabel(f"{LAB[v]}, equal-width bins (right = healthier)")
        ax[1, j].legend(loc="upper right")

    x6 = np.arange(len(BANDS6))
    lab6 = [f"{a}-{b}" for a, b in BANDS6]
    tests = {}
    for k, (v, col) in enumerate([("h", BLUE), ("theta", VERM)]):
        zc = -((c[v] - c[v].mean()) / c[v].std()).to_numpy()          # higher = sicker
        zm = ((m[v] - m[v].mean()) / m[v].std()).to_numpy()
        yc = c["cost_c"].to_numpy(float)
        curv, curv_se, slope, slope_se, mort, mse = [], [], [], [], [], []
        for lo, hi in BANDS6:
            s_ = ((c["age"] >= lo) & (c["age"] <= hi)).to_numpy()
            b, V = ols_cluster(np.column_stack([np.ones(s_.sum()), zc[s_], zc[s_] ** 2]), yc[s_], c["pidp"].to_numpy()[s_])
            slope.append(b[1]); slope_se.append(np.sqrt(V[1, 1])); curv.append(b[2]); curv_se.append(np.sqrt(V[2, 2]))
            s_ = ((m["age"] >= lo) & (m["age"] <= hi)).to_numpy()
            a = (m["age"].to_numpy()[s_] - 55) / 10
            b, V = logit(np.column_stack([np.ones(s_.sum()), zm[s_], a, a ** 2]), m["died"].to_numpy(float)[s_])
            mort.append(-b[1]); mse.append(np.sqrt(V[1, 1]))
        off = (k - 0.5) * 0.22
        ax[0, 2].errorbar(x6 + off, curv, yerr=1.96 * np.array(curv_se), fmt="o", color=col, capsize=2, ms=5, label=f"{LAB[v]}: curvature")
        ax[1, 2].errorbar(x6 + off, mort, yerr=1.96 * np.array(mse), fmt="o", color=col, capsize=2, ms=5, label=LAB[v])
        dec = (c["age"].to_numpy() - 50) / 10
        g = c["pidp"].to_numpy()
        b, V = ols_cluster(np.column_stack([np.ones(len(c)), zc, zc ** 2, dec, dec ** 2, zc * dec, zc ** 2 * dec]), yc, g)
        b0, V0 = ols_cluster(np.column_stack([np.ones(len(c)), zc, zc ** 2]), yc, g)
        a = (m["age"].to_numpy() - 55) / 10
        bm, Vm = logit(np.column_stack([np.ones(len(m)), zm, a, a ** 2, zm * a]), m["died"].to_numpy(float))
        old = (m["age"] >= 45).to_numpy()          # the same test from 45, where the deaths are
        bo, Vo = logit(np.column_stack([np.ones(old.sum()), zm[old], a[old], a[old] ** 2, zm[old] * a[old]]), m["died"].to_numpy(float)[old])
        tests[v] = {"slope": b0[1], "curvature": b0[2], "curvature_t": b0[2] / np.sqrt(V0[2, 2]),
                    "slope_x_age": b[5], "slope_x_age_t": b[5] / np.sqrt(V[5, 5]),
                    "curvature_x_age": b[6], "curvature_x_age_t": b[6] / np.sqrt(V[6, 6]),
                    "mort_slope": -bm[1], "mort_x_age": -bm[4], "mort_x_age_t": -bm[4] / np.sqrt(Vm[4, 4]),
                    "mort_x_age_45plus": -bo[4], "mort_x_age_45plus_t": -bo[4] / np.sqrt(Vo[4, 4])}
        for i in range(len(BANDS6)):
            rows.append({"panel": "bands", "variant": v, "band": lab6[i], "slope": slope[i], "slope_se": slope_se[i],
                         "curvature": curv[i], "curvature_se": curv_se[i], "mort_slope": mort[i], "mort_slope_se": mse[i]})
        print(v, {k2: round(val, 2) for k2, val in tests[v].items()})
        print("   curvature by band", np.round(curv), "slope", np.round(slope), "mortality", np.round(mort, 2))
    for a, ttl, yl, lg in [(ax[0, 2], "(c) convexity by age band" + (", costs capped at p99" if cap_c else ""), "quadratic term, £ per SD$^2$ sicker", "upper left"),
                           (ax[1, 2], "(f) mortality gradient by age band", "log-odds of death per SD sicker\n(pooled SD; quadratic in age within band)", "upper right")]:
        a.set_xticks(x6); a.set_xticklabels(lab6, fontsize=7.5); a.set_xlabel("age band")
        a.set_title(ttl, loc="left"); a.set_ylabel(yl)
        a.legend(loc=lg, fontsize=7.2)
    ax[0, 2].axhline(0, color=INK2, lw=0.6)
    ax[0, 2].set_ylim(bottom=min(0, ax[0, 2].get_ylim()[0]))
    ax[1, 2].set_ylim(0, max(1.2, ax[1, 2].get_ylim()[1]))
    for a in ax.flat:
        a.grid(True, axis="y")
    th, tt = tests["h"], tests["theta"]
    what = "in-patient tier of the cost index" if inpatient else "the cost index (in-patient, out-patient and GP)"
    fig.text(0.01, -0.005,
             f"UKHLS, the paper's health measure, ages 20-90. Cost: {what}, {'capped at the 99th percentile' if cap_ab else 'uncapped'} in (a)-(b), waves 7-15, {len(c):,} person-waves; bins with at least 150 "
             f"person-waves. Mortality: death before the next\nwave (one year), waves 1-14, {len(m):,} person-waves at risk, {int(m['died'].sum()):,} deaths; "
             "bins with at least 10 deaths, lines the within-band logit at the band's mean age. Measures standardised on the pooled sample. Panel (c): "
             f"the quadratic term of cost{' capped at the 99th percentile' if cap_c else ''} on the\nstandardised measure within each band, person-clustered 95% intervals. "
             f"Pooled: curvature t = {th['curvature_t']:.1f} (h), {tt['curvature_t']:.1f} (theta); its change per decade of age t = {th['curvature_x_age_t']:.1f}, {tt['curvature_x_age_t']:.1f}; "
             f"the slope's change t = {th['slope_x_age_t']:.1f}, {tt['slope_x_age_t']:.1f};\nmortality slope's change t = {th['mort_x_age_t']:.1f} (h), {tt['mort_x_age_t']:.1f} (theta), negative = flattens with age.",
             fontsize=7.1, color=INK2, va="top")
    if not six:
        for t in fig.texts:
            t.remove()
        fig.text(0.01, -0.005,
                 f"UKHLS, the paper's health measure, ages 20-90. Cost: {what}, {'capped at the 99th percentile' if cap_ab else 'uncapped'}, waves 7-15, {len(c):,} person-waves; "
                 f"bins with at least 150 person-waves,\n95% intervals. Mortality: death before the next wave (one year), waves 1-14, {len(m):,} person-waves at risk, "
                 f"{int(m['died'].sum()):,} deaths; bins with at least 10 deaths, Wilson 95% intervals; lines are the within-band logit,\nlinear in the measure and quadratic in age, "
                 "drawn at the band's mean age. Measures standardised on the pooled sample, so a bin means the same health at every age.",
                 fontsize=7.1, color=INK2, va="top")
        plt.close(hidden)
    tag = ("_inpatient" if inpatient else "") + ("_capab" if cap_ab else "") + ("_uncappedc" if not cap_c else "") + ("_panels6" if six else "")
    out = (ROOT_DIR / "paper" / "figures" / "fig_cost_mortality_age.png") if not tag else (ARTIFACTS_DIR / "scratch" / f"fig_cost_mortality_age{tag}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(DESC / f"cost_mortality_age{tag}.csv", index=False)
    pd.DataFrame(tests).T.to_csv(DESC / f"cost_mortality_age{tag}_tests.csv")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
