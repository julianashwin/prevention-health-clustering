"""Figure 1 on PSID: cost and mortality against theta, by age band.

The main-mode figure of descriptives/37_paper_cost_mortality_age.py (bins, bands, cell rules,
logits and the pooled cubic copied from it), on PSID data:
  (a) mean annual cost in equal-width bins of theta within three age bands (bins with at least
      150 person-waves, 95% intervals), with the pooled cubic cost curve that the prevention
      figure prices with (31_paper_appendix_extras.fit_cost_curve, uncapped). Grey open circles:
      the UKHLS in-patient tier of the cost index in the same theta bins, all ages pooled, read
      from the aggregate csv artifacts/descriptives/cost_mortality_age_inpatient.csv.
  (b) death before the next wave against theta, log scale, bands 45-64 and 65-90 (bins with at
      least 10 deaths, Wilson intervals), with the within-band logit (linear in theta, quadratic
      in age) at the band's mean age.
PSID differences: the cost index is in-patient only (hospital nights in the calendar year
before the interview, priced at the UKHLS stage 1 NHS rates); death is within two years
(mort_event), not one; waves are biennial.

Outputs: descriptives/psid/figures/main/fig_cost_mortality_age.png,
         artifacts/descriptives/psid/cost_mortality_age.csv and _tests.csv
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _psid_common import DESC, FIG, ROOT, cost_rows, mortality_rows, scores  # noqa: E402
from _style import INK3, apply_style  # noqa: E402

M37 = importlib.import_module("37_paper_cost_mortality_age")        # ols_cluster, logit, wilson, bins
M31 = importlib.import_module("31_paper_appendix_extras")          # fit_cost_curve
BANDS3, BANDS6, BAND_COL = M37.BANDS3, M37.BANDS6, M37.BAND_COL
EDGES_COST, EDGES_MORT = M37.EDGES_COST["theta"], M37.EDGES_MORT["theta"]
CAP_Q = 0.99


def data():
    s = scores()
    c = s.merge(cost_rows(), on=["pidp", "wave"]).dropna(subset=["cost", "theta"]).reset_index(drop=True)
    m = s.merge(mortality_rows(), on=["pidp", "wave"]).dropna(subset=["died", "theta"]).reset_index(drop=True)
    return c, m


def main() -> int:
    apply_style()
    v = "theta"
    c, m = data()
    cap = c["cost"].quantile(CAP_Q)
    c["cost_c"] = c["cost"].clip(upper=cap)
    print(f"cost: {len(c):,} person-waves, mean £{c['cost'].mean():,.0f}, any stay {(c['cost'] > 0).mean():.1%}, p99 £{cap:,.0f}; "
          f"mortality: {len(m):,} at risk, {int(m['died'].sum()):,} deaths")
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.3), gridspec_kw={"wspace": 0.28})
    rows = []
    mu, sd = m[v].mean(), m[v].std()
    for (lo, hi, name), col in zip(BANDS3, BAND_COL):
        s = c[c["age"].between(lo, hi)].assign(bin=lambda x: pd.cut(x[v], EDGES_COST))
        g = s.groupby("bin", observed=True).agg(x=(v, "mean"), mean=("cost", "mean"), se=("cost", "sem"), n=("cost", "size"))
        g = g[g["n"] >= 150]
        ax[0].errorbar(g["x"], g["mean"], yerr=1.96 * g["se"], fmt="o-", ms=3.5, lw=1.1, color=col, elinewidth=0.7, capsize=0, label=f"ages {name}")
        rows += [{"panel": "cost", "band": name, **r} for r in g.reset_index(drop=True).to_dict("records")]
        if lo < 45:
            continue
        s = m[m["age"].between(lo, hi)].assign(bin=lambda x: pd.cut(x[v], EDGES_MORT))
        g = s.groupby("bin", observed=True).agg(x=(v, "mean"), k=("died", "sum"), n=("died", "size"))
        g = g[g["k"] >= 10]
        lo_ci, hi_ci = M37.wilson(g["k"], g["n"])
        rate = g["k"] / g["n"]
        ax[1].errorbar(g["x"], rate, yerr=[rate - lo_ci, hi_ci - rate], fmt="o", ms=4, color=col, elinewidth=0.8, capsize=0, label=f"ages {name}")
        z = ((s[v] - mu) / sd).to_numpy(); a = (s["age"].to_numpy() - 55) / 10
        b, _ = M37.logit(np.column_stack([np.ones(len(s)), z, a, a ** 2]), s["died"].to_numpy(float))
        xs = np.linspace(g["x"].min(), g["x"].max(), 60); am = (s["age"].mean() - 55) / 10
        ax[1].plot(xs, 1 / (1 + np.exp(-(b[0] + b[1] * (xs - mu) / sd + b[2] * am + b[3] * am ** 2))), color=col, lw=1.4)
        rows += [{"panel": "mortality", "band": name, "x": x, "rate": r, "deaths": k, "n": n_} for x, r, k, n_ in zip(g["x"], rate, g["k"], g["n"])]
    # the UKHLS in-patient tier in the same bins, all ages pooled (n-weighted over the three bands)
    uk = pd.read_csv(ROOT / "artifacts" / "descriptives" / "cost_mortality_age_inpatient.csv")
    uk = uk[(uk["panel"] == "cost") & (uk["variant"] == "theta")].copy()
    uk["bin"] = pd.cut(uk["x"], EDGES_COST)
    ukp = uk.groupby("bin", observed=True).apply(lambda q: pd.Series({"x": np.average(q["x"], weights=q["n"]),
                                                                     "mean": np.average(q["mean"], weights=q["n"])}), include_groups=False)
    ax[0].plot(ukp["x"], ukp["mean"], "o", mfc="none", mec=INK3, ms=4.5, label="UKHLS in-patient tier, all ages")
    coef, _, deg = M31.fit_cost_curve(c[v].to_numpy(float), c["cost"].to_numpy(float))
    xs = np.linspace(max(EDGES_COST[0], c[v].min()), min(EDGES_COST[-1], c[v].max()), 200)
    ax[0].plot(xs, np.polynomial.polynomial.polyval(xs, coef), color=M37.VERM, lw=1.8, zorder=5,
               label=f"pooled {'cubic' if deg == 3 else 'quadratic'} (prevention figure)")
    ax[0].set_title(r"(a) annual cost against $\theta$, by age band", loc="left")
    ax[0].set_ylabel("mean in-patient cost index, £ (uncapped)"); ax[0].set_ylim(bottom=0)
    ax[0].set_xlabel(r"$\theta$, equal-width bins (right = healthier)"); ax[0].legend(loc="upper right", fontsize=7.5)
    ax[1].set_yscale("log")
    ax[1].set_title(r"(b) two-year mortality against $\theta$, by age band", loc="left")
    ax[1].set_ylabel("death before next wave (log scale)")
    ax[1].set_xlabel(r"$\theta$, equal-width bins (right = healthier)"); ax[1].legend(loc="upper right")
    for a in ax:
        a.grid(True, axis="y")
    # pooled tests, as the UKHLS script (cost capped at p99 in the curvature regressions)
    zc = -((c[v] - c[v].mean()) / c[v].std()).to_numpy(); yc = c["cost_c"].to_numpy(float); g = c["pidp"].to_numpy()
    dec = (c["age"].to_numpy() - 50) / 10
    b0, V0 = M37.ols_cluster(np.column_stack([np.ones(len(c)), zc, zc ** 2]), yc, g)
    b, V = M37.ols_cluster(np.column_stack([np.ones(len(c)), zc, zc ** 2, dec, dec ** 2, zc * dec, zc ** 2 * dec]), yc, g)
    zm = ((m[v] - mu) / sd).to_numpy(); a = (m["age"].to_numpy() - 55) / 10
    bm, Vm = M37.logit(np.column_stack([np.ones(len(m)), zm, a, a ** 2, zm * a]), m["died"].to_numpy(float))
    tests = {"slope": b0[1], "curvature": b0[2], "curvature_t": b0[2] / np.sqrt(V0[2, 2]),
             "slope_x_age_t": b[5] / np.sqrt(V[5, 5]), "curvature_x_age_t": b[6] / np.sqrt(V[6, 6]),
             "mort_slope": -bm[1], "mort_x_age": -bm[4], "mort_x_age_t": -bm[4] / np.sqrt(Vm[4, 4]),
             "cubic_c0": coef[0], "cubic_c1": coef[1], "cubic_c2": coef[2], "cubic_c3": coef[3] if deg == 3 else 0.0,
             "cost_mean": c["cost"].mean(), "any_stay": (c["cost"] > 0).mean(), "p99": cap, "n_cost": len(c),
             "n_mort": len(m), "deaths": int(m["died"].sum())}
    print({k: round(float(x), 3) for k, x in tests.items()})
    fig.text(0.01, -0.02,
             f"PSID, theta (five-testlet GRM), ages 20-90. Cost: in-patient stays only (nights in hospital in the calendar year before the interview, births excluded), "
             f"priced at the UKHLS index's NHS rates, uncapped,\nwaves 2003-2023, {len(c):,} person-waves; bins with at least 150 person-waves, 95% intervals. "
             f"Mortality: death within two years of the interview, waves 2003-2021, {len(m):,} person-waves at risk, {int(m['died'].sum()):,} deaths;\n"
             "bins with at least 10 deaths, Wilson 95% intervals; lines are the within-band logit, linear in theta and quadratic in age, at the band's mean age. "
             "Grey circles: the UKHLS in-patient tier in the same bins.",
             fontsize=7.1, color=M37.INK2, va="top")
    fig.savefig(FIG / "fig_cost_mortality_age.png", bbox_inches="tight")
    pd.DataFrame(rows).to_csv(DESC / "cost_mortality_age.csv", index=False)
    pd.DataFrame([tests]).to_csv(DESC / "cost_mortality_age_tests.csv", index=False)
    pd.DataFrame({"coef": coef, "power": np.arange(len(coef))}).to_csv(DESC / "psid_cost_curve.csv", index=False)
    print("wrote fig_cost_mortality_age.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
