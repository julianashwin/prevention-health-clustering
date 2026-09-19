"""Appendix B: Exhibit 1 for healthcare cost, and the returns under the estimated cost curve.

Part A (fig_exhibit1_cost.png). The cost index (waves 7-15, one row per
person-wave that answered the utilisation block) through Exhibit 1's three
panels: mean, variance and covariance rows by age, in pounds capped at the
99th percentile and in log(1 + pounds). Rows run eight years, the most the
nine utilisation waves allow.

Part B (fig_returns_cost.png, tab_returns_cost.tex). The framework's returns
with c(.) the ESTIMATED cost curve rather than (1 - h)^2: a cubic in each
variant fitted to the capped cost index (a quadratic if the cubic is not
monotone on the support). With a general c, treatment h -> h + tau (top - h)
and prevention h -> h + mu (hbar_{j,r-1} - hbar_{j,a}) give

  T(r)   = sum_{a>=r} E[ -c'(h_a) (top - h_a) ]
  P_j(r) = sum_{a>=r} E[ -c'(h_a) | j ] (hbar_{j,r-1} - hbar_{j,a})

which reduce to 28_paper_returns.py's formulas when c = (top - h)^2; that
case is recomputed here as a check. Returns are then in pounds of lifetime
cost per unit of intensity. Person-level expectations are age means over the
clustered people (all waves), smoothed like the type paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import (  # noqa: E402
    AGES, DESC, FIG, K, LABEL, MAX_AGE, MIN_AGE, TAB, VARIANTS, draw_exhibit1_row, load_labels, load_measure,
    load_trajectories, profile_rows, smooth_path, write_table,
)
from _style import BLUE, CLUSTER, INK2, INK3, ORANGE, VERM, apply_style  # noqa: E402

from prevention_health_clustering.config import PROCESSED_DATA_DIR  # noqa: E402

R_GRID = np.arange(25, 86)
R_TABLE = [30, 50, 70]
COST = "flat_cost_total"
AGE_BANDS = [(20, 34), (35, 44), (45, 54), (55, 64), (65, 74), (75, 90)]


def fit_cost_curve(x: np.ndarray, y: np.ndarray):
    """Polynomial E[cost | x]; cubic if monotone decreasing on the 1st-99th percentile range, else quadratic."""
    lo, hi = np.quantile(x, [0.01, 0.99])
    grid = np.linspace(lo, hi, 200)
    for deg in (3, 2):
        coef = np.polynomial.polynomial.polyfit(x, y, deg)
        dcoef = np.polynomial.polynomial.polyder(coef)
        if (np.polynomial.polynomial.polyval(grid, dcoef) <= 0).all():
            return coef, dcoef, deg
    return coef, dcoef, deg


def returns(s: pd.DataFrame, v: str, traj: pd.DataFrame, pi: np.ndarray, cprime, top: float):
    """T(r), P_j(r), P(r) and its mean channel for a general cost derivative cprime(x)."""
    cp = -cprime(s[v].to_numpy(float))                       # -c'(h) >= 0
    s = s.assign(_cp=cp, _t=cp * (top - s[v].to_numpy(float)))
    prof = s.groupby("age").agg(t=("_t", "mean"), g=("_cp", "mean"), m=(v, "mean"), n=(v, "size"))
    t_a = smooth_path(prof.index.to_numpy(), prof["t"].to_numpy(), prof["n"].to_numpy(), min_count=100).to_numpy()
    g_a = smooth_path(prof.index.to_numpy(), prof["g"].to_numpy(), prof["n"].to_numpy(), min_count=100).to_numpy()
    hbar = smooth_path(prof.index.to_numpy(), prof["m"].to_numpy(), prof["n"].to_numpy(), min_count=100).to_numpy()
    gj = s.groupby(["cluster", "age"]).agg(g=("_cp", "mean"), n=("_cp", "size")).reset_index()
    G = np.vstack([smooth_path(q["age"].to_numpy(), q["g"].to_numpy(), q["n"].to_numpy(), min_count=25).to_numpy()
                   for c in range(K) for q in [gj[gj["cluster"] == c]]])
    paths = np.vstack([smooth_path(q["age"].to_numpy(), q["mean"].to_numpy(), q["count"].to_numpy(), min_count=25).to_numpy()
                       for c in range(K) for q in [traj[traj["cluster"] == c]]])
    recs = []
    for r in R_GRID:
        idx = slice(r - MIN_AGE, MAX_AGE - MIN_AGE + 1)
        T = float(t_a[idx].sum())
        prevd = paths[:, r - 1 - MIN_AGE][:, None] - paths[:, idx]
        Pj = (G[:, idx] * prevd).sum(axis=1)
        P = float(pi @ Pj)
        mean_ch = float((g_a[idx] * (hbar[r - 1 - MIN_AGE] - hbar[idx])).sum())
        recs.append({"r": r, "T": T, "P": P, "P_over_T": P / T, "mean_channel": mean_ch, "cov_channel": P - mean_ch,
                     "P_max_type": Pj.max(), "type_gain": Pj.max() - P, **{f"P_{c + 1}": Pj[c] for c in range(K)}})
    return pd.DataFrame(recs).set_index("r")


def main() -> int:
    apply_style()
    # ---- Part A: Exhibit 1 for cost -------------------------------------------
    cost = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "cost_index.parquet", columns=["pidp", "wave", "age", COST])
    cost = cost[cost["age"].between(MIN_AGE, MAX_AGE)].assign(age=lambda x: x["age"].astype(int))
    cost = cost.sort_values(["pidp", "age", "wave"]).drop_duplicates(["pidp", "age"])
    cap = cost[COST].quantile(0.99)
    cost["cost_capped"] = cost[COST].clip(upper=cap)
    cost["log_cost"] = np.log1p(cost[COST])
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for i, (v, lab) in enumerate([("cost_capped", f"cost index, £ capped at £{cap:,.0f}"), ("log_cost", "log(1 + cost index)")]):
        smooth, rows = profile_rows(cost, v, row_len=8)
        draw_exhibit1_row(axes[i], smooth, rows, lab, "abc" if i == 0 else "def", legend=(i == 0))
        pd.concat([smooth.reset_index().assign(kind="profile"), rows.assign(kind="row")]).assign(variant=v).to_csv(
            DESC / f"paper_exhibit1_cost_{v}.csv", index=False)
    for ax in axes[1]:
        ax.set_xlabel("age")
    fig.text(0.01, -0.01, f"{len(cost):,} person-ages on {cost['pidp'].nunique():,} people, waves 7-15, ages 20-90. "
             "Five-year centred rolling means; cells with fewer than 100 person-ages dropped; rows run eight years.",
             fontsize=7.4, color=INK2, va="top")
    fig.tight_layout(); fig.savefig(FIG / "fig_exhibit1_cost.png"); plt.close(fig)

    # ---- Part B: returns under the estimated cost curve ---------------------------
    d = load_measure()
    cs = d.merge(cost[["pidp", "wave", "cost_capped"]], on=["pidp", "wave"])
    fig, axes = plt.subplots(5, 3, figsize=(12.5, 15), gridspec_kw={"hspace": 0.45, "wspace": 0.28})
    table, recs, check, inv_rows = [], [], [], []
    ref = pd.read_csv(DESC / "paper_returns.csv")
    for j, (v, lab) in enumerate(VARIANTS):
        L = load_labels(v)
        s = d[d["pidp"].isin(L.index)].assign(cluster=lambda x: x["pidp"].map(L))
        traj = load_trajectories(v)
        pi = L.value_counts(normalize=True).sort_index().reindex(range(K)).to_numpy()
        top = 1.0 if v != "theta" else float(s[v].max())
        x, y = cs[v].to_numpy(float), cs["cost_capped"].to_numpy(float)
        coef, dcoef, deg = fit_cost_curve(x, y)
        cprime = lambda z, dc=dcoef: np.polynomial.polynomial.polyval(z, dc)   # noqa: E731
        R = returns(s, v, traj, pi, cprime, top)
        R.assign(variant=v, degree=deg).reset_index().pipe(recs.append)
        # check: the quadratic gap reproduces 28_paper_returns.py
        Rq = returns(s, v, traj, pi, lambda z, t=top: -2 * (t - z), top)
        rq = ref[ref["variant"] == v].set_index("r")
        scale = 1.0 if v != "theta" else (s[v].max() - s[v].min()) ** 2   # 28 rescales theta to [0, 1]
        check.append({"variant": v, "max_rel_diff_T": float((Rq["T"] / (rq["T"] * scale) - 1).abs().max()),
                      "max_rel_diff_P": float((Rq["P"] / (rq["P"] * scale) - 1).abs().max())})
        # row 0: is the estimated curve age-invariant? the same polynomial within age bands
        ax = axes[0, j]
        cmap = plt.get_cmap("viridis")
        for bi, (blo, bhi) in enumerate(AGE_BANDS):
            sb = cs[cs["age"].between(blo, bhi)]
            xb, yb = sb[v].to_numpy(float), sb["cost_capped"].to_numpy(float)
            cb = np.polynomial.polynomial.polyfit(xb, yb, deg)
            gb = np.linspace(np.quantile(xb, 0.01), np.quantile(xb, 0.99), 80)
            ax.plot(gb, np.polynomial.polynomial.polyval(gb, cb), color=cmap(0.9 * bi / (len(AGE_BANDS) - 1)),
                    lw=1.4, label=f"{blo}-{bhi}")
            for xq in np.quantile(cs[v], [0.05, 0.25, 0.5, 0.75]):
                inv_rows.append({"variant": v, "band": f"{blo}-{bhi}", "x_quantile_pooled": xq,
                                 "cost_at": float(np.polynomial.polynomial.polyval(xq, cb))})
        ax.plot(grid := np.linspace(np.quantile(x, 0.005), x.max(), 100),
                np.polynomial.polynomial.polyval(grid, coef), color="black", lw=1.6, ls="--", label="pooled")
        ax.set_title(f"{lab}: the curve within age bands", loc="left", fontsize=9.5)
        ax.set_ylabel("£ per year"); ax.legend(fontsize=6.8, ncols=2, loc="upper right"); ax.grid(True, axis="y")
        # row 1: the pooled fitted curve against binned means
        ax = axes[1, j]
        q = pd.qcut(cs[v], 20, labels=False, duplicates="drop")
        bm = cs.groupby(q).agg(x=(v, "mean"), y=("cost_capped", "mean"))
        grid = np.linspace(np.quantile(x, 0.005), x.max(), 100)
        ax.plot(grid, np.polynomial.polynomial.polyval(grid, coef), color=ORANGE, lw=1.8, label=f"fitted, degree {deg}")
        ax.scatter(bm["x"], bm["y"], color=BLUE, s=14, zorder=3, label="twentieths")
        ax.set_title(f"{lab}: estimated cost curve $c(\\cdot)$", loc="left", fontsize=9.5)
        ax.set_ylabel("£ per year"); ax.legend(loc="upper right", fontsize=7.5); ax.grid(True, axis="y")
        ax = axes[2, j]
        for c in range(K):
            ax.plot(R.index, R[f"P_{c + 1}"], color=CLUSTER[c], lw=1.8, label=f"type {c + 1} ({pi[c]:.0%})")
        ax.plot(R.index, R["P"], color=INK2, lw=1.4, ls="--", label="uniform $P(r)$")
        ax.axhline(0, color=INK2, lw=0.6); ax.set_title("$P_j(r)$, £ of lifetime cost", loc="left", fontsize=9.5)
        ax.legend(loc="upper right", fontsize=7.5); ax.grid(True, axis="y")
        ax = axes[3, j]
        ax.plot(R.index, R["T"], color=VERM, lw=1.8, label="treatment $T(r)$")
        ax.plot(R.index, R["P"], color=BLUE, lw=1.8, label="uniform prevention $P(r)$")
        ax.plot(R.index, R["mean_channel"], color=BLUE, lw=1.2, ls=":", label="  of which mean channel")
        ax.set_yscale("log"); ax.set_title("returns by start age (log scale, £)", loc="left", fontsize=9.5)
        ax.legend(loc="lower left", fontsize=7.5); ax.grid(True, axis="y")
        ax = axes[4, j]
        ax.plot(R.index, R["P_over_T"], color=ORANGE, lw=1.8, label="$P(r)/T(r)$")
        ax.plot(R.index, R["type_gain"] / R["P"], color=CLUSTER[0], lw=1.8, label=r"$(\max_j P_j - P)/P$")
        ax.set_title("ratios", loc="left", fontsize=9.5); ax.set_xlabel("start age $r$")
        ax.legend(loc="upper right", fontsize=7.5); ax.grid(True, axis="y")
        for r in R_TABLE:
            xx = R.loc[r]
            table.append([lab if r == R_TABLE[0] else "", str(r), f"{xx['T']:,.0f}", f"{xx['P']:,.0f}",
                          f"{xx['P_over_T']:.3f}", f"{xx['cov_channel'] / xx['P']:.2f}", f"{xx['P_1']:,.0f}",
                          f"{xx['P_2']:,.0f}", f"{xx['P_3']:,.0f}", f"{xx['type_gain'] / xx['P']:.2f}"])
    fig.text(0.01, -0.005, "Top row: the same polynomial refitted within six age bands over each band's 1st-99th "
             "percentile range, against the pooled fit (dashed).\nBelow: as the main-text returns figure, with c(.) the "
             f"polynomial fitted to the capped cost index (£{cap:,.0f} cap) on each variant; treatment moves a person a\n"
             "share tau of the way to full health (1 for h and the deficit index, the sample maximum for theta), "
             "prevention a share mu of the type's preventable\ndecline. Expectations of c'(.) are age means over the "
             "clustered people, smoothed as the type paths.", fontsize=7.4, color=INK2, va="top")
    fig.savefig(FIG / "fig_returns_cost.png"); plt.close(fig)
    write_table(TAB / "tab_returns_cost.tex",
                ["variant", "$r$", "$T(r)$", "$P(r)$", "$P/T$", "cov.\\ channel share", "$P_1$", "$P_2$", "$P_3$",
                 "type gain $/P$"], table, colspec="llrrrrrrrr")
    out = pd.concat(recs)
    out.to_csv(DESC / "paper_returns_cost.csv", index=False)
    inv = pd.DataFrame(inv_rows)
    inv.to_csv(DESC / "paper_cost_curve_by_band.csv", index=False)
    piv = inv.pivot_table(index=["variant", "x_quantile_pooled"], columns="band", values="cost_at")
    print("estimated cost at pooled quantiles of each variant, by age band (£):\n", piv.round(0).to_string())
    print("quadratic-gap check against 28_paper_returns.py:\n", pd.DataFrame(check).round(4).to_string(index=False))
    print(out[out["r"].isin(R_TABLE)].round(1).to_string(index=False))
    print(f"wrote fig_exhibit1_cost.png, fig_returns_cost.png, tab_returns_cost.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
