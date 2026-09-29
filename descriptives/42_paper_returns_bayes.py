"""Section 5: the returns to prevention and treatment on the Bayesian classes, priced in pounds.

The framework's returns (Section 2.2, general-cost form) on the K = 3 mixture classes:
  T(r)   = sum_{a>=r} E[ -c'(h_a) (h_max - h_a) ]                       treatment
  P_j(r) = sum_{a>=r} E[ -c'(h_a) | j ] (hbar_{j,r-1} - hbar_{j,a})    prevention, class j
  P(r)   = sum_j pi_j P_j(r) = mean channel + between-class channel
  class targeting: max_j P_j(r) - P(r)
with c(.) the cost curve estimated in pounds on the measure (a cubic in the score on the cost
index, made monotone; fit_cost_curve in 31_paper_appendix_extras.py), on the uncapped index by
default (--capped to cap it at the 99th percentile). The class shares pi_j and paths hbar_{j,a}
are the fitted mixture at its posterior mean (quadratics in age, 20-90); E[-c'(h) | j] at each age
is the posterior-weighted mean over the person-waves observed there, and the population terms are
the observed cross-section. h_max is the healthiest observed score. Horizon 90, no survival
weighting, no discounting. Returns are pounds of lifetime cost per unit of policy intensity.

Default: theta, the AR(1) plus "spike" classes (the paper's Section 5), with the independent-
residuals classes alongside for comparison; --h the same on h for the appendix; --kmeans the same
calculation on the partial K-means types (hard assignments, smoothed type means) for the appendix;
--capped the capped cost curve (robustness); --curve the appendix figure of the cost curve itself,
pooled and refitted within six age bands (fig_cost_curve.png).

Outputs: paper/figures/fig_returns_bayes{,_h}.png, paper/tables/tab_returns_bayes{,_h}.tex,
         artifacts/descriptives/paper_returns_bayes{,_h}{,_capped}_{ssm,base}.csv
"""

from __future__ import annotations

import importlib
import textwrap
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import AGES, DESC, FIG, K, MAX_AGE, MIN_AGE, TAB, bayes_fit, load_labels, load_measure, load_trajectories, smooth_path, write_table  # noqa: E402
from _style import BLUE, CLUSTER, INK2, ORANGE, VERM, apply_style  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR  # noqa: E402

M31 = importlib.import_module("31_paper_appendix_extras")
R_GRID = np.arange(25, 86)
R_TABLE = [30, 50, 70]
COST = "flat_cost_total"
V = "h" if "--h" in sys.argv else "theta"
UNCAPPED = "--capped" not in sys.argv          # the paper prices with the uncapped curve; --capped for the robustness check
SFX = ("" if V == "theta" else "_h") + ("" if UNCAPPED else "_capped")
VL = {"theta": r"$\theta$", "h": "$h$"}[V]
KMEANS = "--kmeans" in sys.argv                # the appendix version on the partial K-means types
SFX += "_kmeans" if KMEANS else ""
SETS = [("kmeans", "K-means types")] if KMEANS else [("ssm", 'AR(1) + "spike" classes'), ("base", "independent-residuals classes")]
AGE_BANDS = [(20, 34), (35, 44), (45, 54), (55, 64), (65, 74), (75, 90)]


def smooth(x: pd.Series, n: pd.Series, min_n: int = 100) -> np.ndarray:
    """An age profile on every age 20-90: five-year rolling mean where supported, ends held flat."""
    y = x.where(n >= min_n).reindex(AGES)
    y = y.rolling(5, center=True, min_periods=2).mean()
    return y.interpolate(limit_direction="both").to_numpy()


def returns(rows: pd.DataFrame, W: np.ndarray, paths: np.ndarray, pi: np.ndarray, cprime, top: float) -> pd.DataFrame:
    cp = -cprime(rows[V].to_numpy(float))                    # -c'(h) >= 0
    t = cp * (top - rows[V].to_numpy(float))
    age = rows["age"].to_numpy()
    g = pd.DataFrame({"age": age, "t": t, "cp": cp, "m": rows[V].to_numpy(float)}).groupby("age")
    n = g.size()
    t_a, g_a, hbar = smooth(g["t"].mean(), n), smooth(g["cp"].mean(), n), smooth(g["m"].mean(), n)
    G = []
    for k in range(K):
        wk = W[:, k]
        num = pd.Series(wk * cp).groupby(age).sum(); den = pd.Series(wk).groupby(age).sum()
        G.append(smooth(num / den, den, min_n=25))
    G = np.vstack(G)
    recs = []
    for r in R_GRID:
        idx = slice(r - MIN_AGE, MAX_AGE - MIN_AGE + 1)
        prevd = paths[:, r - 1 - MIN_AGE][:, None] - paths[:, idx]
        Pj = (G[:, idx] * prevd).sum(axis=1)
        P = float(pi @ Pj)
        mean_ch = float((g_a[idx] * (hbar[r - 1 - MIN_AGE] - hbar[idx])).sum())
        T = float(t_a[idx].sum())
        recs.append({"r": r, "T": T, "P": P, "P_over_T": P / T, "mean_channel": mean_ch, "cov_channel": P - mean_ch,
                     "cov_share": (P - mean_ch) / P, "P_max_class": Pj.max(), "best_class": int(Pj.argmax()) + 1,
                     "type_gain": (Pj.max() - P) / P, **{f"P_{c + 1}": Pj[c] for c in range(K)}})
    return pd.DataFrame(recs).set_index("r")


def curve_figure(cs: pd.DataFrame, cap: float) -> None:
    """The estimated cost curve on the measure: the pooled uncapped cubic, the same fit within six age
    bands (the age-invariance check the pricing rests on), and the capped fit, with twentieth-bin means."""
    x = cs[V].to_numpy(float); yraw = cs["raw"].to_numpy(float)
    lo, hi = np.quantile(x, [0.01, 0.99]); grid = np.linspace(lo, hi, 200)
    coef, _, deg = M31.fit_cost_curve(x, yraw)
    coef_c, _, deg_c = M31.fit_cost_curve(x, np.minimum(yraw, cap))
    pooled = np.polynomial.polynomial.polyval(grid, coef)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"wspace": 0.25})
    ax = axes[0]
    cmap = plt.get_cmap("viridis"); rows = []
    for bi, (blo, bhi) in enumerate(AGE_BANDS):
        sb = cs[cs["age"].between(blo, bhi)]
        cb = np.polynomial.polynomial.polyfit(sb[V].to_numpy(float), sb["raw"].to_numpy(float), deg)
        yb = np.polynomial.polynomial.polyval(grid, cb)
        ax.plot(grid, yb, color=cmap(0.9 * bi / (len(AGE_BANDS) - 1)), lw=1.3, label=f"{blo}-{bhi}")
        rows.append({"band": f"{blo}-{bhi}", "mean_ratio_to_pooled": float(np.mean(yb / pooled))})
    ax.plot(grid, pooled, color="black", lw=2.0, ls="--", label="pooled")
    ax.set_title(f"(a) cost curve on {VL}, refitted within age bands", loc="left", fontsize=9.5)
    ax.set_xlabel(f"{VL} (right = healthier)"); ax.set_ylabel("expected annual cost, £ (uncapped)"); ax.legend(fontsize=7.5, ncols=2); ax.grid(True, axis="y")
    ax = axes[1]
    b = pd.qcut(cs[V], 20, labels=False, duplicates="drop")
    g = cs.assign(b=b).groupby("b").agg(x=(V, "mean"), y=("raw", "mean"), yc=("raw", lambda z: np.minimum(z, cap).mean()))
    ax.plot(grid, pooled, color=VERM, lw=1.8, label=f"uncapped fit (degree {deg})")
    ax.plot(grid, np.polynomial.polynomial.polyval(grid, coef_c), color=BLUE, lw=1.8, label=f"capped at p99 (degree {deg_c})")
    ax.plot(g["x"], g["y"], "o", color=VERM, ms=3.5, mfc="none", label="twentieths, uncapped")
    ax.plot(g["x"], g["yc"], "o", color=BLUE, ms=3.5, mfc="none", label="twentieths, capped")
    ax.set_title("(b) the pooled fit against twentieths", loc="left", fontsize=9.5)
    ax.set_xlabel(f"{VL} (right = healthier)"); ax.legend(fontsize=7.5); ax.grid(True, axis="y")
    fig.text(0.01, -0.03, "\n".join(textwrap.wrap(
        f"The cost index in pounds, waves 7-15, {len(cs):,} person-waves, against {V}: a polynomial (cubic unless that is not monotone on the 1st-99th percentile range), "
        f"fitted pooled and within each age band, over the 1st-99th percentile range of {V}. Panel (b): the pooled fits on uncapped costs and on costs capped at the "
        f"99th percentile (£{cap:,.0f}), with the mean cost in twentieths of {V}.", 190)), fontsize=7.2, color=INK2, va="top")
    fig.savefig(FIG / f"fig_cost_curve{'' if V == 'theta' else '_h'}.png", bbox_inches="tight"); plt.close(fig)
    r = pd.DataFrame(rows); r.to_csv(DESC / f"paper_cost_curve_by_band{'' if V == 'theta' else '_h'}.csv", index=False)
    print("cost curve by band, mean ratio to the pooled curve:\n" + r.round(3).to_string(index=False))


def main() -> int:
    import textwrap  # noqa: F401
    apply_style()
    d = load_measure([V])
    cost = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "cost_index.parquet", columns=["pidp", "wave", COST]).dropna()
    cap = cost[COST].quantile(0.99)
    cost["y"] = cost[COST] if UNCAPPED else cost[COST].clip(upper=cap)
    cost["raw"] = cost[COST]
    cs = d.merge(cost[["pidp", "wave", "y", "raw"]], on=["pidp", "wave"])
    if "--curve" in sys.argv:
        curve_figure(cs, cap)
        return 0
    coef, dcoef, deg = M31.fit_cost_curve(cs[V].to_numpy(float), cs["y"].to_numpy(float))
    cprime = lambda z: np.polynomial.polynomial.polyval(z, dcoef)  # noqa: E731
    print(f"cost curve on {V}: degree {deg}, {'uncapped' if UNCAPPED else f'capped at p99 (£{cap:,.0f})'}, {len(cs):,} person-waves")
    long = pd.read_csv(PROCESSED_DATA_DIR / "contracts" / "health_lifecycle_20_89_minobs3_v1" / "long.csv").sort_values(["pidp", "age"]).reset_index(drop=True)
    A = (AGES - 55) / 10
    out, tabrows = {}, []
    for fs, name in SETS:
        if fs == "kmeans":        # hard assignments; paths are the smoothed type means
            L = load_labels(V)
            W = np.eye(K)[long["pidp"].map(L).to_numpy().astype(int)]
            traj = load_trajectories(V)
            paths = np.vstack([smooth_path(q["age"].to_numpy(), q["mean"].to_numpy(), q["count"].to_numpy(), min_count=25).to_numpy()
                               for c in range(K) for q in [traj[traj["cluster"] == c]]])
            shares = np.array([(L == c).mean() for c in range(K)])
        else:
            f = bayes_fit(ARTIFACTS_DIR / ("health-base" if fs == "base" else "health-ssm") / f"health-{V}-{fs}", V, K)
            post = f["post"].set_index("pidp")
            W = post.loc[long["pidp"], [f"class{k + 1}" for k in range(K)]].to_numpy()
            paths = np.vstack([f["mom"]["mean"] + f["mom"]["sd"] * (c[0] + c[1] * A + c[2] * A ** 2) for c in f["coef"]])
            shares = f["theta"]
        top = float(long[V].max())
        R = returns(long, W, paths, shares, cprime, top)
        out[fs] = (R, shares, name)
        R.assign(variant=V, set=fs, degree=deg).reset_index().to_csv(DESC / f"paper_returns_bayes{SFX}_{fs}.csv", index=False)
        for r in R_TABLE:
            q = R.loc[r]
            tabrows.append([name.replace('"spike"', "``spike''"), f"{r}", f"{q['T']:,.0f}", f"{q['P']:,.0f}", f"{q['P_over_T']:.2f}",
                            f"{100 * q['cov_share']:+.0f}\\%", *[f"{q[f'P_{c + 1}']:,.0f}" for c in range(K)], f"{q['type_gain']:.2f}"])
        print(f"{fs}: shares {np.round(shares, 3)}")
        print(R.loc[R_TABLE, ["T", "P", "P_over_T", "cov_share", "P_1", "P_2", "P_3", "best_class", "type_gain"]].round(3).to_string())
    write_table(TAB / f"tab_returns_bayes{SFX}.tex",
                ["classes", "$r$", "$T(r)$, \\pounds", "$P(r)$, \\pounds", "$P/T$", "between-class share", "$P_1$", "$P_2$", "$P_3$", "class gain"],
                tabrows, colspec="lrrrrrrrrr")
    # figure: one column per class set, as the K-means returns figure
    fig, axes = plt.subplots(3, len(SETS), figsize=(5.4 * len(SETS), 9.6), gridspec_kw={"hspace": 0.42, "wspace": 0.25}, squeeze=False)
    for j, (fs, _) in enumerate(SETS):
        R, pi, name = out[fs]
        ax = axes[0, j]
        for c in range(K):
            ax.plot(R.index, R[f"P_{c + 1}"], color=CLUSTER[c], lw=1.0 + 3.5 * pi[c], label=f"{'type' if KMEANS else 'class'} {c + 1} ({pi[c]:.0%})")
        ax.plot(R.index, R["P"], color=INK2, lw=1.4, ls="--", label="uniform $P(r)$")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(f"({'ab'[j] if len(SETS) > 1 else 'a'}) {VL}, {name}: $P_j(r)$, £", loc="left", fontsize=9.5); ax.legend(fontsize=7.5)
        ax = axes[1, j]
        ax.plot(R.index, R["T"], color=VERM, lw=1.8, label="treatment $T(r)$")
        ax.plot(R.index, R["P"], color=BLUE, lw=1.8, label="uniform prevention $P(r)$")
        ax.plot(R.index, R["mean_channel"], color=BLUE, lw=1.0, ls=":", label="its mean channel")
        ax.set_yscale("log"); ax.set_title(f"({'cd'[j] if len(SETS) > 1 else 'b'}) treatment and prevention, £ (log scale)", loc="left", fontsize=9.5); ax.legend(fontsize=7.5)
        ax = axes[2, j]
        ax.plot(R.index, R["P_over_T"], color=ORANGE, lw=1.8, label="$P(r)/T(r)$")
        ax.plot(R.index, R["type_gain"], color=CLUSTER[0], lw=1.8, label=r"$(\max_j P_j - P)/P$")
        ax.plot(R.index, R["cov_share"], color=CLUSTER[2], lw=1.4, ls="--", label=f"between-{'type' if KMEANS else 'class'} share of $P(r)$")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(f"({'ef'[j] if len(SETS) > 1 else 'c'}) ratios", loc="left", fontsize=9.5); ax.legend(fontsize=7.5); ax.set_xlabel("start age $r$")
        for a in axes[:, j]:
            a.grid(True, axis="y"); a.set_xlim(R_GRID[0], R_GRID[-1])
    import textwrap
    footer = (f"Returns on the K = 3 {'partial K-means types' if KMEANS else 'mixture classes'} on {V}, with the cost curve estimated in pounds on {V} (a degree-{deg} polynomial on the cost index, "
             f"{'uncapped' if UNCAPPED else 'capped at the 99th percentile'}, waves 7-15).\nTreatment closes a share tau of the gap to the healthiest observed score; "
             "prevention scales each class's future decline by (1 - mu). Pounds of lifetime cost to 90 per unit of intensity; no survival weighting, no discounting.\n"
             + ("Type shares and smoothed type means; E[-c'(h) | type] from the members observed at each age." if KMEANS else
                "Class shares and quadratic paths at the posterior mean; E[-c'(h) | class] from the person-waves observed at each age, weighted by the posterior class probabilities."))
    fig.text(0.01, 0.0, "\n".join(textwrap.wrap(footer.replace("\n", " "), 95 * len(SETS))), fontsize=7.2, color=INK2, va="top")
    fig.savefig(FIG / f"fig_returns_bayes{SFX}.png", bbox_inches="tight")
    print(f"wrote fig_returns_bayes{SFX}.png, tab_returns_bayes{SFX}.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
