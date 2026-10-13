"""Section 5: the treatment bill and the returns to prevention by class, in pounds.

Prevention is priced as the future treatment cost it avoids. The framework's objects
(Section 2.2, general-cost form) on the K = 3 classes of a typology:
  C(r)   = sum_{a>=r} E[ c(h_a) ]                                        the treatment bill still to come
         = sum_a [ c(hbar_a) + (sum_j pi_{j,a} c(hbar_{j,a}) - c(hbar_a)) + (E c(h_a) - sum_j pi_{j,a} c(hbar_{j,a})) ]
           mean path       between-class (type differences)              within-class (realised deviations, incl. the "spike")
  P_j(r) = sum_{a>=r} E[ -c'(h_a) | j ] (hbar_{j,r-1} - hbar_{j,a})    prevention, class j: treatment avoided per unit mu
  P(r)   = sum_j pi_j P_j(r) = mean channel + between-class channel
  P(r)/C(r): the share of the remaining bill a unit of prevention avoids; class targeting: max_j P_j(r) - P(r)
with c(.) the cost curve estimated in pounds on the measure (a cubic in the score on the cost
index, made monotone; fit_cost_curve in 31_paper_appendix_extras.py), on the uncapped index by
default (--capped to cap it at the 99th percentile). On the frailty indices, where higher is
frailer, the decline is a rise and -c' becomes +c'. Class shares pi_j and paths hbar_{j,a} are
the fitted typology (quadratics in age at the posterior mean for the mixtures; smoothed type
means for the K-means); E[-c'(h) | j] at each age is the posterior-weighted mean over the
person-waves observed there, and the population terms are the observed cross-section.
Horizon 90, no survival weighting, no discounting.

Typologies (``class_set``): kmeans; base (independent residuals); ssm (AR(1) + "spike");
ssm-cohort (AR(1) + "spike" with birth-decade shifts, paths at the 1950s; the decline is shift
invariant); multidim and multidim-cohort (the physical channel of the multidimensional fits,
on the multidim contract's rows). The mixtures' paths and posteriors come from _paper_common.bayes_fit
and 39_paper_multidim.load_fit.

Modes
  default            theta, ssm and base side by side: fig_returns_bayes.png (the detailed figure,
                     appendix) and tab_returns_bayes.tex (main text); --h, --kmeans, --capped as before
  --all              panel (a), P_j(r) by class, for six typologies on theta: fig_returns_by_model.png
  --measures         panel (a) under AR(1) + "spike" on theta, h, the frailty index and log frailty:
                     fig_returns_by_measure.png
  --curve            the cost-curve figure (fig_cost_curve.png)
47_paper_prevention_main.py composes the main-text 2 x 2 figure from this module and 46.

Outputs: paper/figures/fig_returns_bayes{,_h,_kmeans,_capped}.png, fig_returns_by_model.png,
fig_returns_by_measure.png; paper/tables/tab_returns_bayes*.tex;
artifacts/descriptives/paper_returns_bayes*_{set}.csv, paper_returns_by_model.csv, paper_returns_by_measure.csv
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import (  # noqa: E402
    AGES, DESC, FIG, HEALTHIER_HIGH, K, MAX_AGE, MIN_AGE, MODE_CHAINS, TAB, bayes_fit, load_labels, load_measure, load_trajectories,
    smooth_path, write_table)
from _style import BLUE, CLUSTER, INK2, ORANGE, VERM, apply_style, share_ylim  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR  # noqa: E402

M31 = importlib.import_module("31_paper_appendix_extras")
R_GRID = np.arange(25, 86)
R_TABLE = [30, 50, 70]
COST = "flat_cost_total"
CONTRACT = PROCESSED_DATA_DIR / "contracts" / "health_lifecycle_20_89_minobs3_v1"
AGE_BANDS = [(20, 34), (35, 44), (45, 54), (55, 64), (65, 74), (75, 90)]
MEASURE_LABEL = {"theta": r"$\theta$", "h": "$h$", "frailty": "frailty index", "logfrailty": "log(frailty + 1/31)"}
SET_NAMES = {"kmeans": "K-means types", "base": "independent-residuals classes", "ssm": 'AR(1) + "spike" classes',
             "ssm-cohort": 'AR(1) + "spike" classes, cohort shifts', "multidim": "multidimensional classes",
             "multidim-cohort": "multidimensional classes, cohort shifts"}
MD_TAGS = {("multidim", "theta"): ("theta-ssm-mort", None), ("multidim", "h"): ("h-ssm-mort", [1, 2, 3]),
           ("multidim-cohort", "theta"): ("theta-ssm-mort-cohort", None)}
PALETTE = {3: CLUSTER, 4: ["#08306b", "#2171b5", "#4292c6", "#9ecae1"],
           5: ["#08306b", "#08519c", "#2171b5", "#4292c6", "#9ecae1"]}     # worst -> best, as 36_paper_trajectories.py
CHAINS = MODE_CHAINS                                     # fits read over the chains that share a mode (_paper_common)


def smooth(x: pd.Series, n: pd.Series, min_n: int = 100) -> np.ndarray:
    """An age profile on every age 20-90: five-year rolling mean where supported, ends held flat."""
    y = x.where(n >= min_n).reindex(AGES)
    y = y.rolling(5, center=True, min_periods=2).mean()
    return y.interpolate(limit_direction="both").to_numpy()


def cost_curve(v: str, uncapped: bool = True):
    """The polynomial cost curve on measure ``v`` in pounds: c(.), c'(.), its degree, the fitting rows and the p99 cap."""
    d = load_measure([v])
    cost = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "cost_index.parquet", columns=["pidp", "wave", COST]).dropna()
    cap = cost[COST].quantile(0.99)
    cost["y"] = cost[COST] if uncapped else cost[COST].clip(upper=cap)
    cost["raw"] = cost[COST]
    cs = d.merge(cost[["pidp", "wave", "y", "raw"]], on=["pidp", "wave"]).dropna(subset=[v])
    coef, dcoef, deg = M31.fit_cost_curve(cs[v].to_numpy(float), cs["y"].to_numpy(float))
    cprime = lambda z: np.polynomial.polynomial.polyval(z, dcoef)  # noqa: E731
    cfun = lambda z: np.maximum(np.polynomial.polynomial.polyval(z, coef), 0.0)  # noqa: E731
    print(f"cost curve on {v}: degree {deg}, {'uncapped' if uncapped else f'capped at p99 (£{cap:,.0f})'}, {len(cs):,} person-waves")
    return cfun, cprime, deg, cs, cap


def returns(rows: pd.DataFrame, v: str, W: np.ndarray, paths: np.ndarray, pi: np.ndarray, cfun, cprime) -> pd.DataFrame:
    """The bill and the returns by start age, for a typology given by per-row class weights ``W``
    (rows x K), class paths on AGES and shares ``pi``, on the measure ``v``."""
    sign = 1.0 if HEALTHIER_HIGH.get(v, True) else -1.0           # a decline in health is a fall in theta, a rise in frailty
    x = rows[v].to_numpy(float)
    cp = -sign * cprime(x)                                          # the cost of one unit of decline, >= 0
    cost = cfun(x)
    age = rows["age"].to_numpy()
    g = pd.DataFrame({"age": age, "c": cost, "cp": cp, "m": x}).groupby("age")
    n = g.size()
    c_a, g_a, hbar = smooth(g["c"].mean(), n), smooth(g["cp"].mean(), n), smooth(g["m"].mean(), n)
    kc = W.shape[1]
    G, M, S = [], [], []
    for k in range(kc):
        wk = W[:, k]
        den = pd.Series(wk).groupby(age).sum()
        G.append(smooth(pd.Series(wk * cp).groupby(age).sum() / den, den, min_n=25))
        M.append(smooth(pd.Series(wk * x).groupby(age).sum() / den, den, min_n=25))
        S.append(smooth(den / n, n))
    G, M, S = np.vstack(G), np.vstack(M), np.vstack(S)
    S = S / S.sum(axis=0, keepdims=True)
    c_mean = cfun(hbar)
    c_between = (S * cfun(M)).sum(axis=0) - c_mean
    c_within = c_a - c_mean - c_between
    recs = []
    for r in R_GRID:
        idx = slice(r - MIN_AGE, MAX_AGE - MIN_AGE + 1)
        prevd = sign * (paths[:, r - 1 - MIN_AGE][:, None] - paths[:, idx])
        Pj = (G[:, idx] * prevd).sum(axis=1)
        P = float(pi @ Pj)
        mean_ch = float((g_a[idx] * sign * (hbar[r - 1 - MIN_AGE] - hbar[idx])).sum())
        C = float(c_a[idx].sum()); C_mean = float(c_mean[idx].sum()); C_between = float(c_between[idx].sum()); C_within = float(c_within[idx].sum())
        recs.append({"r": r, "C": C, "C_mean": C_mean, "C_between": C_between, "C_within": C_within,
                     "within_share": C_within / C, "between_share_C": C_between / C,
                     "P": P, "P_over_C": P / C, "mean_channel": mean_ch, "cov_channel": P - mean_ch,
                     "cov_share": (P - mean_ch) / P, "P_max_class": Pj.max(), "best_class": int(Pj.argmax()) + 1,
                     "type_gain": (Pj.max() - P) / P, **{f"P_{c + 1}": Pj[c] for c in range(kc)}})
    return pd.DataFrame(recs).set_index("r")


def savings(rows: pd.DataFrame, v: str, W: np.ndarray, paths: np.ndarray, pi: np.ndarray, cprime) -> dict:
    """When the saving arrives: the per-age contributions to P_j(r), s_j(a | r) = E[-c'(h_a) | j] (hbar_{j,r-1} - hbar_{j,a}),
    for every start age r on R_GRID and age a >= r, by class and for the uniform policy (pi-weighted).
    Returns arrays [K, len(R_GRID), len(AGES)] and [len(R_GRID), len(AGES)], zero before r."""
    sign = 1.0 if HEALTHIER_HIGH.get(v, True) else -1.0
    x = rows[v].to_numpy(float); cp = -sign * cprime(x); age = rows["age"].to_numpy()
    kc = W.shape[1]
    G = []
    for k in range(kc):
        wk = W[:, k]; den = pd.Series(wk).groupby(age).sum()
        G.append(smooth(pd.Series(wk * cp).groupby(age).sum() / den, den, min_n=25))
    G = np.vstack(G)
    S = np.zeros((kc, len(R_GRID), len(AGES)))
    for i, r in enumerate(R_GRID):
        idx = slice(r - MIN_AGE, MAX_AGE - MIN_AGE + 1)
        S[:, i, idx] = G[:, idx] * sign * (paths[:, r - 1 - MIN_AGE][:, None] - paths[:, idx])
    U = np.einsum("k,kra->ra", pi, S)
    return {"S": S, "U": U}


def timing_stats(S: np.ndarray, r: int, rate: float = 0.035) -> dict:
    """For a savings profile s(a | r) on AGES: the age by which half and 90% of the return has arrived,
    the share arriving within ten years, and the present value at ``rate`` as a share of the undiscounted sum."""
    prof = S[r - MIN_AGE:]; total = prof.sum()
    if total <= 0:
        return {"half_age": np.nan, "p90_age": np.nan, "within10": np.nan, "pv_share": np.nan}
    cum = np.cumsum(prof) / total
    ages = AGES[r - MIN_AGE:]
    disc = (1 + rate) ** -(ages - r)
    return {"half_age": float(ages[np.searchsorted(cum, 0.5)]), "p90_age": float(ages[np.searchsorted(cum, 0.9)]),
            "within10": float(cum[min(10, len(cum) - 1)]), "pv_share": float((prof * disc).sum() / total)}


def draw_timing_profiles(ax, sv: dict, cs: dict, r_list=(30, 50, 70), letter: str = "e") -> None:
    """Panel: the yearly saving by age for the uniform policy (thick) and each class (thin), for a few start ages."""
    kc = len(cs["shares"]); pal = PALETTE[kc]
    styles = ["-", "--", ":"]
    for r, ls in zip(r_list, styles):
        i = int(np.where(R_GRID == r)[0][0])
        ax.plot(AGES[r - MIN_AGE:], sv["U"][i, r - MIN_AGE:], color=INK2, lw=2.2, ls=ls, label=f"uniform, start {r}")
        for k in range(kc):
            ax.plot(AGES[r - MIN_AGE:], sv["S"][k, i, r - MIN_AGE:], color=pal[k], lw=0.9 + 2.0 * cs["shares"][k], ls=ls, alpha=0.9,
                    label=f"{cs['kind']} {k + 1}" if r == r_list[0] else None)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_title(f"({letter}) when the saving arrives: £ per year of age, by start age", loc="left", fontsize=9.5)
    ax.set_xlabel("age $a$"); ax.set_ylabel("saving in year $a$ per unit of $\\mu$, £"); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE + 5, MAX_AGE)
    ax.legend(fontsize=6.8, ncols=2)


def draw_timing_cumulative(ax, sv: dict, cs: dict, r_list=(30, 50, 70), letter: str = "f", rate: float = 0.035) -> None:
    """Panel: the share of the return collected by age, uniform (thick) and by class (thin), with the half-way ages."""
    kc = len(cs["shares"]); pal = PALETTE[kc]
    styles = ["-", "--", ":"]
    for r, ls in zip(r_list, styles):
        i = int(np.where(R_GRID == r)[0][0]); ages = AGES[r - MIN_AGE:]
        u = sv["U"][i, r - MIN_AGE:]; cum = np.cumsum(u) / u.sum()
        st = timing_stats(sv["U"][i], r, rate)
        ax.plot(ages, cum, color=INK2, lw=2.2, ls=ls, label=f"uniform, start {r}: half by {st['half_age']:.0f}, PV {st['pv_share']:.0%}")
        for k in range(kc):
            sk = sv["S"][k, i, r - MIN_AGE:]
            if sk.sum() > 0:
                ax.plot(ages, np.cumsum(sk) / sk.sum(), color=pal[k], lw=0.9 + 2.0 * cs["shares"][k], ls=ls, alpha=0.9)
    ax.axhline(0.5, color=INK2, lw=0.6, ls=":")
    ax.set_title(f"({letter}) share of the return collected by age (PV at {100 * rate:.1f}% as a share of the sum)", loc="left", fontsize=9.5)
    ax.set_xlabel("age $a$"); ax.set_ylim(0, 1); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE + 5, MAX_AGE)
    ax.legend(fontsize=6.8, loc="lower right")


def class_set(fs: str, v: str, kc: int = K) -> dict | None:
    """Rows, per-row class weights, class paths and shares for typology ``fs`` on measure ``v``
    with ``kc`` classes; None when that fit does not exist (the K = 4, 5 fits are theta and h only)."""
    A = (AGES - 55) / 10
    if fs.startswith("multidim"):
        M39 = importlib.import_module("39_paper_multidim")
        tag, chains = MD_TAGS[(fs, v)]
        f = M39.load_fit(tag, v, chains)
        long = pd.read_csv(M39.CONTRACT / "long.csv").sort_values(["pidp", "age"]).reset_index(drop=True)
        post = f["post"].set_index("pidp")
        W = post.loc[long["pidp"], [f"class{k + 1}" for k in range(K)]].to_numpy()
        return {"rows": long, "W": W, "paths": f["paths"][0], "shares": f["share"], "name": SET_NAMES[fs], "kind": "class"}
    long = pd.read_csv(CONTRACT / "long.csv").sort_values(["pidp", "age"]).reset_index(drop=True)
    if fs == "kmeans":
        L = load_labels(v)
        W = np.eye(K)[long["pidp"].map(L).to_numpy().astype(int)]
        traj = load_trajectories(v)
        paths = np.vstack([smooth_path(q["age"].to_numpy(), q["mean"].to_numpy(), q["count"].to_numpy(), min_count=25).to_numpy()
                           for c in range(K) for q in [traj[traj["cluster"] == c]]])
        return {"rows": long, "W": W, "paths": paths, "shares": np.array([(L == c).mean() for c in range(K)]), "name": SET_NAMES[fs], "kind": "type"}
    if kc != K:
        fit_dir = ARTIFACTS_DIR / "health-k45" / f"health-{v}-{fs}-k{kc}"
    else:
        folder = "health-frailty" if v in ("frailty", "logfrailty") else {"base": "health-base", "ssm": "health-ssm", "ssm-cohort": "health-ssm-cohort"}[fs]
        fit_dir = ARTIFACTS_DIR / folder / f"health-{v}-{fs}"
    if not (fit_dir / "run_summary.json").exists():
        return None
    f = bayes_fit(fit_dir, v, kc, chains=CHAINS.get(fit_dir.name))
    post = f["post"].set_index("pidp")
    W = post.loc[long["pidp"], [f"class{k + 1}" for k in range(kc)]].to_numpy()
    paths = np.vstack([f["mom"]["mean"] + f["mom"]["sd"] * (c[0] + c[1] * A + c[2] * A ** 2) for c in f["coef"]])
    return {"rows": long, "W": W, "paths": paths, "shares": f["theta"], "name": SET_NAMES[fs], "kind": "class"}


def compute(v: str, fs: str, uncapped: bool = True, curve=None, kc: int = K):
    """The returns table for typology ``fs`` on measure ``v``; ``curve`` reuses a cost curve.
    (None, None) when the fit does not exist."""
    cfun, cprime, deg, _, _ = curve or cost_curve(v, uncapped)
    cs = class_set(fs, v, kc)
    if cs is None:
        return None, None
    R = returns(cs["rows"], v, cs["W"], cs["paths"], cs["shares"], cfun, cprime)
    cs["deg"] = deg
    return R, cs


# ---- panels, shared with 47_paper_prevention_main.py --------------------------------------------
def draw_pj(ax, R: pd.DataFrame, cs: dict, v: str, letter: str, title: str | None = None) -> None:
    """Panel (a): the return by class and the uniform return, by start age."""
    kc = len(cs["shares"]); pal = PALETTE[kc]
    for c in range(kc):
        ax.plot(R.index, R[f"P_{c + 1}"], color=pal[c], lw=1.0 + 3.5 * cs["shares"][c], label=f"{cs['kind']} {c + 1} ({cs['shares'][c]:.0%})")
    ax.plot(R.index, R["P"], color=INK2, lw=1.4, ls="--", label="uniform $P(r)$")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_title(title or f"({letter}) {MEASURE_LABEL[v]}, {cs['name']}: $P_j(r)$, £", loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, ncols=2 if kc > 3 else 1); ax.grid(True, axis="y"); ax.set_xlim(R_GRID[0], R_GRID[-1])


def draw_bill(ax, R: pd.DataFrame, cs: dict, letter: str) -> None:
    """Panel (c): the bill stacked into its three parts, with the uniform return over it."""
    ax.stackplot(R.index, R["C_mean"], R["C_between"], R["C_within"], colors=[INK2, CLUSTER[1], "#d9d9d9"], alpha=0.85,
                 labels=["$C(r)$: cost at the mean path", f"+ {cs['kind']} differences (between)", "+ realised deviations (within)"])
    ax.plot(R.index, R["P"], color=VERM, lw=2.0, label="uniform prevention $P(r)$: treatment avoided")
    ax.plot(R.index, R["mean_channel"], color=VERM, lw=1.0, ls=":", label="its mean channel")
    ax.set_title(f"({letter}) the bill $C(r)$ and what prevention avoids, £", loc="left", fontsize=9.5)
    ax.legend(fontsize=7.0); ax.grid(True, axis="y"); ax.set_xlim(R_GRID[0], R_GRID[-1])


def draw_ratios(ax, R: pd.DataFrame, cs: dict, letter: str) -> None:
    ax.plot(R.index, R["P_over_C"], color=ORANGE, lw=1.8, label="$P(r)/C(r)$: share of the bill avoided")
    ax.plot(R.index, R["type_gain"], color=CLUSTER[0], lw=1.8, label=r"$(\max_j P_j - P)/P$")
    ax.plot(R.index, R["cov_share"], color=CLUSTER[2], lw=1.4, ls="--", label=f"between-{cs['kind']} share of $P(r)$")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_title(f"({letter}) ratios", loc="left", fontsize=9.5); ax.legend(fontsize=7.5); ax.set_xlabel("start age $r$")
    ax.grid(True, axis="y"); ax.set_xlim(R_GRID[0], R_GRID[-1])


def footer_text(v: str, deg: int, uncapped: bool, kmeans: bool) -> str:
    return (f"Returns on the K = 3 {'partial K-means types' if kmeans else 'mixture classes'} on {v}, with the cost curve estimated in pounds on {v} (a degree-{deg} polynomial on the cost index, "
            f"{'uncapped' if uncapped else 'capped at the 99th percentile'}, waves 7-15). C(r) is the expected treatment cost from r to 90 on the observed cross-section, "
            "split into the cost at the mean path, what the type differences add and what realised deviations add (the part prevention cannot reach); "
            "prevention scales each class's future decline by (1 - mu) and P(r) is the treatment cost that avoids, to first order, per unit of mu. Pounds to 90; no survival weighting, no discounting. "
            + ("Type shares and smoothed type means; E[-c'(h) | type] from the members observed at each age." if kmeans else
               "Class shares and quadratic paths at the posterior mean; E[-c'(h) | class] from the person-waves observed at each age, weighted by the posterior class probabilities."))


def curve_figure(v: str, cs: pd.DataFrame, cap: float) -> None:
    """The estimated cost curve on the measure: the pooled uncapped cubic, the same fit within six age
    bands (the age-invariance check the pricing rests on), and the capped fit, with twentieth-bin means."""
    vl = MEASURE_LABEL[v]
    x = cs[v].to_numpy(float); yraw = cs["raw"].to_numpy(float)
    lo, hi = np.quantile(x, [0.01, 0.99]); grid = np.linspace(lo, hi, 200)
    coef, _, deg = M31.fit_cost_curve(x, yraw)
    coef_c, _, deg_c = M31.fit_cost_curve(x, np.minimum(yraw, cap))
    pooled = np.polynomial.polynomial.polyval(grid, coef)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"wspace": 0.25})
    ax = axes[0]
    cmap = plt.get_cmap("viridis"); rows = []
    for bi, (blo, bhi) in enumerate(AGE_BANDS):
        sb = cs[cs["age"].between(blo, bhi)]
        cb = np.polynomial.polynomial.polyfit(sb[v].to_numpy(float), sb["raw"].to_numpy(float), deg)
        yb = np.polynomial.polynomial.polyval(grid, cb)
        ax.plot(grid, yb, color=cmap(0.9 * bi / (len(AGE_BANDS) - 1)), lw=1.3, label=f"{blo}-{bhi}")
        rows.append({"band": f"{blo}-{bhi}", "mean_ratio_to_pooled": float(np.mean(yb / pooled))})
    ax.plot(grid, pooled, color="black", lw=2.0, ls="--", label="pooled")
    ax.set_title(f"(a) cost curve on {vl}, refitted within age bands", loc="left", fontsize=9.5)
    ax.set_xlabel(f"{vl} (right = healthier)"); ax.set_ylabel("expected annual cost, £ (uncapped)"); ax.legend(fontsize=7.5, ncols=2); ax.grid(True, axis="y")
    ax = axes[1]
    b = pd.qcut(cs[v], 20, labels=False, duplicates="drop")
    g = cs.assign(b=b).groupby("b").agg(x=(v, "mean"), y=("raw", "mean"), yc=("raw", lambda z: np.minimum(z, cap).mean()))
    ax.plot(grid, pooled, color=VERM, lw=1.8, label=f"uncapped fit (degree {deg})")
    ax.plot(grid, np.polynomial.polynomial.polyval(grid, coef_c), color=BLUE, lw=1.8, label=f"capped at p99 (degree {deg_c})")
    ax.plot(g["x"], g["y"], "o", color=VERM, ms=3.5, mfc="none", label="twentieths, uncapped")
    ax.plot(g["x"], g["yc"], "o", color=BLUE, ms=3.5, mfc="none", label="twentieths, capped")
    ax.set_title("(b) the pooled fit against twentieths", loc="left", fontsize=9.5)
    ax.set_xlabel(f"{vl} (right = healthier)"); ax.legend(fontsize=7.5); ax.grid(True, axis="y")
    fig.text(0.01, -0.03, "\n".join(textwrap.wrap(
        f"The cost index in pounds, waves 7-15, {len(cs):,} person-waves, against {v}: a polynomial (cubic unless that is not monotone on the 1st-99th percentile range), "
        f"fitted pooled and within each age band, over the 1st-99th percentile range of {v}. Panel (b): the pooled fits on uncapped costs and on costs capped at the "
        f"99th percentile (£{cap:,.0f}), with the mean cost in twentieths of {v}.", 190)), fontsize=7.2, color=INK2, va="top")
    sfx = "" if v == "theta" else f"_{v}"
    fig.savefig(FIG / f"fig_cost_curve{sfx}.png", bbox_inches="tight"); plt.close(fig)
    r = pd.DataFrame(rows); r.to_csv(DESC / f"paper_cost_curve_by_band{sfx}.csv", index=False)
    print("cost curve by band, mean ratio to the pooled curve:\n" + r.round(3).to_string(index=False))


def main() -> int:
    apply_style()
    v = "h" if "--h" in sys.argv else "theta"
    uncapped = "--capped" not in sys.argv
    if "--curve" in sys.argv:
        _, _, _, cs, cap = cost_curve(v, True)
        curve_figure(v, cs, cap)
        return 0

    if "--all" in sys.argv:         # appendix: panel (a) for six typologies on theta
        curve = cost_curve("theta", True)
        sets = ["kmeans", "base", "ssm", "ssm-cohort", "multidim", "multidim-cohort"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), gridspec_kw={"hspace": 0.38, "wspace": 0.25})
        recs = []
        for ax, fs, letter in zip(axes.ravel(), sets, "abcdef"):
            R, cs = compute("theta", fs, True, curve)
            draw_pj(ax, R, cs, "theta", letter, title=f"({letter}) {cs['name']}")
            recs.append(R.assign(set=fs).reset_index())
            print(f"{fs}: shares {np.round(cs['shares'], 3)}; P(30) {R.loc[30, 'P']:,.0f}, P_j(30) " + " / ".join(f"{R.loc[30, f'P_{c + 1}']:,.0f}" for c in range(K)))
        for ax in axes[1]:
            ax.set_xlabel("start age $r$")
        share_ylim(list(axes.ravel()))
        pd.concat(recs).to_csv(DESC / "paper_returns_by_model.csv", index=False)
        fig.text(0.01, -0.02, "\n".join(textwrap.wrap(
            "The return to prevention by class, P_j(r), on theta under six typologies, all K = 3: the partial K-means types; the independent-residuals and AR(1) + \"spike\" mixtures, "
            "the latter also with birth-decade shifts (paths at the 1950s; the preventable decline is shift invariant); and the physical channel of the multidimensional fits (theta, the mental GRM and mortality), "
            "without and with the shifts, on the multidim contract's rows. Line width is proportional to the class share; dashed, the uniform return. Same cost curve throughout (the cubic in pounds on theta); "
            "one vertical scale for all panels.", 210)), fontsize=7.2, color=INK2, va="top")
        fig.savefig(FIG / "fig_returns_by_model.png", bbox_inches="tight")
        print(f"wrote fig_returns_by_model.png to {FIG}")
        return 0

    if "--timing" in sys.argv:      # when the savings arrive, by class and start age (the ssm classes on theta)
        cfun, cprime, deg, _, _ = cost_curve("theta", True)
        cs = class_set("ssm", "theta")
        sv = savings(cs["rows"], "theta", cs["W"], cs["paths"], cs["shares"], cprime)
        r_list = (30, 50, 70)
        fig, axes = plt.subplots(2, K, figsize=(15, 8), gridspec_kw={"hspace": 0.4, "wspace": 0.25})
        recs = []
        for k in range(K):
            ax = axes[0, k]
            for r, ls in zip(r_list, ["-", "--", ":"]):
                i = int(np.where(R_GRID == r)[0][0])
                ax.plot(AGES[r - MIN_AGE:], sv["S"][k, i, r - MIN_AGE:], color=CLUSTER[k], lw=1.8, ls=ls, label=f"start at {r}")
                st = timing_stats(sv["S"][k, i], r); recs.append({"class": k + 1, "r": r, "P_j": float(sv["S"][k, i].sum()), **st})
            ax.axhline(0, color=INK2, lw=0.6)
            ax.set_title(f"({'abc'[k]}) class {k + 1} ({cs['shares'][k]:.0%}): saving per year of age, £", loc="left", fontsize=9.5)
            ax.legend(fontsize=7.5); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE + 5, MAX_AGE)
            ax = axes[1, k]
            for r, ls in zip(r_list, ["-", "--", ":"]):
                i = int(np.where(R_GRID == r)[0][0]); sk = sv["S"][k, i, r - MIN_AGE:]
                st = timing_stats(sv["S"][k, i], r)
                ax.plot(AGES[r - MIN_AGE:], np.cumsum(sk) / sk.sum(), color=CLUSTER[k], lw=1.8, ls=ls,
                        label=f"start {r}: half by {st['half_age']:.0f}, 90% by {st['p90_age']:.0f}; PV {st['pv_share']:.0%}")
            ax.axhline(0.5, color=INK2, lw=0.6, ls=":")
            ax.set_title(f"({'def'[k]}) class {k + 1}: share of the return collected by age", loc="left", fontsize=9.5)
            ax.set_xlabel("age $a$"); ax.set_ylim(0, 1); ax.legend(fontsize=7, loc="lower right"); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE + 5, MAX_AGE)
        for r in r_list:
            i = int(np.where(R_GRID == r)[0][0]); st = timing_stats(sv["U"][i], r)
            recs.append({"class": 0, "r": r, "P_j": float(sv["U"][i].sum()), **st})
        share_ylim(list(axes[0]))
        t = pd.DataFrame(recs); t.to_csv(DESC / "paper_returns_timing.csv", index=False)
        print(t.round(3).to_string(index=False))
        fig.text(0.01, -0.02, "\n".join(textwrap.wrap(
            "The K = 3 AR(1) + \"spike\" classes on theta with the cost curve in pounds. Top: the saving in each year of age a from prevention started at r, s_j(a | r) = E[-c'(h_a) | j] (hbar_{j,r-1} - hbar_{j,a}), "
            "the per-age terms whose sum over a >= r is the class return P_j(r); for starts at 30, 50 and 70. Bottom: the cumulative share of P_j(r) collected by age a, with the ages by which half and 90% has arrived "
            "and the present value at 3.5% a year (the Green Book rate) as a share of the undiscounted return. Pounds per unit of mu; no survival weighting.", 210)), fontsize=7.2, color=INK2, va="top")
        fig.savefig(FIG / "fig_prevention_timing.png", bbox_inches="tight")
        print(f"wrote fig_prevention_timing.png to {FIG}")
        return 0

    if "--measures" in sys.argv:    # appendix: panel (a) under AR(1) + "spike" on the four measures, rows K = 3, 4, 5
        measures = ["theta", "h", "frailty", "logfrailty"]
        KS = [3, 4, 5]
        fig, axes = plt.subplots(len(KS), 4, figsize=(18, 4.2 * len(KS)), gridspec_kw={"wspace": 0.25, "hspace": 0.36})
        recs, curves, drawn, missing = [], {m: cost_curve(m, True) for m in measures}, [], []
        letters = iter("abcdefghijkl")
        for i, kc in enumerate(KS):
            for j, m in enumerate(measures):
                ax, letter = axes[i, j], next(letters)
                R, cs = compute(m, "ssm", True, curves[m], kc)
                if R is None:
                    missing.append(f"{m} K = {kc}")
                    ax.text(0.5, 0.5, "not fitted", ha="center", va="center", transform=ax.transAxes, color=INK2, fontsize=10)
                    ax.set_title(f"({letter}) {MEASURE_LABEL[m]}, $K = {kc}$", loc="left", fontsize=9.5)
                    ax.set_xticks([]); ax.set_yticks([])
                    for sp in ax.spines.values():
                        sp.set_visible(False)
                    continue
                draw_pj(ax, R, cs, m, letter, title=f"({letter}) {MEASURE_LABEL[m]}, $K = {kc}$")
                drawn.append(ax)
                if i == len(KS) - 1:
                    ax.set_xlabel("start age $r$")
                recs.append(R.assign(measure=m, K=kc).reset_index())
                print(f"{m} K={kc}: shares {np.round(cs['shares'], 3)}; P(30) {R.loc[30, 'P']:,.0f}, P_j(30) " + " / ".join(f"{R.loc[30, f'P_{c + 1}']:,.0f}" for c in range(kc)))
        share_ylim(drawn)
        pd.concat(recs).to_csv(DESC / "paper_returns_by_measure.csv", index=False)
        fig.text(0.01, -0.02, "\n".join(textwrap.wrap(
            "The return to prevention by class, P_j(r), under the AR(1) + \"spike\" mixture on each measure (columns) at K = 3, 4 and 5 (rows), each measure with its own cost curve in pounds (a cubic in the measure on the cost index, uncapped). "
            "On the frailty indices higher is frailer, so the preventable decline is the class's rise in frailty and the classes are relabelled with class 1 the frailest. Line width is proportional to the class share; "
            "dashed, the uniform return. One vertical scale for all panels." + (f" Not fitted: {', '.join(missing)}." if missing else "")
            + " The h fit at K = 5 is read over the three chains that share a mode.", 230)), fontsize=7.2, color=INK2, va="top")
        fig.savefig(FIG / "fig_returns_by_measure.png", bbox_inches="tight")
        print(f"wrote fig_returns_by_measure.png to {FIG}")
        return 0

    # ---- the detailed figure and the table: ssm and base side by side (or the K-means types)
    kmeans = "--kmeans" in sys.argv
    sfx = ("" if v == "theta" else "_h") + ("" if uncapped else "_capped") + ("_kmeans" if kmeans else "")
    sets = ["kmeans"] if kmeans else ["ssm", "base"]
    curve = cost_curve(v, uncapped)
    out, tabrows = {}, []
    for fs in sets:
        R, cs = compute(v, fs, uncapped, curve)
        out[fs] = (R, cs)
        R.assign(variant=v, set=fs, degree=cs["deg"]).reset_index().to_csv(DESC / f"paper_returns_bayes{sfx}_{fs}.csv", index=False)
        for r in R_TABLE:
            q = R.loc[r]
            tabrows.append([cs["name"].replace('"spike"', "``spike''"), f"{r}", f"{q['C']:,.0f}", f"{100 * q['within_share']:.0f}\\%", f"{q['P']:,.0f}", f"{q['P_over_C']:.2f}",
                            f"{100 * q['cov_share']:+.0f}\\%", *[f"{q[f'P_{c + 1}']:,.0f}" for c in range(K)], f"{q['type_gain']:.2f}"])
        print(f"{fs}: shares {np.round(cs['shares'], 3)}")
        print(R.loc[R_TABLE, ["C", "C_mean", "C_between", "C_within", "within_share", "P", "P_over_C", "cov_share", "P_1", "P_2", "P_3", "best_class", "type_gain"]].round(3).to_string())
    write_table(TAB / f"tab_returns_bayes{sfx}.tex",
                ["classes", "$r$", "$C(r)$", "within", "$P(r)$", "$P/C$", "between", "$P_1$", "$P_2$", "$P_3$", "gain"],
                tabrows, colspec="lrrrrrrrrrr")
    fig, axes = plt.subplots(3, len(sets), figsize=(5.4 * len(sets), 9.6), gridspec_kw={"hspace": 0.42, "wspace": 0.25}, squeeze=False)
    letters = "abcdef" if len(sets) > 1 else "abc"
    for j, fs in enumerate(sets):
        R, cs = out[fs]
        draw_pj(axes[0, j], R, cs, v, letters[j] if len(sets) > 1 else "a")
        draw_bill(axes[1, j], R, cs, letters[2 + j] if len(sets) > 1 else "b")
        draw_ratios(axes[2, j], R, cs, letters[4 + j] if len(sets) > 1 else "c")
    for r in range(3):
        share_ylim(list(axes[r, :]))
    fig.text(0.01, 0.0, "\n".join(textwrap.wrap(footer_text(v, curve[2], uncapped, kmeans), 95 * len(sets))), fontsize=7.2, color=INK2, va="top")
    fig.savefig(FIG / f"fig_returns_bayes{sfx}.png", bbox_inches="tight")
    print(f"wrote fig_returns_bayes{sfx}.png, tab_returns_bayes{sfx}.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
