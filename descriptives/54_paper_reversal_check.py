"""Is the level-slope reversal driven by the unhealthy tail? Two appendix figures on theta.

Figure 1 (fig_exhibit2_samples.png): Exhibit 2 (Figure 3; the Case 4 fit of 29_paper_moments.py,
main design) refitted on six samples, one row each: everyone; everyone but the bottom decile,
quintile and quartile of the person's mean age-standardised theta (theta standardised within
single year of age, then averaged over the person's rows); everyone but the K-means worst type;
everyone but the modal worst class of the AR(1) + "spike" mixture. The csv records Corr(H, d),
C_Hd and V_H by band and sample, with the constraint flags.

Figure 2 (fig_reversal_types.png): the level-slope covariance across types split into each
type's term pi_j(a) (hbar_j,a - hbar_a)(d_j,a - d_a), age-specific composition shares, for
K-means and the univariate mixture at K = 3 and the multidimensional mixture at K = 4 and 5,
plus the K-means covariance with slopes from within-person changes (next observed age minus
this one, per year, gaps of one to three years) instead of the type mean paths.

    PYTHONPATH=src .venv/bin/python descriptives/54_paper_reversal_check.py
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
from _paper_common import AGES, DESC, FIG, K, MAX_AGE, MIN_AGE, load_labels, load_measure, load_trajectories, smooth_path, ssm_fit  # noqa: E402
from _style import BLUE, INK2, PURPLE, apply_style  # noqa: E402

M29 = importlib.import_module("29_paper_moments")
M39 = importlib.import_module("39_paper_multidim")
QUANTILES = [("decile", 0.10), ("quintile", 0.20), ("quartile", 0.25)]


def samples(d: pd.DataFrame) -> dict:
    z = d.assign(z=lambda x: (x["theta"] - x.groupby("age")["theta"].transform("mean")) / x.groupby("age")["theta"].transform("std"))
    pz = z.groupby("pidp")["z"].mean()
    out = {"everyone": d["pidp"].unique()}
    for name, q in QUANTILES:
        out[f"without the bottom {name}"] = pz[pz > pz.quantile(q)].index.to_numpy()
    L = load_labels("theta")
    out["without the K-means worst type"] = np.setdiff1d(d["pidp"].unique(), L[L == 0].index)
    r = ssm_fit("theta"); modal = pd.Series(r["post"][["class1", "class2", "class3"]].to_numpy().argmax(1), index=r["post"]["pidp"])
    out['without the AR(1) + "spike" worst class'] = np.setdiff1d(d["pidp"].unique(), modal[modal == 0].index)
    return out


def exhibit2_samples() -> None:
    d = M29.ident_panel(variants=("theta",))
    spec = M29.DESIGNS["main"]; fits = {}; rows = []
    for name, pids in samples(d).items():
        key = name
        M29.LABEL[key] = name.replace("without the ", "without\n")          # the row label the bar chart prints
        fits[key] = M29.fit_design(M29.wide(d[d["pidp"].isin(pids)], "theta"), spec)
        for (bn, case), (m, n, f) in fits[key].items():
            if case == "case4":
                rows.append({"sample": name, "band": bn, "n": n, "people": len(pids), "corr": M29.corr_of(f["b"]), "chd": f["b"][1], "vh": f["b"][0],
                             "binds": bool(f.get("binds")), "weak": bool(f.get("weak"))})
    T = pd.DataFrame(rows); T.to_csv(DESC / "paper_exhibit2_samples.csv", index=False)
    print(T.pivot(index="sample", columns="band", values="corr").round(2).to_string())
    footer = ("Exhibit 2 (Case 4: AR(1) plus one-period error) on theta, main design (four ages four years apart, pairwise moments pooled within band), refitted on six samples, one per row: everyone; everyone but the bottom decile, quintile and quartile of the person's mean "
              "age-standardised theta (standardised within single year of age, then averaged over the person's rows); everyone but the K-means worst type; everyone but the modal worst class of the AR(1) + \"spike\" mixture. "
              "Bars stack the fitted layers, black ticks the observed moments; the panel title gives rho and Corr(H, d), marked where the constraint |Corr| <= 1 binds (the estimator at its corner).")
    M29.barchart(FIG / "fig_exhibit2_samples.png", [(k, "case4") for k in fits], fits, spec, "", "\n".join(textwrap.wrap(footer, 210)))
    print(f"wrote fig_exhibit2_samples.png to {FIG}")


def contributions(paths, slopes, comp) -> pd.DataFrame:
    gm = (comp * paths.T).sum(1); gd = (comp * slopes.T).sum(1)
    return pd.DataFrame(comp * (paths.T - gm[:, None]) * (slopes.T - gd[:, None]), index=AGES).rolling(10, center=True, min_periods=1).mean()


def reversal_types() -> None:
    L = load_labels("theta"); dm = load_measure(["theta"]); s = dm[dm["pidp"].isin(L.index)].assign(cluster=lambda x: x["pidp"].map(L))
    cell = s.groupby(["age", "cluster"])["theta"].agg(["mean", "count"]).reset_index(); cell["share"] = cell["count"] / cell.groupby("age")["count"].transform("sum")
    comp_k = cell.pivot(index="age", columns="cluster", values="share").reindex(AGES).ffill().bfill().to_numpy()
    traj = load_trajectories("theta")
    paths_k = np.vstack([smooth_path(t["age"].to_numpy(), t["mean"].to_numpy(), t["count"].to_numpy(), min_count=25, window=10).to_numpy() for c in range(K) for t in [traj[traj["cluster"] == c]]])
    sl_k = np.diff(paths_k, axis=1); sl_k = np.column_stack([sl_k, sl_k[:, -1]])
    sets = [("K-means, $K=3$", contributions(paths_k, sl_k, comp_k))]
    r = ssm_fit("theta"); comp_b = r["comp"][["class1", "class2", "class3"]].reindex(AGES).ffill().bfill().to_numpy()
    sets.append(('AR(1) + "spike", $K=3$', contributions(r["paths"], r["slopes"], comp_b)))
    for k, tag, ch in ((4, "theta-ssm-mort-k4", [1, 3, 4]), (5, "theta-ssm-mort-k5", [1])):
        f = M39.load_fit(tag, "theta", ch, kc=k)
        comp = f["comp"][[f"class{j + 1}" for j in range(k)]].reindex(AGES).ffill().bfill().to_numpy()
        sets.append((f'multidimensional AR(1) + "spike", $K={k}$', contributions(f["paths"][0], np.gradient(f["paths"][0], AGES, axis=1), comp)))
    # within-person slopes for the K-means types
    ss = s.sort_values(["pidp", "age"]); g = ss.groupby("pidp"); nxt_age, nxt_th = g["age"].shift(-1), g["theta"].shift(-1)
    pairs = ss.assign(gap=nxt_age - ss["age"], dth=nxt_th - ss["theta"]).dropna(subset=["dth"])
    pairs = pairs[pairs["gap"].between(1, 3)].assign(slope=lambda x: x["dth"] / x["gap"], mid=lambda x: x["age"] + x["gap"] / 2)
    wp = np.full((K, len(AGES)), np.nan)
    for c in range(K):
        q = pairs[pairs["cluster"] == c]
        for i, a in enumerate(AGES):
            sel = q[(q["mid"] >= a - 5) & (q["mid"] < a + 5)]
            if len(sel) >= 100:
                wp[c, i] = sel["slope"].mean()
    def cov_series(paths, slopes, comp):
        sl = np.nan_to_num(slopes); gm = (comp * paths.T).sum(1); gd = (comp * sl.T).sum(1)
        c = (comp * (paths.T - gm[:, None]) * (sl.T - gd[:, None])).sum(1); c[np.isnan(slopes).any(0)] = np.nan
        return pd.Series(c, index=AGES).rolling(10, center=True, min_periods=1).mean()
    cov_path, cov_wp = cov_series(paths_k, sl_k, comp_k), cov_series(paths_k, wp, comp_k)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), gridspec_kw={"hspace": 0.45, "wspace": 0.28})
    for j, (name, C) in enumerate(sets):
        ax = axes.flat[j]; k = C.shape[1]; cols = M39.PALETTE[k]
        ax.stackplot(AGES, *[C[c].clip(lower=0) for c in range(k)], colors=cols, alpha=0.85)
        ax.stackplot(AGES, *[C[c].clip(upper=0) for c in range(k)], colors=cols, alpha=0.85)
        ax.plot(AGES, C.sum(1), color=INK2, lw=1.6, label="total")
        for c in range(k):
            ax.plot([], [], color=cols[c], lw=6, label=f"type {c + 1}")
        ax.axhline(0, color=INK2, lw=0.8); ax.set_title(f"({'abcd'[j]}) contributions by type: {name}", loc="left", fontsize=9.5); ax.legend(fontsize=6.5, ncols=2, loc="lower left")
    ax = axes.flat[4]
    ax.plot(AGES, cov_path, color=BLUE, lw=1.8, label="slopes of the type mean paths")
    ax.plot(AGES, cov_wp, color=PURPLE, lw=1.8, ls="--", label="within-person changes by type")
    ax.axhline(0, color=INK2, lw=0.8); ax.set_title("(e) K-means, $K=3$: path-based against within-person slopes", loc="left", fontsize=9.5); ax.legend(fontsize=7)
    axes.flat[5].axis("off")
    for ax in axes.flat[:5]:
        ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
    for ax in axes[1, :2]:
        ax.set_xlabel("age")
    footer = ("The level-slope covariance across types, Cov_j(hbar_j,a, d_j,a), split into each type's term pi_j(a)(hbar_j,a - hbar_a)(d_j,a - d_a) with age-specific composition shares, stacked: (a) the K-means types and (b) the univariate AR(1) + \"spike\" classes at K = 3, "
              "(c), (d) the multidimensional mixture's physical channel at K = 4 and 5 (chains sharing the better mode; K = 5 from its best chain). A type's term is negative when it sits below the mean and declines more slowly than the mean, or above and faster: it is closing in. "
              "(e) The K-means covariance with slopes from within-person changes (next observed age minus this one, per year of gap, gaps of one to three years, ten-year window by midpoint age) instead of the type mean paths, which removes composition and selection from the slopes. Ten-year rolling means.")
    fig.text(0.01, -0.03, "\n".join(textwrap.wrap(footer, 230)), fontsize=7.2, color=INK2, va="top")
    fig.savefig(FIG / "fig_reversal_types.png", bbox_inches="tight")
    print(f"wrote fig_reversal_types.png to {FIG}")


def main() -> int:
    apply_style()
    exhibit2_samples()
    reversal_types()
    return 0


if __name__ == "__main__":
    sys.exit(main())
