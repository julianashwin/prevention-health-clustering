"""Shared helpers for the paper's figure scripts (descriptives/2x_paper_*.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, ROOT_DIR

DESC = ARTIFACTS_DIR / "descriptives"
FIG = ROOT_DIR / "paper" / "figures"
TAB = ROOT_DIR / "paper" / "tables"
MIN_AGE, MAX_AGE, K = 20, 90, 3
AGES = np.arange(MIN_AGE, MAX_AGE + 1)
VARIANTS = [("h", "$h$"), ("theta", r"$\theta$"), ("fi10", "deficit index")]
LABEL = dict(VARIANTS)


def load_measure(variants=("h", "theta", "fi10")) -> pd.DataFrame:
    d = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "health_measure.parquet",
                        columns=["pidp", "wave", "age", *variants])
    return d[d["age"].between(MIN_AGE, MAX_AGE)].assign(age=lambda x: x["age"].astype(int))


def load_labels(variant: str, lo: int = MIN_AGE, hi: int = MAX_AGE) -> pd.Series:
    lab = pd.read_parquet(DESC / "paper_kmeans_labels.parquet")
    lab = lab[(lab["variant"] == variant) & (lab["lo"] == lo) & (lab["hi"] == hi)]
    return lab.set_index("pidp")["cluster"].astype(int)


def load_trajectories(variant: str, lo: int = MIN_AGE, hi: int = MAX_AGE) -> pd.DataFrame:
    t = pd.read_csv(DESC / "paper_kmeans_trajectories.csv")
    return t[(t["variant"] == variant) & (t["lo"] == lo) & (t["hi"] == hi)].copy()


def smooth_path(age: np.ndarray, value: np.ndarray, count: np.ndarray, *, min_count: int,
                lo: int = MIN_AGE, hi: int = MAX_AGE, window: int = 5) -> pd.Series:
    """A cluster or population mean path on every age lo..hi.

    Five-year centred rolling mean over the ages with at least ``min_count``
    observations; the ends, where the rolling window or the support runs out,
    are extended linearly at the slope of the last five available years.
    """
    s = pd.Series(value, index=age)[count >= min_count].sort_index()
    s = s.reindex(range(s.index.min(), s.index.max() + 1)).interpolate()
    r = s.rolling(window, center=True).mean().dropna()
    out = pd.Series(np.nan, index=range(lo, hi + 1), dtype=float)
    out.loc[r.index] = r.to_numpy()
    first, last = r.index.min(), r.index.max()
    slope_lo = (r.loc[first + 4] - r.loc[first]) / 4 if last - first >= 4 else 0.0
    slope_hi = (r.loc[last] - r.loc[last - 4]) / 4 if last - first >= 4 else 0.0
    for a in range(lo, first):
        out.loc[a] = r.loc[first] + slope_lo * (a - first)
    for a in range(last + 1, hi + 1):
        out.loc[a] = r.loc[last] + slope_hi * (a - last)
    return out


def write_table(path, header: list[str], rows: list[list[str]], colspec: str | None = None) -> None:
    colspec = colspec or "l" + "r" * (len(header) - 1)
    with open(path, "w") as f:
        f.write("\\begin{tabular}{" + colspec + "}\n\\toprule\n")
        f.write(" & ".join(header) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def profile_rows(d: pd.DataFrame, v: str, *, row_len: int = 9, min_cell: int = 100):
    """Exhibit 1's ingredients for one variable: the smoothed mean and variance
    profile by age, and the rows of the age-covariance matrix leaving the
    diagonal at five-year base-age bins and running ``row_len`` years."""
    prof = d.groupby("age")[v].agg(["mean", "var", "count"])
    prof = prof[prof["count"] >= min_cell]
    smooth = prof[["mean", "var"]].rolling(5, center=True).mean().dropna()
    base = d[d["age"] <= MAX_AGE - row_len]
    pairs = base.merge(d, on="pidp", suffixes=("", "_t"))
    pairs = pairs[(pairs["age_t"] >= pairs["age"]) & (pairs["age_t"] - pairs["age"] <= row_len)]
    g = pairs.groupby(["age", "age_t"])
    cell = pd.DataFrame({"cov": g.apply(lambda x: np.cov(x[v], x[f"{v}_t"], ddof=1)[0, 1], include_groups=False),
                         "n": g.size()}).reset_index()
    cell = cell[cell["n"] >= min_cell]
    cell["start"] = 5 * (cell["age"] // 5)
    cell["lag"] = cell["age_t"] - cell["age"]
    rows = (cell.groupby(["start", "lag"])
            .apply(lambda x: pd.Series({"cov": np.average(x["cov"], weights=x["n"]), "n": x["n"].sum()}),
                   include_groups=False).reset_index())
    rows["later_age"] = rows["start"] + rows["lag"]
    return smooth, rows


def draw_exhibit1_row(axes, smooth, rows, label: str, letters: str = "abc", legend: bool = False):
    """Three Exhibit 1 panels (mean, variance, rows) on a row of axes."""
    import matplotlib.pyplot as plt
    from _style import BLUE, VERM
    ylim = (min(0, rows["cov"].min(), smooth["var"].min()), max(rows["cov"].max(), smooth["var"].max()) * 1.05)
    axes[0].plot(smooth.index, smooth["mean"], color=BLUE, lw=1.8)
    axes[0].set_title(f"({letters[0]}) mean, {label}", loc="left", fontsize=9.5)
    axes[1].plot(smooth.index, smooth["var"], color=VERM, lw=1.8)
    axes[1].set_ylim(*ylim); axes[1].set_title(f"({letters[1]}) variance", loc="left", fontsize=9.5)
    cmap = plt.get_cmap("viridis")
    starts = sorted(rows["start"].unique())
    for k, st in enumerate(starts):
        seg = rows[rows["start"] == st]
        axes[2].plot(seg["later_age"], seg["cov"], color=cmap(0.9 * k / max(len(starts) - 1, 1)), lw=1.0,
                     marker="o", ms=1.8, label=f"{st}" if st % 10 == 0 else None)
    axes[2].plot(smooth.index, smooth["var"], color=VERM, lw=0.8, ls=":", alpha=0.7)
    axes[2].set_ylim(*ylim); axes[2].set_title(f"({letters[2]}) rows of the covariance matrix", loc="left", fontsize=9.5)
    if legend:
        axes[2].legend(title="base age", ncols=2, fontsize=7.5, title_fontsize=7.5, loc="upper left")
    for ax in axes:
        ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
