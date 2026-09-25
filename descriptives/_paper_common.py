"""Shared helpers for the paper's figure scripts (descriptives/2x_paper_*.py)."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from pathlib import Path

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, ROOT_DIR

DESC = ARTIFACTS_DIR / "descriptives"
FIG = ROOT_DIR / "paper" / "figures"
TAB = ROOT_DIR / "paper" / "tables"
CONTRACT = PROCESSED_DATA_DIR / "contracts" / "health_lifecycle_20_89_minobs3_v1"
MIN_AGE, MAX_AGE, K = 20, 90, 3
AGES = np.arange(MIN_AGE, MAX_AGE + 1)
VARIANTS = [("h", "$h$"), ("theta", r"$\theta$"), ("fi10", "deficit index")]
LABEL = dict(VARIANTS)


def load_measure(variants=("h", "theta", "fi10")) -> pd.DataFrame:
    d = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "health_measure.parquet",
                        columns=["pidp", "wave", "age", *variants])
    return d[d["age"].between(MIN_AGE, MAX_AGE)].assign(age=lambda x: x["age"].astype(int))


def load_cost_rows(cap_q: float = 0.99) -> pd.DataFrame:
    """The cost index on the contract's people, shaped like the contract: ages 20-89,
    one row per person-age (duplicates averaged), waves 7-15 only since that is when
    the index exists. Capped at the ``cap_q`` quantile so the top 1% does not drive a
    squared-error clustering. Column ``cost``."""
    c = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "cost_index.parquet",
                        columns=["pidp", "age", "flat_cost_total"]).dropna()
    people = pd.read_csv(CONTRACT / "person_roster.csv", usecols=["pidp"])["pidp"]
    c = c[c["pidp"].isin(people) & c["age"].between(MIN_AGE, 89)].assign(age=lambda x: x["age"].astype(int))
    c["cost"] = c["flat_cost_total"].clip(upper=c["flat_cost_total"].quantile(cap_q))
    return c.groupby(["pidp", "age"], as_index=False)["cost"].mean()


def load_predicted_cost_rows(v: str = "h", cap_q: float = 0.99, degree: int = 3) -> pd.DataFrame:
    """Each contract row's predicted cost in pounds: the pooled cost curve on variant ``v``
    (a polynomial in the standardised measure fitted to costs capped at ``cap_q``, waves
    7-15, then made monotone decreasing in health) evaluated at every contract row.
    A cost-anchored cardinalisation of the measure. Column ``predcost``."""
    long = pd.read_csv(CONTRACT / "long.csv", usecols=["pidp", "age", v])
    c = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "cost_index.parquet", columns=["age", v, "flat_cost_total"]).dropna()
    c = c[c["age"].between(MIN_AGE, MAX_AGE)]
    y = c["flat_cost_total"].clip(upper=c["flat_cost_total"].quantile(cap_q)).to_numpy()
    m, sd = c[v].mean(), c[v].std()
    b = np.polyfit(((c[v] - m) / sd).to_numpy(), y, degree)
    z = ((long[v] - m) / sd).to_numpy()
    grid = np.linspace(z.min(), z.max(), 400)
    curve = np.minimum.accumulate(np.polyval(b, grid))
    return long[["pidp", "age"]].assign(predcost=np.interp(z, grid, curve))


def observed_class_means(post: pd.DataFrame, v: str, Kc: int, min_weight: float = 25.0) -> pd.DataFrame:
    """Posterior-weighted mean of the measure by class and age on the contract rows:
    the dotted 'observed' lines drawn against fitted class paths. NaN where the
    summed posterior weight at an age is below ``min_weight`` people."""
    cols = [f"class{k + 1}" for k in range(Kc)]
    long = pd.read_csv(CONTRACT / "long.csv", usecols=["pidp", "age", v])
    d = long.merge(post[["pidp"] + cols], on="pidp")
    out = {}
    for c in cols:
        w = d[c].to_numpy(); y = d[v].to_numpy()
        g = pd.DataFrame({"age": d["age"], "wy": w * y, "w": w}).groupby("age").sum()
        out[c] = (g["wy"] / g["w"]).where(g["w"] >= min_weight)
    return pd.DataFrame(out)


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


def profile_rows_balanced(d: pd.DataFrame, v: str, *, row_len: int = 9, min_cell: int = 100):
    """Exhibit 1's covariance rows on balanced samples: each row over the people
    observed at every one of its ages, base to base + ``row_len``. The pairwise
    version (profile_rows) uses, for each cell, everyone observed at both of its
    two ages; this one uses a single sample per row, so the whole row describes
    the same people, at the cost of most of the sample and the oldest rows."""
    W = d.drop_duplicates(["pidp", "age"]).pivot(index="pidp", columns="age", values=v)
    M = W.reindex(columns=range(MIN_AGE, MAX_AGE + 1)).to_numpy()
    rows = []
    for a in range(MIN_AGE, MAX_AGE - row_len + 1):
        X = M[:, a - MIN_AGE:a - MIN_AGE + row_len + 1]
        ok = ~np.isnan(X).any(axis=1)
        if ok.sum() < min_cell:
            continue
        Xb = X[ok]
        rows += [{"age": a, "lag": k, "cov": np.cov(Xb[:, 0], Xb[:, k], ddof=1)[0, 1], "n": int(ok.sum())}
                 for k in range(row_len + 1)]
    cell = pd.DataFrame(rows)
    cell["start"] = 5 * (cell["age"] // 5)
    out = (cell.groupby(["start", "lag"])
           .apply(lambda x: pd.Series({"cov": np.average(x["cov"], weights=x["n"]), "n": x["n"].sum()}),
                  include_groups=False).reset_index())
    out["later_age"] = out["start"] + out["lag"]
    return out


def draw_exhibit1_row(axes, smooth, rows, label: str, letters: str = "abc", legend: bool = False,
                      titles: "list[str] | None" = None):
    """Three Exhibit 1 panels (mean, variance, covariance rows) on three axes.

    The axes may be a row or a column; pass ``titles`` to override the default
    captions, which name the variant only on the first panel.
    """
    import matplotlib.pyplot as plt
    from _style import BLUE, VERM
    ylim = (min(0, rows["cov"].min(), smooth["var"].min()), max(rows["cov"].max(), smooth["var"].max()) * 1.05)
    t = titles or [f"({letters[0]}) mean, {label}", f"({letters[1]}) variance",
                   f"({letters[2]}) rows of the covariance matrix"]
    axes[0].plot(smooth.index, smooth["mean"], color=BLUE, lw=1.8)
    axes[0].set_title(t[0], loc="left", fontsize=9.5)
    axes[1].plot(smooth.index, smooth["var"], color=VERM, lw=1.8)
    axes[1].set_ylim(*ylim); axes[1].set_title(t[1], loc="left", fontsize=9.5)
    cmap = plt.get_cmap("viridis")
    starts = sorted(rows["start"].unique())
    for k, st in enumerate(starts):
        seg = rows[rows["start"] == st]
        axes[2].plot(seg["later_age"], seg["cov"], color=cmap(0.9 * k / max(len(starts) - 1, 1)), lw=1.0,
                     marker="o", ms=1.8, label=f"{st}" if st % 10 == 0 else None)
    axes[2].plot(smooth.index, smooth["var"], color=VERM, lw=0.8, ls=":", alpha=0.7)
    axes[2].set_ylim(*ylim); axes[2].set_title(t[2], loc="left", fontsize=9.5)
    if legend:
        axes[2].legend(title="base age", ncols=2, fontsize=7.5, title_fontsize=7.5, loc="upper left")
    for ax in axes:
        ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)


# ---- reading a fitted mixture at its posterior mean ---------------------------------


def kalman_person_lp(Y, X, coef, sigma, sigma_meas, rho, gap, fs, fe):
    """Per-person, per-class log-likelihood under ar_mode 2, as the Stan program computes it."""
    n_person, Kc = len(fs), coef.shape[0]
    out = np.zeros((n_person, Kc))
    mu_all = X @ coef.T
    v_stat = sigma ** 2 / (1 - rho ** 2)
    var_meas = sigma_meas ** 2
    for i in range(n_person):
        rows = np.arange(fs[i] - 1, fe[i])
        a = np.zeros(Kc); Pv = v_stat.copy(); total = np.zeros(Kc)
        for j, n in enumerate(rows):
            if j > 0:
                g = gap[n]
                rg = rho ** g
                a = rg * a
                Pv = rg ** 2 * Pv + v_stat * np.maximum(1 - rho ** (2 * g), 1e-9)
            F = Pv + var_meas
            v = Y[n] - mu_all[n] - a
            total += -0.5 * (np.log(2 * np.pi * F) + v ** 2 / F)
            Kg = Pv / F
            a = a + Kg * v
            Pv = Pv - Kg * Pv
        out[i] = total
    return out


def _chain_means(fit_dir: Path, chains) -> tuple[dict, float]:
    """Posterior means over a subset of chains, read from the CmdStan csvs, and the
    split R-hat of the structural parameters over that subset. Used when one chain sat
    in a separate, worse mode so that the pooled summary would average across modes."""
    files = sorted(Path(fit_dir, "chains").glob("*_[0-9].csv"))
    keep = [f for f in files if int(f.stem.rsplit("_", 1)[1]) in set(chains)]
    assert len(keep) == len(chains), f"chains {chains} not all found in {fit_dir}"
    dfs = [pd.read_csv(f, comment="#") for f in keep]
    struct = [c for c in dfs[0].columns if re.match(r"^(theta|coef|sigma|rho|sigma_meas)\.", c) or c == "lp__"]
    pooled = pd.concat(dfs)
    out, worst = {}, 0.0
    for c in struct:
        name = c if c == "lp__" else re.sub(r"^(\w+)\.(.+)$", lambda m: f"{m[1]}[{m[2].replace('.', ',')}]", c)
        out[name] = {"mean": float(pooled[c].mean())}
        halves = [x[: len(x) // 2] for x in (d[c].to_numpy() for d in dfs)] + [x[len(x) // 2:] for x in (d[c].to_numpy() for d in dfs)]
        n = len(halves[0]); mu = np.array([h.mean() for h in halves]); vs = np.array([h.var(ddof=1) for h in halves])
        W = vs.mean(); B = n * mu.var(ddof=1)
        if W > 0 and c != "lp__":
            worst = max(worst, float(np.sqrt(((n - 1) / n * W + B / n) / W)))
    return out, worst


def bayes_fit(fit_dir, v: str, Kc: int = K, chains=None) -> dict:
    """A fitted K-class mixture on the health contract, read at its posterior mean.

    Returns the class shares, the quadratic coefficients on the standardised channel
    (a = (age - 55)/10), the residual scale, the contract moments for mapping back to
    the variant's units, each person's class posterior (``post``), the posterior class
    composition of the person-waves observed at each age (``comp``), and for
    ar_mode 2 the persistence and "spike" scale. Works for any K. ``chains`` restricts
    the posterior mean to those CmdStan chains (1-based) when the others sat in a
    separate mode; ``rhat`` is then the split R-hat over that subset.
    """
    fit_dir = Path(fit_dir)
    d = json.load(open(fit_dir / "stan_data.json"))
    summ = json.load(open(fit_dir / "run_summary.json"))
    p, rhat = summ["params"], summ["max_structural_rhat"]
    if chains is not None:  # posterior mean over the chains that share a mode
        p, rhat = _chain_means(fit_dir, chains)
    assert d["K"] == Kc, f"{fit_dir.name} has K = {d['K']}, not {Kc}"
    theta = np.array([p[f"theta[{k}]"]["mean"] for k in range(1, Kc + 1)])
    coef = np.array([[p[f"coef[1,{k},{j}]"]["mean"] for j in (1, 2, 3)] for k in range(1, Kc + 1)])
    sigma = p["sigma[1,1]"]["mean"]
    X, Y = np.asarray(d["X"]), np.asarray(d["y"][0])
    fs, fe = np.asarray(d["fit_start"]), np.asarray(d["fit_end"])
    long = pd.read_csv(CONTRACT / "long.csv").sort_values(["pidp", "age"]).reset_index(drop=True)
    mom = json.load(open(CONTRACT / "manifest.json"))["metric_moments"][v]
    z = (long[v].to_numpy() - mom["mean"]) / mom["sd"]
    assert len(long) == len(Y) and np.abs(z - Y).max() < 1e-6, "payload rows do not match the contract"
    person = np.repeat(np.arange(len(fs)), fe - fs + 1)
    assert (np.diff(long["pidp"].to_numpy()) != 0).sum() + 1 == len(fs)
    if d["ar_mode"] == 2:
        rho = np.array([p[f"rho[{k}]"]["mean"] for k in range(1, Kc + 1)])
        sigma_meas = p["sigma_meas[1]"]["mean"]
        per = kalman_person_lp(Y, X, coef, sigma, sigma_meas, rho, np.asarray(d["age_gap"], float), fs, fe)
        extra = {"rho": rho, "sigma_meas": sigma_meas}
    else:
        mu = X @ coef.T
        lp = -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((Y[:, None] - mu) / sigma) ** 2
        per = np.zeros((len(fs), Kc)); np.add.at(per, person, lp)
        extra = {"rho": np.zeros(Kc), "sigma_meas": 0.0}
    un = np.log(theta)[None, :] + per
    w = np.exp(un - un.max(axis=1, keepdims=True)); w /= w.sum(axis=1, keepdims=True)
    pids = long.groupby("pidp", sort=False)["pidp"].first().to_numpy()
    cols = [f"class{k + 1}" for k in range(Kc)]
    post = pd.DataFrame(w, columns=cols).assign(pidp=pids)
    comp = pd.DataFrame(w[person], columns=cols).assign(age=long["age"].to_numpy()).groupby("age").mean()
    return {"theta": theta, "coef": coef, "sigma": sigma, "mom": mom, "post": post, "comp": comp,
            "rhat": rhat, "chains": chains, "K": Kc, **extra}
