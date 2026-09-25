"""The covariance-moment machinery behind Exhibits 1 and 2 and the framework's
empirical figures, self-contained in this repository.

Ports the companion pipeline (redo_concepts/emp_00, emp_25/27, 08_limitation_
moments.R) to the project's measure:

  ident_panel      the identification panel: one row per person-age (earliest
                   wave kept), implausible age sequences dropped, ages 20-90,
                   people with at least four observed ages (the companion's s4)
  wide             people x ages matrix of a variant
  pooled_moments   the lower triangle of the covariance matrix of the ages
                   a0 + offsets, pooled over base ages a0 in [lo, hi] with
                   count weights; 'balanced' uses people observed at all
                   offsets, 'pairwise' every pair's own people
  fit_case         Cases 1 to 4 (and the denoising solve) on pooled
                   moments, rho profiled on a grid, the rest by least squares;
                   returns the best fit and the best admissible fit
  row_cells        Cov(h_s, h_{s+k}), k = 0..K, for every anchor s
  fit_rows_bundle  the bundle fit per anchor: level + k + k^2 + one-period
                   spike, no decay column (the rho -> 1 corner)
  fit_rows_linear  line + geometric decay per anchor, rho profiled (row-average
                   local covariance)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prevention_health_clustering.config import PROCESSED_DATA_DIR

MIN_AGE, MAX_AGE = 20, 90
AGES = np.arange(MIN_AGE, MAX_AGE + 1)
RHO_GRID = {"case1": np.array([0.0]), "case2": np.arange(0.05, 0.975, 0.02), "case3": np.arange(0.05, 0.975, 0.02),
            "case4": np.arange(0.30, 0.975, 0.01)}
BANDS = {"25-40": (25, 40), "40-60": (40, 60), "60-75": (60, 75), "75-90": (75, 90)}
OFFS4 = np.array([0, 2, 4, 6])


def ident_panel(variants=("theta", "h", "fi10"), min_obs: int = 4) -> pd.DataFrame:
    d = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "health_measure.parquet",
                        columns=["pidp", "wave", "age", *variants])
    d = d[d["age"].notna()].assign(age=lambda x: x["age"].astype(int)).sort_values(["pidp", "wave"])
    dage = d.groupby("pidp")["age"].diff()
    dwave = d.groupby("pidp")["wave"].diff()
    bad = d.loc[dage.notna() & ((dage < 0) | (dage > dwave + 1)), "pidp"].unique()
    d = d[~d["pidp"].isin(bad)]
    d = d.sort_values(["pidp", "age", "wave"]).drop_duplicates(["pidp", "age"], keep="first")
    d = d[d["age"].between(MIN_AGE, MAX_AGE)]
    nobs = d.groupby("pidp")["age"].transform("size")
    return d[nobs >= min_obs].reset_index(drop=True)


def wide(d: pd.DataFrame, v: str) -> np.ndarray:
    ppl = pd.Index(d["pidp"].unique())
    W = np.full((len(ppl), len(AGES)), np.nan)
    ok = d[v].notna()
    W[ppl.get_indexer(d.loc[ok, "pidp"]), d.loc[ok, "age"].to_numpy() - MIN_AGE] = d.loc[ok, v].to_numpy()
    return W


def pairs_of(offs):
    """Cell order: the diagonal first, then the first row, then the interior."""
    offs = np.asarray(offs)
    P = len(offs)
    i = [k for k in range(P)] + [0] * (P - 1) + [s for s in range(1, P) for t in range(s + 1, P)]
    j = [k for k in range(P)] + [k for k in range(1, P)] + [t for s in range(1, P) for t in range(s + 1, P)]
    return offs[i], offs[j]


def _cell_index(P: int):
    a, b = pairs_of(np.arange(P))
    return a, b


def _cov_lower(X: np.ndarray) -> np.ndarray:
    S = np.cov(X, rowvar=False, ddof=1)
    a, b = _cell_index(S.shape[0])
    return S[a, b]


def pooled_moments(W: np.ndarray, offs, lo: int, hi: int, how: str = "balanced", nmin: int = 150,
                   cell_min: int | None = None):
    """Count-weighted pooled lower-triangle moments over base ages lo..hi.

    A base age enters when at least nmin people are observed at all four ages.
    With how="pairwise" and cell_min set, that rule is replaced by a per-cell one:
    a base age enters when every cell rests on at least cell_min people observed
    at both of its ages, which is what pairwise moments actually use.
    """
    K = len(offs) * (len(offs) + 1) // 2
    num, den = np.zeros(K), np.zeros(K)
    for a0 in range(lo, hi + 1):
        cols = a0 + np.asarray(offs) - MIN_AGE
        if cols.max() >= W.shape[1]:
            continue
        X = W[:, cols]
        bal = ~np.isnan(X).any(axis=1)
        if how == "pairwise" and cell_min is not None:
            N = (~np.isnan(X)).astype(float)
            a, b = _cell_index(len(offs))
            if (N.T @ N)[a, b].min() < cell_min:
                continue
        elif bal.sum() < nmin:
            continue
        if how == "balanced":
            m, w = _cov_lower(X[bal]), np.full(K, bal.sum(), float)
        else:
            df = pd.DataFrame(X)
            S = df.cov(min_periods=2).to_numpy()
            N = (~np.isnan(X)).astype(float)
            a, b = _cell_index(len(offs))
            m, w = S[a, b], (N.T @ N)[a, b]
        if np.isnan(m).any():
            continue
        num += w * m
        den += w
    if den.min() == 0:
        return None, 0.0
    return num / den, den.max()


def design(case: str, i, j, rho: float, offs=None):
    sm = np.column_stack([np.ones(len(i)), i + j, i * j])
    if case == "case1":                       # i.i.d. noise, one variance: on the diagonal only
        return np.column_stack([sm, (i == j).astype(float)])
    if case == "case2":
        return np.column_stack([sm, rho ** np.abs(i - j)])
    if case == "case3":
        offs = np.unique(np.r_[i, j]) if offs is None else np.asarray(offs)
        return np.column_stack([sm] + [rho ** (j - i) * (i == k) for k in offs])
    if case == "case4":
        return np.column_stack([sm, rho ** np.abs(i - j), (i == j).astype(float)])
    raise ValueError(case)


def admissible(b) -> bool:
    return b[0] >= 0 and b[2] >= 0 and b[1] ** 2 <= b[0] * b[2] * (1 + 1e-6) and (b[3:] >= -1e-8).all()


def fit_case(m: np.ndarray, case: str, i, j, offs=None, rho_grid=None):
    best, best_ok = None, None
    for rho in (RHO_GRID[case] if rho_grid is None else rho_grid):
        X = design(case, i, j, rho, offs)
        b, *_ = np.linalg.lstsq(X, m, rcond=None)
        ss = float(((m - X @ b) ** 2).sum())
        rec = {"b": b, "ss": ss, "rho": float(rho), "X": X, "fit": X @ b}
        if best is None or ss < best["ss"]:
            best = rec
        if admissible(b) and (best_ok is None or ss < best_ok["ss"]):
            best_ok = rec
    return best, best_ok


def _constrained_at(m: np.ndarray, X: np.ndarray, b0: np.ndarray) -> tuple[np.ndarray, float]:
    """Least squares at a fixed rho with the admissibility constraints imposed.

    V_H, V_d and every noise variance non-negative and C_Hd^2 <= V_H V_d. At a
    fixed rho this is a convex problem (a quadratic objective over a convex
    cone), so SLSQP from a feasible start finds its optimum.
    """
    from scipy.optimize import minimize
    sc = max(np.abs(m).max(), 1e-12)
    mm, k = m / sc, X.shape[1]
    start = b0 / sc
    start[[0, 2]] = np.maximum(start[[0, 2]], 1e-6)
    start[3:] = np.maximum(start[3:], 0.0)
    lim = np.sqrt(start[0] * start[2])
    start[1] = np.clip(start[1], -lim, lim)
    r = minimize(lambda b: ((mm - X @ b) ** 2).sum(), start, jac=lambda b: -2 * X.T @ (mm - X @ b),
                 method="SLSQP", bounds=[(0, None), (None, None), (0, None)] + [(0, None)] * (k - 3),
                 constraints=[{"type": "ineq", "fun": lambda b: b[0] * b[2] - b[1] ** 2,
                               "jac": lambda b: np.r_[b[2], -2 * b[1], b[0], np.zeros(k - 3)]}],
                 options={"maxiter": 500, "ftol": 1e-14})
    b = r.x * sc
    return b, float(((m - X @ b) ** 2).sum())


def binds(b, tol: float = 0.995, vh_rel: float = 1e-3) -> bool:
    """Whether a constrained fit sits on the admissibility boundary: |Corr| at 1,
    V_H pushed to (near) zero relative to the fitted moments, or V_d at zero.
    With V_H at zero the implied correlation is not identified at all."""
    c = corr_of(b)
    scale = max(abs(b[0]) + np.abs(b[3:]).sum(), 1e-12)
    return bool((np.isfinite(c) and abs(c) > tol) or b[0] <= vh_rel * scale or b[2] <= 1e-12)


def fit_constrained(m: np.ndarray, case: str, i, j, offs=None, rho_grid=None) -> dict:
    """The properly constrained fit: at every rho the admissible least-squares
    solution (the unconstrained one where that is already admissible), then the
    best rho. Also returns the unconstrained best, to report what the constraint
    costs."""
    best, unc = None, None
    for rho in (RHO_GRID[case] if rho_grid is None else rho_grid):
        X = design(case, i, j, rho, offs)
        b, *_ = np.linalg.lstsq(X, m, rcond=None)
        ss_u = float(((m - X @ b) ** 2).sum())
        if unc is None or ss_u < unc["ss"]:
            unc = {"b": b, "ss": ss_u, "rho": float(rho), "X": X, "fit": X @ b}
        bc, ss = (b, ss_u) if admissible(b) else _constrained_at(m, X, b)
        if best is None or ss < best["ss"]:
            best = {"b": bc, "ss": ss, "rho": float(rho), "X": X, "fit": X @ bc}
    best["binds"] = binds(best["b"]) and not admissible(unc["b"])
    best["unconstrained"] = unc
    best["cost"] = best["ss"] / unc["ss"] - 1 if unc["ss"] > 0 else 0.0
    return best


def fit_common(ms: dict, case: str, i, j, offs=None, rho_grid=None) -> dict:
    """Constrained fits with one rho shared by every band: the rho that minimises
    the summed squared error, each band keeping its own V_H, C_Hd, V_d and noise."""
    grid = RHO_GRID[case] if rho_grid is None else rho_grid
    per = {bn: [] for bn in ms}
    for rho in grid:
        X = design(case, i, j, rho, offs)
        for bn, m in ms.items():
            b, *_ = np.linalg.lstsq(X, m, rcond=None)
            bc, ss = (b, float(((m - X @ b) ** 2).sum())) if admissible(b) else _constrained_at(m, X, b)
            per[bn].append({"b": bc, "ss": ss, "rho": float(rho), "X": X, "fit": X @ bc,
                            "admissible_unconstrained": admissible(b)})
    k = int(np.argmin([sum(per[bn][r]["ss"] for bn in ms) for r in range(len(grid))]))
    out = {}
    for bn in ms:
        rec = per[bn][k]
        rec["binds"] = binds(rec["b"]) and not rec["admissible_unconstrained"]
        out[bn] = rec
    return out


def denoising_solve(m, i, j):
    """Independent noise, free v_a: the deterministic block from off-diagonal cells only."""
    off = i != j
    X = np.column_stack([np.ones(off.sum()), (i + j)[off], (i * j)[off]])
    b, *_ = np.linalg.lstsq(X, m[off], rcond=None)
    return b


def corr_of(b):
    return b[1] / np.sqrt(b[0] * b[2]) if b[0] * b[2] > 0 else np.nan


def layers(case: str, b, X, i, j):
    """Fitted layers of every cell: V_H, (s+t)C_Hd, st V_d, carried noise, spike."""
    out = {"VH": np.full(len(i), b[0]), "CHd": (i + j) * b[1], "Vd": i * j * b[2]}
    if case == "case4":
        out["noise"] = X[:, 3] * b[3]
        out["spike"] = X[:, 4] * b[4]
    elif case == "case1":
        out["noise"] = np.zeros(len(i))
        out["spike"] = X[:, 3] * b[3]
    else:
        out["noise"] = X[:, 3:] @ b[3:]
        out["spike"] = np.zeros(len(i))
    return out


def cell_names(i, j, step: int):
    return [f"V({a // step})" if a == b else f"C({a // step},{b // step})" for a, b in zip(i, j)]


# ---- rows ------------------------------------------------------------------

def row_cells(W: np.ndarray, K: int, nmin: int = 100) -> np.ndarray:
    """anchors x (K+1): Cov(h_s, h_{s+k}) over people with both, NaN if n < nmin."""
    A = W.shape[1] - K
    out = np.full((A, K + 1), np.nan)
    for s in range(A):
        x = W[:, s]
        for k in range(K + 1):
            y = W[:, s + k]
            ok = ~np.isnan(x) & ~np.isnan(y)
            if ok.sum() >= nmin:
                out[s, k] = np.cov(x[ok], y[ok], ddof=1)[0, 1]
    return out


def fit_rows_bundle(cells: np.ndarray, K: int) -> pd.DataFrame:
    k = np.arange(K + 1)
    X = np.column_stack([np.ones(K + 1), k, k ** 2, (k == 0).astype(float)])
    P = np.linalg.solve(X.T @ X, X.T)
    ok = ~np.isnan(cells).any(axis=1)
    out = np.full((cells.shape[0], 4), np.nan)
    out[ok] = cells[ok] @ P.T
    return pd.DataFrame(out, columns=["L", "CHd", "curv", "spike"])


def fit_rows_linear(cells: np.ndarray, K: int, rho_grid=np.linspace(0.30, 0.97, 47)) -> pd.DataFrame:
    k = np.arange(K + 1)
    ok = ~np.isnan(cells).any(axis=1)
    best_ss = np.full(cells.shape[0], np.inf)
    best = np.full((cells.shape[0], 4), np.nan)
    for rho in rho_grid:
        X = np.column_stack([np.ones(K + 1), k, rho ** k])
        P = np.linalg.solve(X.T @ X, X.T)
        cf = cells[ok] @ P.T
        rs = ((cells[ok] - cf @ X.T) ** 2).sum(axis=1)
        valid = (cf[:, 2] >= 0) & (cf[:, 2] <= cells[ok, 0])
        rs[~valid] = np.inf
        idx = np.flatnonzero(ok)
        hit = rs < best_ss[idx]
        best_ss[idx[hit]] = rs[hit]
        best[idx[hit]] = np.column_stack([cf[hit], np.full(hit.sum(), rho)])
    return pd.DataFrame(best, columns=["L", "CHd", "v", "rho"])
