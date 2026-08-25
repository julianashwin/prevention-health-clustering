"""Samejima graded-response measurement of physical health.

Port of the own-code estimator in AnalysisForEIT emp_09_grm.R (the EIT
empirics note, sections 10-11), validated against its August 2026 fit.

ITEMS. The six physical SF-12 items, recoded so 1 is worst, then summed into
four testlets to remove the within-pair local dependence the note documents
(Yen's Q3 +0.38 in the PF pair; sf3b's discrimination ran to 7.7 fitted raw):

    GH = sf1 (reversed, 5 levels)      PF = sf2a + sf2b - 1  (5 levels)
    BP = sf5 (reversed, 5 levels)      RP = sf3a + sf3b - 1  (9 levels)

MODEL. Unidimensional logistic GRM: P(y_j >= k | theta) =
logistic(a_j (theta - b_jk)). Marginal ML by EM over a 121-point quadrature
grid on [-6, 6], run over response PATTERNS with counts. Identification: the
pooled latent distribution is N(0, 1).

MULTI-GROUP. Item parameters common; each single year of age gets its own
N(mu_a, sigma_a^2), rescaled so the pooled latent stays N(0, 1). mu_a and
sigma_a are the age profiles of latent physical health, free of measurement
error and of EAP shrinkage.

SCORES. ``theta`` is the EAP under the pooled fit; ``grmh`` is theta pushed
through the test characteristic curve onto [0, 1] (1 = best attainable
response on every item) — a fixed monotone transform, identical ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

TESTLETS: tuple[str, ...] = ("GH", "PF", "RP", "BP")
NCAT: dict[str, int] = {"GH": 5, "PF": 5, "RP": 9, "BP": 5}
RAW_ITEMS: dict[str, int] = {"sf1": 5, "sf2a": 3, "sf2b": 3,
                             "sf3a": 5, "sf3b": 5, "sf5": 5}
REVERSED_ITEMS: tuple[str, ...] = ("sf1", "sf5")
DEFAULT_NQ = 121
DEFAULT_QLIM = 6.0
DEFAULT_AGE_RANGE = (20, 90)


def build_testlets(
    items: pd.DataFrame, age_range: tuple[int, int] = DEFAULT_AGE_RANGE
) -> pd.DataFrame:
    """Person-waves with the four testlets, complete cases, 1 = worst."""
    d = items.copy()
    d = d[d["age"].notna() & d["age"].between(*age_range)]
    for v, k in RAW_ITEMS.items():
        col = d[v].where(d[v].isin(range(1, k + 1)))
        d[v] = (k + 1 - col) if v in REVERSED_ITEMS else col
    d = d.dropna(subset=list(RAW_ITEMS))
    out = pd.DataFrame(
        {
            "pidp": d["pidp"].to_numpy(),
            "wave": d["wave"].to_numpy(),
            "age": d["age"].to_numpy(int),
            "GH": d["sf1"].to_numpy(int),
            "PF": (d["sf2a"] + d["sf2b"] - 1).to_numpy(int),
            "RP": (d["sf3a"] + d["sf3b"] - 1).to_numpy(int),
            "BP": d["sf5"].to_numpy(int),
        }
    )
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# model primitives (verbatim ports)
# ---------------------------------------------------------------------------

def _unpack(p: np.ndarray, k: int) -> tuple[float, np.ndarray]:
    return float(np.exp(p[0])), np.cumsum(np.r_[p[1], np.exp(p[2:k])])


def _pack(a: float, b: np.ndarray) -> np.ndarray:
    return np.r_[np.log(a), b[0], np.log(np.diff(b))]


def cat_probs(a: float, b: np.ndarray, th: np.ndarray) -> np.ndarray:
    """(len(th), K) category probabilities, floored at 1e-12."""
    k = len(b) + 1
    P = np.ones((len(th), k + 1))
    P[:, 1:k] = 1.0 / (1.0 + np.exp(-a * (th[:, None] - b[None, :])))
    P[:, k] = 0.0
    return np.maximum(P[:, :k] - P[:, 1:], 1e-12)


def _start_par(y: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    s = np.bincount(y - 1, weights=w, minlength=k)
    q = np.cumsum(s[::-1])[::-1][1:] / w.sum()
    z = norm.isf(np.clip(q, 1e-4, 1 - 1e-4))
    return _pack(1.0, np.sort(z))


@dataclass
class GRMFit:
    items: dict[str, tuple[float, np.ndarray]]   # name -> (a, thresholds)
    ncat: dict[str, int]
    th: np.ndarray
    log_likelihood: float
    mu: np.ndarray                                # per group (scalar 0 pooled)
    sigma: np.ndarray
    n_iter: int
    n_parameters: int


def fit_grm(
    Y: np.ndarray,
    weights: np.ndarray,
    ncat: list[int],
    names: list[str],
    grp: np.ndarray | None = None,
    *,
    tol: float = 1e-8,
    maxit: int = 400,
    nq: int = DEFAULT_NQ,
    qlim: float = DEFAULT_QLIM,
    verbose: bool = False,
) -> GRMFit:
    """MML-EM over response patterns; mirrors the R estimator exactly."""
    n, J = Y.shape
    th = np.linspace(-qlim, qlim, nq)
    if grp is None:
        grp = np.zeros(n, dtype=int)
    G = int(grp.max()) + 1
    mu, sg = np.zeros(G), np.ones(G)
    par = [_start_par(Y[:, j], weights, ncat[j]) for j in range(J)]

    ll_old, ll = -np.inf, -np.inf
    for it in range(1, maxit + 1):
        # ---- E step -------------------------------------------------------
        logP = np.zeros((n, nq))
        for j in range(J):
            a, b = _unpack(par[j], ncat[j])
            logP += np.log(cat_probs(a, b, th))[:, Y[:, j] - 1].T
        pri = norm.pdf(th[None, :], mu[:, None], sg[:, None])
        lpri = np.log(pri / pri.sum(axis=1, keepdims=True))
        lw = logP + lpri[grp]
        mx = lw.max(axis=1)
        W = np.exp(lw - mx[:, None])
        rs = W.sum(axis=1)
        ll = float(np.sum(weights * (np.log(rs) + mx)))
        W /= rs[:, None]
        Wn = W * weights[:, None]

        if verbose and (it % 25 == 0 or it == 1):
            print(f"  EM {it:3d}  logL {ll:.3f}")
        if abs(ll - ll_old) < tol * abs(ll):
            break
        ll_old = ll

        # ---- M step: items ------------------------------------------------
        for j in range(J):
            k = ncat[j]
            r = np.zeros((k, nq))
            np.add.at(r, Y[:, j] - 1, Wn)

            def nll(p, k=k, r=r):
                a, b = _unpack(p, k)
                return -float(np.sum(r.T * np.log(cat_probs(a, b, th))))

            par[j] = minimize(nll, par[j], method="BFGS",
                              options={"maxiter": 200}).x
        # ---- M step: group latent distributions ---------------------------
        if G > 1:
            gs = np.zeros((G, nq))
            np.add.at(gs, grp, Wn)
            n_g = gs.sum(axis=1)
            mu = gs @ th / n_g
            sg = np.sqrt(np.maximum(gs @ th**2 / n_g - mu**2, 1e-6))
            m0 = float(np.sum(n_g * mu) / n_g.sum())
            v0 = float(np.sum(n_g * (sg**2 + (mu - m0) ** 2)) / n_g.sum())
            mu = (mu - m0) / np.sqrt(v0)
            sg = sg / np.sqrt(v0)
            for j in range(J):
                a, b = _unpack(par[j], ncat[j])
                par[j] = _pack(a * np.sqrt(v0), (b - m0) / np.sqrt(v0))

    items = {names[j]: _unpack(par[j], ncat[j]) for j in range(J)}
    npar = sum(len(p) for p in par) + (2 * (G - 1) if G > 1 else 0)
    return GRMFit(items=items, ncat=dict(zip(names, ncat)), th=th,
                  log_likelihood=ll, mu=mu, sigma=sg, n_iter=it,
                  n_parameters=npar)


def score_eap(
    fit: GRMFit, Y: np.ndarray, *, mu: float = 0.0, sg: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """EAP theta and posterior sd per response pattern, pooled-prior."""
    th = fit.th
    logP = np.zeros((len(Y), len(th)))
    for j, name in enumerate(fit.items):
        a, b = fit.items[name]
        logP += np.log(cat_probs(a, b, th))[:, Y[:, j] - 1].T
    p = norm.pdf(th, mu, sg)
    lw = logP + np.log(p / p.sum())
    W = np.exp(lw - lw.max(axis=1, keepdims=True))
    W /= W.sum(axis=1, keepdims=True)
    eap = W @ th
    sd = np.sqrt(np.maximum(W @ th**2 - eap**2, 0.0))
    return eap, sd


def test_information(fit: GRMFit, th: np.ndarray) -> np.ndarray:
    """(len(th), J) Fisher information by item, forward-difference as in R."""
    eps = 1e-4
    cols = []
    for name in fit.items:
        a, b = fit.items[name]
        pr = cat_probs(a, b, th)
        p2 = cat_probs(a, b, th + eps)
        cols.append((((p2 - pr) / eps) ** 2 / pr).sum(axis=1))
    return np.column_stack(cols)


def tcc(fit: GRMFit, th: np.ndarray) -> np.ndarray:
    """Test characteristic curve on [0, 1]: expected share of the max score."""
    num, den = np.zeros(len(th)), 0
    for name in fit.items:
        a, b = fit.items[name]
        k = fit.ncat[name]
        num += cat_probs(a, b, th) @ np.arange(k)
        den += k - 1
    return num / den


def grmh_from_theta(fit: GRMFit, theta: np.ndarray) -> np.ndarray:
    """theta -> [0, 1] through the TCC (monotone; ranking unchanged)."""
    return np.interp(theta, fit.th, tcc(fit, fit.th))


def yens_q3(
    fit: GRMFit, Y: np.ndarray, weights: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """Weighted residual correlations after the testlets (|Q3| > 0.2 flags)."""
    resid = np.zeros_like(Y, dtype=float)
    for j, name in enumerate(fit.items):
        a, b = fit.items[name]
        k = fit.ncat[name]
        expected = cat_probs(a, b, fit.th) @ np.arange(1, k + 1)
        resid[:, j] = Y[:, j] - np.interp(theta, fit.th, expected)
    w = weights / weights.sum()
    m = w @ resid
    c = (resid - m).T @ ((resid - m) * w[:, None])
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


__all__ = [
    "DEFAULT_AGE_RANGE",
    "GRMFit",
    "NCAT",
    "TESTLETS",
    "build_testlets",
    "cat_probs",
    "fit_grm",
    "grmh_from_theta",
    "score_eap",
    "tcc",
    "test_information",
    "yens_q3",
]
