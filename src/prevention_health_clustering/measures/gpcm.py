"""Generalised partial credit measurement on the same testlet banks as the GRM.

MODEL. Muraki's GPCM over categories k = 0..K-1 (data category k + 1, so 1 is
still worst): P(y_j = k | theta) is proportional to
exp(sum_{v <= k} a_j (theta - b_jv)). The step parameters b_jv are free and may
come out unordered. The likelihood of a response pattern depends on theta only
through T = sum_j a_j (y_j - 1), so under one common prior the EAP is a
monotone function of that weighted sum: the score is exactly a fixed curve
applied to a weighted sum, with weight a_j per category step.

ESTIMATION mirrors measures.grm line for line: marginal ML by EM over response
patterns with counts, a 121-point grid on [-6, 6], the pooled latent N(0, 1),
optional multigroup latent distributions rescaled so the pooled latent stays
N(0, 1). The item M step uses analytic gradients.

SCORES. ``theta`` is the EAP under the pooled prior; ``h`` pushes theta
through the expected-score curve onto [0, 1], exactly as grmh does for the GRM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm

from prevention_health_clustering.measures.grm import DEFAULT_NQ, DEFAULT_QLIM


def log_cat_probs(a: float, b: np.ndarray, th: np.ndarray) -> np.ndarray:
    """(len(th), K) log category probabilities, exact (log-softmax).

    Unlike the GRM's differences of logistics, adjacent-category probabilities
    never underflow to a negative or zero difference, so no floor is needed,
    and the M-step objective stays consistent with its analytic gradient."""
    k = len(b) + 1
    steps = np.r_[0.0, np.cumsum(b)]
    z = a * (np.arange(k)[None, :] * th[:, None] - steps[None, :])
    return z - logsumexp(z, axis=1, keepdims=True)


def cat_probs(a: float, b: np.ndarray, th: np.ndarray) -> np.ndarray:
    """(len(th), K) category probabilities."""
    return np.exp(log_cat_probs(a, b, th))


def _start_par(y: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    s = np.bincount(y - 1, weights=w, minlength=k) + 0.5
    return np.r_[0.0, -np.log(s[1:] / s[:-1])]


def _nll_grad(p: np.ndarray, r: np.ndarray, th: np.ndarray) -> tuple[float, np.ndarray]:
    """Expected complete-data negative log-likelihood of one item and its
    gradient in (log a, b). r is (K, nq) expected counts."""
    k = r.shape[0]
    a, b = float(np.exp(p[0])), p[1:]
    logp = log_cat_probs(a, b, th)
    P = np.exp(logp)
    s = np.arange(k)[None, :] * th[:, None] - np.r_[0.0, np.cumsum(b)][None, :]
    R = r.T
    N = R.sum(axis=1)
    nll = -float(np.sum(R * logp))
    g_a = -float(np.sum((R * s).sum(axis=1) - N * (P * s).sum(axis=1)))
    r_ge = np.cumsum(R[:, ::-1], axis=1)[:, ::-1][:, 1:]
    p_ge = np.cumsum(P[:, ::-1], axis=1)[:, ::-1][:, 1:]
    g_b = a * (r_ge - N[:, None] * p_ge).sum(axis=0)
    return nll, np.r_[g_a * a, g_b]


@dataclass
class GPCMFit:
    items: dict[str, tuple[float, np.ndarray]]   # name -> (a, step parameters)
    ncat: dict[str, int]
    th: np.ndarray
    log_likelihood: float
    mu: np.ndarray
    sigma: np.ndarray
    n_iter: int
    n_parameters: int


def fit_gpcm(
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
) -> GPCMFit:
    """MML-EM over response patterns, structured exactly as grm.fit_grm."""
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
            a, b = float(np.exp(par[j][0])), par[j][1:]
            logP += log_cat_probs(a, b, th)[:, Y[:, j] - 1].T
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
            yj = Y[:, j] - 1
            for cat in range(k):
                mask = yj == cat
                if mask.any():
                    r[cat] = Wn[mask].sum(axis=0)
            par[j] = minimize(_nll_grad, par[j], args=(r, th), jac=True,
                              method="BFGS", options={"maxiter": 200}).x
        # ---- M step: group latent distributions ---------------------------
        if G > 1:
            gs = np.zeros((G, nq))
            for g in range(G):
                mask = grp == g
                if mask.any():
                    gs[g] = Wn[mask].sum(axis=0)
            n_g = gs.sum(axis=1)
            mu = gs @ th / n_g
            sg = np.sqrt(np.maximum(gs @ th**2 / n_g - mu**2, 1e-6))
            m0 = float(np.sum(n_g * mu) / n_g.sum())
            v0 = float(np.sum(n_g * (sg**2 + (mu - m0) ** 2)) / n_g.sum())
            mu = (mu - m0) / np.sqrt(v0)
            sg = sg / np.sqrt(v0)
            for j in range(J):
                a, b = float(np.exp(par[j][0])), par[j][1:]
                par[j] = np.r_[np.log(a * np.sqrt(v0)), (b - m0) / np.sqrt(v0)]

    items = {names[j]: (float(np.exp(par[j][0])), par[j][1:].copy()) for j in range(J)}
    npar = sum(len(p) for p in par) + (2 * (G - 1) if G > 1 else 0)
    return GPCMFit(items=items, ncat=dict(zip(names, ncat)), th=th,
                   log_likelihood=ll, mu=mu, sigma=sg, n_iter=it,
                   n_parameters=npar)


def score_eap(
    fit: GPCMFit, Y: np.ndarray, *, mu: float = 0.0, sg: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """EAP theta and posterior sd per response pattern, pooled prior."""
    th = fit.th
    logP = np.zeros((len(Y), len(th)))
    for j, name in enumerate(fit.items):
        a, b = fit.items[name]
        logP += log_cat_probs(a, b, th)[:, Y[:, j] - 1].T
    p = norm.pdf(th, mu, sg)
    lw = logP + np.log(p / p.sum())
    W = np.exp(lw - lw.max(axis=1, keepdims=True))
    W /= W.sum(axis=1, keepdims=True)
    eap = W @ th
    sd = np.sqrt(np.maximum(W @ th**2 - eap**2, 0.0))
    return eap, sd


def weighted_sum(fit: GPCMFit, Y: np.ndarray) -> np.ndarray:
    """The sufficient statistic T = sum_j a_j (y_j - 1)."""
    a = np.array([fit.items[name][0] for name in fit.items])
    return (Y - 1) @ a


def test_information(fit: GPCMFit, th: np.ndarray) -> np.ndarray:
    """(len(th), J) Fisher information by item: a^2 Var(k | theta)."""
    cols = []
    for name in fit.items:
        a, b = fit.items[name]
        pr = cat_probs(a, b, th)
        k = np.arange(pr.shape[1])
        m = pr @ k
        cols.append(a**2 * (pr @ k**2 - m**2))
    return np.column_stack(cols)


def tcc(fit: GPCMFit, th: np.ndarray) -> np.ndarray:
    """Expected share of the maximum score, on [0, 1]."""
    num, den = np.zeros(len(th)), 0
    for name in fit.items:
        a, b = fit.items[name]
        k = fit.ncat[name]
        num += cat_probs(a, b, th) @ np.arange(k)
        den += k - 1
    return num / den


def h_from_theta(fit: GPCMFit, theta: np.ndarray) -> np.ndarray:
    """theta -> [0, 1] through the expected-score curve (monotone)."""
    return np.interp(theta, fit.th, tcc(fit, fit.th))


def yens_q3(
    fit: GPCMFit, Y: np.ndarray, weights: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """Weighted residual correlations, as grm.yens_q3."""
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
    "GPCMFit",
    "cat_probs",
    "fit_gpcm",
    "h_from_theta",
    "log_cat_probs",
    "score_eap",
    "tcc",
    "test_information",
    "weighted_sum",
    "yens_q3",
]
