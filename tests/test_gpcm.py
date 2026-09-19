"""Data-free tests of the GPCM estimator: gradient, recovery, scoring identity,
and the exact weighted-sum property."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime

from prevention_health_clustering.measures.gpcm import (
    _nll_grad,
    cat_probs,
    fit_gpcm,
    score_eap,
    weighted_sum,
)

NAMES = ["GH", "PF", "RP", "LIM"]
NCATS = [5, 5, 9, 3]
A0 = [1.2, 2.0, 1.6, 0.9]
B0 = [np.array([-1.4, -0.5, 0.3, 1.2]),
      np.array([-1.2, -0.9, 0.1, 0.4]),
      np.array([-2.0, -1.4, -1.0, -0.6, -0.2, 0.2, 0.9, 0.5]),   # one reversal
      np.array([-2.2, -1.5])]


def simulate(n=40000, seed=5):
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=n)
    Y = np.zeros((n, len(NAMES)), dtype=int)
    for j in range(len(NAMES)):
        P = cat_probs(A0[j], B0[j], theta)
        Y[:, j] = 1 + (P.cumsum(axis=1) < rng.random(n)[:, None]).sum(axis=1)
    return Y, theta


def test_analytic_gradient_matches_numeric():
    rng = np.random.default_rng(0)
    th = np.linspace(-6, 6, 121)
    r = rng.random((5, 121)) * 50
    p = np.r_[np.log(1.4), -1.0, -0.2, 0.5, 1.1]
    _, g = _nll_grad(p, r, th)
    num = approx_fprime(p, lambda q: _nll_grad(q, r, th)[0], 1e-6)
    assert np.allclose(g, num, rtol=1e-4, atol=1e-3), (g, num)


def test_recovery_identity_and_weighted_sum():
    Y, theta = simulate()
    df = pd.DataFrame(Y, columns=NAMES)
    pat = df.value_counts().reset_index(name="N")
    fit = fit_gpcm(pat[NAMES].to_numpy(), pat["N"].to_numpy(float), NCATS, NAMES)
    for j, nm in enumerate(NAMES):
        a, b = fit.items[nm]
        assert abs(a - A0[j]) < 0.08, (nm, a, A0[j])
        assert np.abs(b - B0[j]).max() < 0.15, (nm, b, B0[j])
    eap, psd = score_eap(fit, pat[NAMES].to_numpy())
    pat["eap"], pat["psd"] = eap, psd
    m = df.merge(pat, on=NAMES, how="left")
    assert np.corrcoef(m["eap"], theta)[0, 1] > 0.85
    identity = m["eap"].var(ddof=0) + (m["psd"] ** 2).mean()
    assert abs(identity - 1.0) < 0.02, identity
    # the posterior depends on the pattern only through T = sum a_j (y_j - 1)
    T = weighted_sum(fit, pat[NAMES].to_numpy())
    order = np.argsort(T, kind="stable")
    steps = np.diff(eap[order])
    tied = np.isclose(np.diff(T[order]), 0.0)
    assert (steps[~tied] > -1e-9).all()
