"""Data-free tests of the GRM estimator: recodes, recovery, scoring identity.

The licensed-data replication (parameters, theta, and age distributions vs the
EIT note's grm.rds fit) lives in data_cleaning/04_build_grm.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prevention_health_clustering.measures.grm import (
    build_testlets,
    cat_probs,
    fit_grm,
    grmh_from_theta,
    score_eap,
    tcc,
)

NAMES = ["GH", "PF", "RP", "BP"]
NCATS = [5, 5, 9, 5]


def test_build_testlets_recodes():
    d = pd.DataFrame({
        "pidp": [1, 2, 3, 4],
        "wave": [1, 1, 1, 1],
        "age": [50, 50, 19, 50],
        # person 1 best health on raw items, person 2 worst, person 3 age-out,
        # person 4 has a missing item
        "sf1":  [1, 5, 1, 1], "sf2a": [3, 1, 3, 3], "sf2b": [3, 1, 3, 3],
        "sf3a": [5, 1, 5, 5], "sf3b": [5, 1, 5, 5],
        "sf5":  [1, 5, 1, np.nan],
    })
    out = build_testlets(d)
    assert list(out["pidp"]) == [1, 2]          # 3 out of age range, 4 incomplete
    best, worst = out.iloc[0], out.iloc[1]
    # sf1/sf5 are reversed: raw 1 (excellent / no pain) -> top category
    assert (best[["GH", "PF", "RP", "BP"]] == [5, 5, 9, 5]).all()
    assert (worst[["GH", "PF", "RP", "BP"]] == [1, 1, 1, 1]).all()


def simulate(n=30000, seed=11):
    rng = np.random.default_rng(seed)
    a0 = [1.7, 2.6, 3.1, 1.5]
    b0 = [np.array([-1.6, -0.6, 0.4, 1.5]),
          np.array([-1.5, -0.8, -0.1, 0.7]),
          np.array([-1.9, -1.5, -1.1, -0.7, -0.3, 0.1, 0.5, 0.9]),
          np.array([-1.8, -1.0, -0.2, 0.9])]
    theta = rng.normal(size=n)
    Y = np.zeros((n, 4), dtype=int)
    for j in range(4):
        P = cat_probs(a0[j], b0[j], theta)
        Y[:, j] = 1 + (P.cumsum(axis=1) < rng.random(n)[:, None]).sum(axis=1)
    return Y, theta, a0, b0


def test_parameter_recovery_and_eap_identity():
    Y, theta, a0, b0 = simulate()
    df = pd.DataFrame(Y, columns=NAMES)
    pat = df.value_counts().reset_index(name="N")
    fit = fit_grm(pat[NAMES].to_numpy(), pat["N"].to_numpy(float), NCATS, NAMES)
    for j, nm in enumerate(NAMES):
        a, b = fit.items[nm]
        assert abs(a - a0[j]) < 0.08, (nm, a, a0[j])
        assert np.abs(b - b0[j]).max() < 0.05, (nm, b, b0[j])
    eap, psd = score_eap(fit, pat[NAMES].to_numpy())
    pat["eap"], pat["psd"] = eap, psd
    m = df.merge(pat, on=NAMES, how="left")
    assert np.corrcoef(m["eap"], theta)[0, 1] > 0.90
    # law of total variance under a correct model: var(EAP) + E[post var] = 1
    identity = m["eap"].var(ddof=0) + (m["psd"] ** 2).mean()
    assert abs(identity - 1.0) < 0.02, identity


def test_multigroup_recovers_age_gradient():
    rng = np.random.default_rng(7)
    n = 40000
    grp = rng.integers(0, 4, size=n)          # four "ages"
    true_mu = np.array([0.6, 0.2, -0.2, -0.6])
    theta = true_mu[grp] + rng.normal(0, 0.8, size=n)
    theta = (theta - theta.mean()) / theta.std()   # pooled N(0,1) as identified
    a0 = [1.7, 2.6, 3.1, 1.5]
    b0 = [np.array([-1.6, -0.6, 0.4, 1.5]),
          np.array([-1.5, -0.8, -0.1, 0.7]),
          np.array([-1.9, -1.5, -1.1, -0.7, -0.3, 0.1, 0.5, 0.9]),
          np.array([-1.8, -1.0, -0.2, 0.9])]
    Y = np.zeros((n, 4), dtype=int)
    for j in range(4):
        P = cat_probs(a0[j], b0[j], theta)
        Y[:, j] = 1 + (P.cumsum(axis=1) < rng.random(n)[:, None]).sum(axis=1)
    df = pd.DataFrame(Y, columns=NAMES)
    df["g"] = grp
    pat = df.value_counts().reset_index(name="N")
    fit = fit_grm(pat[NAMES].to_numpy(), pat["N"].to_numpy(float), NCATS,
                  NAMES, grp=pat["g"].to_numpy())
    got = fit.mu
    # Direction and ordering are the testable content; the scale is set by the
    # pooled-N(0,1) identification, not by the simulation's raw mu values.
    assert (np.diff(got) < 0).all(), got       # monotone decreasing across groups
    assert np.corrcoef(got, true_mu)[0, 1] > 0.99


def test_tcc_and_grmh_monotone():
    Y, theta, a0, b0 = simulate(n=5000)
    pat = pd.DataFrame(Y, columns=NAMES).value_counts().reset_index(name="N")
    fit = fit_grm(pat[NAMES].to_numpy(), pat["N"].to_numpy(float), NCATS, NAMES)
    curve = tcc(fit, fit.th)
    assert (np.diff(curve) > 0).all()
    assert 0.0 <= curve.min() and curve.max() <= 1.0
    g = grmh_from_theta(fit, np.array([-2.0, 0.0, 2.0]))
    assert (np.diff(g) > 0).all()


if __name__ == "__main__":
    import sys, traceback

    failures = 0
    for name, fn in sorted(
        (k, v) for k, v in globals().items()
        if k.startswith("test_") and callable(v)
    ):
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failures += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print("GRM TESTS OK" if failures == 0 else f"GRM TESTS FAILED ({failures})")
    sys.exit(1 if failures else 0)
