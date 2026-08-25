"""Fast, data-free tests of the health-measure construction machinery.

Everything here runs on synthetic inputs: hand-computed subscale values, the
published SF-6D worked example, and a simulated two-factor population for the
UK-structure recovery. The licensed-data validation (reconstruction of
sf12pcs_dv to max error 0.0054, the UK norms table, phi = 0.897) lives in
data_cleaning/03_build_measures.py, which needs the raw extract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prevention_health_clustering.measures.composites import (
    equal_weight_composite,
    farivar_composites,
)
from prevention_health_clustering.measures.items import wave_item_columns
from prevention_health_clustering.measures.sf12 import (
    SCALES,
    US_COEF,
    US_NORMS,
    build_subscales,
    derive_uk_structure,
    score_phys_only,
    score_uk_variants,
    score_us,
)
from prevention_health_clustering.measures.sf6d import (
    classify,
    utility,
    validate_tariff,
)


def full_items(**overrides) -> pd.DataFrame:
    """One complete person-year at best health, with overrides."""
    best = {"sf1": 1, "sf2a": 3, "sf2b": 3, "sf3a": 5, "sf3b": 5,
            "sf4a": 5, "sf4b": 5, "sf5": 1, "sf6a": 1, "sf6b": 1,
            "sf6c": 5, "sf7": 5}
    best.update(overrides)
    return pd.DataFrame([best], dtype=float)


def test_subscales_hand_computed():
    sub = build_subscales(full_items())
    for s in SCALES:
        assert sub[s].iloc[0] == 100.0, f"{s} at best health should be 100"
    # GH recalibration: sf1=2 ("very good") -> 4.4 -> (4.4-1)/4*100 = 85
    sub = build_subscales(full_items(sf1=2))
    assert abs(sub["GH"].iloc[0] - 85.0) < 1e-12
    # Worst health: every subscale 0
    worst = full_items(sf1=5, sf2a=1, sf2b=1, sf3a=1, sf3b=1, sf4a=1,
                       sf4b=1, sf5=5, sf6a=5, sf6b=5, sf6c=1, sf7=1)
    sub = build_subscales(worst)
    for s in SCALES:
        assert sub[s].iloc[0] == 0.0, f"{s} at worst health should be 0"


def test_subscales_incomplete_row_is_all_nan():
    items = full_items()
    items.loc[0, "sf6b"] = np.nan
    sub = build_subscales(items)
    assert sub.iloc[0].isna().all()
    sub = build_subscales(items, require_complete=False)
    assert np.isnan(sub["VT"].iloc[0]) and sub["PF"].iloc[0] == 100.0


def test_score_us_at_norms_is_50():
    sub = pd.DataFrame([{s: US_NORMS[s][0] for s in SCALES}])
    scores = score_us(sub)
    assert abs(scores["PCS_us"].iloc[0] - 50.0) < 1e-9
    assert abs(scores["MCS_us"].iloc[0] - 50.0) < 1e-9
    # +1 US SD on PF alone moves PCS by exactly 10 * 0.42402
    sub_pf = sub.copy()
    sub_pf["PF"] += US_NORMS["PF"][1]
    delta = score_us(sub_pf)["PCS_us"].iloc[0] - 50.0
    assert abs(delta - 10 * US_COEF["PCS"]["PF"]) < 1e-9


def test_orthogonality_artifact_direction():
    """Worse mental health raises measured PCS — the artifact, by construction."""
    base = pd.DataFrame([{s: US_NORMS[s][0] for s in SCALES}])
    worse_mh = base.copy()
    worse_mh["MH"] -= US_NORMS["MH"][1]
    worse_mh["RE"] -= US_NORMS["RE"][1]
    delta = score_us(worse_mh)["PCS_us"].iloc[0] - score_us(base)["PCS_us"].iloc[0]
    expected = -10 * (US_COEF["PCS"]["MH"] + US_COEF["PCS"]["RE"])
    assert abs(delta - expected) < 1e-9 and delta > 4.0


def simulate_population(n=20000, factor_corr=0.6, seed=20260825):
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, factor_corr], [factor_corr, 1.0]])
    f = rng.multivariate_normal([0, 0], cov, size=n)
    loading = {"PF": (0.85, 0.0), "RP": (0.80, 0.10), "BP": (0.75, 0.10),
               "GH": (0.65, 0.25), "VT": (0.25, 0.65), "SF": (0.15, 0.70),
               "RE": (0.10, 0.75), "MH": (0.0, 0.85)}
    sub = pd.DataFrame(index=range(n))
    for s in SCALES:
        lp, lm = loading[s]
        uniq = np.sqrt(max(1e-6, 1 - lp**2 - lm**2 - 2 * lp * lm * factor_corr))
        latent = lp * f[:, 0] + lm * f[:, 1] + uniq * rng.normal(size=n)
        sub[s] = 70 + 20 * latent
    return sub


def test_uk_structure_recovery():
    sub = simulate_population()
    weights = pd.Series(np.ones(len(sub)))
    st = derive_uk_structure(sub, weights)
    # Orientation: physical factor loads positively on PF, mental on MH
    assert st.loadings_varimax[SCALES.index("PF"), 0] > 0.5
    assert st.loadings_varimax[SCALES.index("MH"), 1] > 0.5
    # Promax recovers the simulated inter-factor correlation to first order.
    # st.phi is the properly normalised correlation; st.phi_raw is the
    # unnormalised off-diagonal the sandbox printed (diag != 1), kept only to
    # reproduce the construction note's reported figure.
    assert abs(st.phi - 0.6) < 0.12, f"phi {st.phi:.3f} far from 0.6"
    assert st.phi <= 1.0
    assert st.phi_raw != st.phi  # the two conventions genuinely differ
    # Norms are the sample moments
    assert abs(st.norms["PF"][0] - sub["PF"].mean()) < 1e-9
    assert abs(st.phys_only_weights.sum() - 1.0) < 1e-12
    # Variant scores: anchored at 50/10 in the reference population
    scores = score_uk_variants(sub, st, sub, weights)
    for col in scores.columns:
        assert abs(scores[col].mean() - 50.0) < 1e-6
        assert abs(scores[col].std(ddof=0) - 10.0) < 1e-6
    phys = score_phys_only(sub, st)
    assert abs(phys.mean() - 50.0) < 1e-6
    # Phys-only ignores pure mental variation entirely
    shifted = sub.copy()
    for s in ("VT", "SF", "RE", "MH"):
        shifted[s] += 15
    assert np.allclose(score_phys_only(shifted, st), phys)


def test_sf6d_classification_corners():
    # RL: v2 role items dichotomised at "none of the time" (5)
    cases = [
        ({"sf3a": 5, "sf4a": 5}, 1),   # no limitation
        ({"sf3a": 1, "sf4a": 5}, 2),   # physical only
        ({"sf3a": 5, "sf4a": 1}, 3),   # emotional only
        ({"sf3a": 2, "sf4a": 3}, 4),   # both
    ]
    for overrides, expected in cases:
        st = classify(full_items(**overrides))
        assert st["RL"].iloc[0] == expected, (overrides, expected)
    st = classify(full_items())
    assert all(st[d].iloc[0] == 1 for d in ("PF", "SF", "PAIN", "MH", "VIT"))
    # Vitality is positively worded: maps directly, no reversal
    assert classify(full_items(sf6b=4))["VIT"].iloc[0] == 4


def test_sf6d_utility_and_most_severe():
    checks = validate_tariff()
    assert all(ok for _, ok, _ in checks), checks
    # PAIN level 4 alone: decrement 0.077 AND the starred most-severe 0.077
    st = classify(full_items(sf5=4))
    assert st["PAIN"].iloc[0] == 4
    assert abs(utility(st).iloc[0] - (1.0 - 0.077 - 0.077)) < 1e-12
    # RL level 2 (not starred): only its own decrement
    st = classify(full_items(sf3a=1))
    assert abs(utility(st).iloc[0] - (1.0 - 0.063)) < 1e-12
    # Missing item -> NaN utility
    items = full_items()
    items.loc[0, "sf5"] = np.nan
    assert utility(classify(items)).isna().iloc[0]


def test_composites():
    sub = simulate_population(n=5000)
    norms = {s: (sub[s].mean(), sub[s].std(ddof=0)) for s in SCALES}
    far = farivar_composites(sub, norms)
    assert abs(far["PCS_c_farivar"].mean() - 50.0) < 1e-6
    assert abs(far["PCS_c_farivar"].std(ddof=0) - 10.0) < 1e-6
    # Anchoring is population-relative, so test ORDERING: within the frame,
    # role-physical limitation moves PCS_c, mental health moves MCS_c.
    assert far["PCS_c_farivar"].corr(sub["RP"]) > 0.5
    assert far["MCS_c_farivar"].corr(sub["MH"]) > 0.5
    # A uniform shift of every row is absorbed entirely by the re-anchoring.
    shifted = sub.copy()
    shifted["RP"] -= 20
    assert np.allclose(farivar_composites(shifted, norms)["PCS_c_farivar"],
                       far["PCS_c_farivar"])
    eq = equal_weight_composite(sub)
    assert abs(eq.iloc[0] - sub.iloc[0][list(SCALES)].mean()) < 1e-9
    sub.loc[0, "MH"] = np.nan
    assert np.isnan(equal_weight_composite(sub).iloc[0])


def test_wave_item_columns_prefers_self_completion():
    header = ["pidp", "b_scsf1", "b_sf1", "b_scsf2a", "b_sf12pcs_dv", "b_dvage"]
    mapping = wave_item_columns("b", header)
    assert mapping["sf1"] == "b_scsf1"
    assert mapping["sf2a"] == "b_scsf2a"
    assert mapping["sf12pcs_dv"] == "b_sf12pcs_dv"
    assert mapping["age"] == "b_dvage"
    # Wave-1 style: no sc prefix
    mapping = wave_item_columns("a", ["pidp", "a_sf1", "a_sf2a", "a_dvage"])
    assert mapping["sf1"] == "a_sf1"


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
    print("MEASURES TESTS OK" if failures == 0 else
          f"MEASURES TESTS FAILED ({failures})")
    sys.exit(1 if failures else 0)
