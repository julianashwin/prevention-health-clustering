"""Data-free tests of the two-dimensional GRM item banks (measures/grm2.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prevention_health_clustering.measures.grm2 import (
    GHQ_NEGATIVE,
    GHQ_POSITIVE,
    MENT_ITEMS,
    build_mental_items,
    collapse_ghq_wording,
)


def frame(**cols) -> pd.DataFrame:
    base = {
        "pidp": [1], "wave": [1], "age": [50.0],
        "sf1": [1.0], "sf2a": [3.0], "sf2b": [3.0], "sf3a": [5.0],
        "sf3b": [5.0], "sf4a": [5.0], "sf4b": [5.0], "sf5": [1.0],
        "sf6a": [1.0], "sf6b": [1.0], "sf6c": [5.0], "sf7": [5.0],
        "health": [2.0],
        **{f"disdif{i}": [np.nan] for i in range(1, 13)},
        "disdif96": [np.nan],
        **{f"scghq{c}": [1.0] for c in "abcdefghijkl"},
    }
    base.update(cols)
    return pd.DataFrame(base)


def chronic_frame(**cols) -> pd.DataFrame:
    base = {"pidp": [1], "wave": [1],
            **{f"n_{g}": [0.0] for g in
               ("cvd", "metab", "resp", "msk", "cancer", "other")},
            **{f"nrec10_{g}": [0.0] for g in
               ("cvd", "metab", "resp", "msk", "cancer", "other")},
            "ever_17": [0.0]}
    base.update(cols)
    return pd.DataFrame(base)


def test_mental_bank_directions():
    ch = chronic_frame(ever_17=[1.0])
    bank = build_mental_items(frame(), ch)
    row = bank.iloc[0]
    # best-health inputs: MH = (6-1)+5-1 = 9, RE = 9, SF = 5, GHQ all 5-1 = 4
    assert row["MH"] == 9 and row["RE"] == 9 and row["SF"] == 5
    assert all(row[s.upper()] == 4 for s in
               [f"scghq{c}" for c in "abcdefghijkl"])
    assert row["DEPR"] == 1       # diagnosed depression -> worst category
    worst = frame(sf4a=[1.0], sf4b=[1.0], sf6a=[5.0], sf6c=[1.0], sf7=[1.0],
                  **{f"scghq{c}": [4.0] for c in "abcdefghijkl"})
    row = build_mental_items(worst, chronic_frame()).iloc[0]
    assert row["MH"] == 1 and row["RE"] == 1 and row["SF"] == 1
    assert row["DEPR"] == 2 and row["SCGHQA"] == 1


def test_ghq_wording_collapse():
    ch = chronic_frame()
    bank = build_mental_items(frame(), ch)
    t = collapse_ghq_wording(bank).iloc[0]
    assert t["GHQPOS"] == 19 and t["GHQNEG"] == 19   # six 4s -> 24 - 5
    assert len(GHQ_POSITIVE) == len(GHQ_NEGATIVE) == 6
    assert set(GHQ_POSITIVE) | set(GHQ_NEGATIVE) == {
        s.upper() for s in [f"scghq{c}" for c in "abcdefghijkl"]}
    assert len(MENT_ITEMS) == 16


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
    print("GRM2 TESTS OK" if failures == 0 else f"GRM2 TESTS FAILED ({failures})")
    sys.exit(1 if failures else 0)
