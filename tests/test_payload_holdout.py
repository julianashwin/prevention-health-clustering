"""The last-k within-person holdout in build_payload (no Stan needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prevention_health_clustering.models.registry import get_model
from prevention_health_clustering.runner.fit import build_payload


def frame():
    rng = np.random.default_rng(1)
    rows = []
    for pid, n in [(1, 6), (2, 5), (3, 4), (4, 3), (5, 8)]:
        ages = np.sort(rng.choice(np.arange(30, 70), n, replace=False))
        for a in ages:
            rows.append({"pidp": pid, "age": a, "wave": 1, "birthy": 1960,
                         "sf12pcs_dv": rng.normal(50, 10)})
    return pd.DataFrame(rows)


def test_last_k_holdout_windows():
    spec = get_model("pcs-ar1")
    p = build_payload(frame(), spec=None) if False else build_payload(
        spec, frame(), holdout_last_k=2, holdout_min_person_obs=5)
    d = p.data
    assert d["N_person"] == 3          # pids 1, 2, 5 have >= 5 observations
    assert d["N_obs"] == 19
    for i in range(d["N_person"]):
        held = d["hold_end"][i] - d["hold_start"][i] + 1
        assert held == 2
        assert d["hold_start"][i] == d["fit_end"][i] + 1   # contiguous suffix
    fitted = sum(d["fit_end"][i] - d["fit_start"][i] + 1
                 for i in range(d["N_person"]))
    assert fitted == 19 - 6


def test_modes_exclusive_and_baseline_unchanged():
    spec = get_model("pcs-ar1")
    p0 = build_payload(spec, frame())
    assert p0.data["N_person"] == 5
    assert all(p0.data["hold_end"][i] == 0 for i in range(5))
    try:
        build_payload(spec, frame(), holdout_min_age=50, holdout_last_k=1)
        raise AssertionError("exclusive modes not enforced")
    except ValueError:
        pass
    try:
        build_payload(spec, frame(), holdout_last_k=3, holdout_min_person_obs=3)
        raise AssertionError("min_obs <= k not rejected")
    except ValueError:
        pass


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
    print("HOLDOUT TESTS OK" if failures == 0 else "HOLDOUT TESTS FAILED")
    sys.exit(1 if failures else 0)
