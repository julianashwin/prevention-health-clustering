"""Unit tests for the SF-12 item extraction helpers."""

from __future__ import annotations

from prevention_health_clustering.measures.items import wave_item_columns


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
