"""Exactly how the P-FULL roster differs from the P-FUNC roster.

Kept as the record behind section 9's measurement-error comparison: whether
P-FULL's lower measurement noise could be a sample artefact rather than a
property of the instrument. Answers three things: whole people or shorter
windows, who the missing people are, and whether they differ on anything
that would move sigma_meas.

    PYTHONPATH=src .venv/bin/python descriptives/18_roster_difference.py
"""

from __future__ import annotations

import sys

import pandas as pd

from prevention_health_clustering.config import PROCESSED_DATA_DIR as P

FULL = "physgrm_lifecycle_20_89_minobs3_v1"
FUNC = "physfunc_lifecycle_20_89_minobs3_v1"


def main() -> int:
    full = pd.read_csv(P / "contracts" / FULL / "long.csv", usecols=["pidp"])
    func = pd.read_csv(P / "contracts" / FUNC / "long.csv")
    keep = set(full["pidp"])
    func["dropped"] = ~func["pidp"].isin(keep)

    n_func, n_full = func["pidp"].nunique(), len(keep)
    print(f"FUNC {n_func:,} people / {len(func):,} rows；"
          f"FULL {n_full:,} people / {len(full):,} rows".replace("；", "; "))
    print(f"in FULL but not FUNC: {len(keep - set(func['pidp'])):,} "
          "(P-FULL is a strict subset)")

    lost = len(func) - len(full)
    from_people = int(func["dropped"].sum())
    print(f"\nrows FUNC has and FULL does not: {lost:,}")
    print(f"  from people absent entirely: {from_people:,} "
          f"({from_people / lost:.1%})")
    print(f"  from shorter windows:        {lost - from_people:,} "
          f"({1 - from_people / lost:.1%})")

    a = func[~func["dropped"]].groupby("pidp").size().rename("func_obs")
    b = pd.read_csv(P / "contracts" / FULL / "long.csv", usecols=["pidp"]) \
        .groupby("pidp").size().rename("full_obs")
    j = pd.concat([a, b], axis=1)
    same = (j["func_obs"] == j["full_obs"]).mean()
    print(f"\nof the {len(j):,} people in both, {same:.1%} keep an IDENTICAL "
          f"window (mean {a.mean():.2f} waves either way);")
    lostw = (j["func_obs"] - j["full_obs"])
    print(f"  {(lostw > 0).sum():,} lose any waves at all, "
          f"mean {lostw[lostw > 0].mean():.2f} waves when it happens")

    g = func.groupby("dropped")
    print("\n                         retained          dropped")
    for lab, col in (("mean obs/person", None), ("mean age", "age"),
                     ("mean theta", "theta_phys_func"),
                     ("sd theta", "theta_phys_func")):
        if col is None:
            v = (g.size() / g["pidp"].nunique())
        elif lab.startswith("sd"):
            v = g[col].std()
        else:
            v = g[col].mean()
        print(f"  {lab:22s} {v[False]:8.3f} {v[True]:16.3f}")

    fw = func.groupby("pidp").agg(first=("wave", "min"), d=("dropped", "first"))
    t = pd.crosstab(fw["first"], fw["d"], normalize="columns")
    print(f"\nfirst observed at wave 2: retained {t.loc[2, False]:.1%}, "
          f"dropped {t.loc[2, True]:.1%}  (the BHPS entrants)")

    func["absdiff"] = (func.sort_values(["pidp", "age"])
                       .groupby("pidp")["theta_phys_func"].diff().abs())
    m = func.groupby("dropped")["absdiff"].mean()
    print(f"\nmean |year-to-year change| in theta: retained {m[False]:.4f}, "
          f"dropped {m[True]:.4f}")
    print("  the dropped are slightly LESS volatile, so removing them should "
          "raise\n  measured noise, not lower it: the confound runs against "
          "the finding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
