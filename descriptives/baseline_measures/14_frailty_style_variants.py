"""Frailty-index style unweighted versions of the P-LIM3 variable set.

The question is whether one variable set can carry the graded-response theta,
its 0-1 expected score, and a deficit-accumulation index that anybody can
compute by hand. FI-15 shows that "unweighted" is not one thing: equal weights
per deficit put eight of fifteen on the limitation indicators, and its cost
convexity does not survive capping. These variants change only how the same
answers are pooled, from most limitation weight to least:

  fi_area     14 deficits: six SF-12 items graded 0-1 by response step, and the
              eight impairment areas as binary deficits (FI-15 without the
              long-standing illness deficit)
  fi_group     9 deficits: the six SF-12 items, and functional limitations,
              self-care and sensory each as a share of their areas
  fi_testlet   7 deficits: the four SF-12 testlets, each 0-1 by response step,
              and the same three limitation shares
Each is reported as one minus the mean deficit, so higher is better health, and
each has a +CC version adding the condition count as one more deficit.

Writes data/processed/baseline_measures/frailty_variants.parquet; the criteria
come from 05_master_criteria.py, 06_master_ident.R and 13_cost_age_invariance.py.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from prevention_health_clustering.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from prevention_health_clustering.measures.grm2 import DISDIF_LISTED

OUT = PROCESSED_DATA_DIR / "baseline_measures"
AREAS = {"mobility": 1, "lifting": 2, "dexterity": 3, "coordination": 10, "personal_care": 11,
         "continence": 4, "hearing": 5, "sight": 6}
GROUPS = {"functional": ["mobility", "lifting", "dexterity", "coordination"],
          "self_care": ["continence", "personal_care"], "sensory": ["hearing", "sight"]}
SF_ITEMS = ["gh", "pf_mod", "pf_stairs", "rp_less", "rp_kind", "pain"]
SF_TESTLETS = {"gh_t": ["gh"], "pf_t": ["pf_mod", "pf_stairs"], "rp_t": ["rp_less", "rp_kind"], "bp_t": ["pain"]}


def main() -> int:
    items = pd.read_parquet(INTERIM_DATA_DIR / "sf12_items_long.parquet")
    items = items[items["age"].notna() & items["age"].between(20, 90)].copy()

    def valid(s, k):
        return s.where(s.isin(range(1, k + 1)))

    sf1, sf2a, sf2b = valid(items.sf1, 5), valid(items.sf2a, 3), valid(items.sf2b, 3)
    sf3a, sf3b, sf5 = valid(items.sf3a, 5), valid(items.sf3b, 5), valid(items.sf5, 5)
    health = items.health.where(items.health.isin([1, 2]))
    D = pd.DataFrame(index=items.index)
    D["gh"] = (sf1 - 1) / 4                      # each SF-12 item graded 0 to 1 by response step
    D["pf_mod"] = (3 - sf2a) / 2
    D["pf_stairs"] = (3 - sf2b) / 2
    D["rp_less"] = (5 - sf3a) / 4
    D["rp_kind"] = (5 - sf3b) / 4
    D["pain"] = (sf5 - 1) / 4
    asked = items[list(DISDIF_LISTED)].notna().any(axis=1)
    for name, code in AREAS.items():           # the wave-7 rule, as everywhere else
        v = (items[f"disdif{code}"] == 1).astype(float).where((health == 1) & asked)
        D[name] = v.mask(health == 2, 0.0)
    for g, members in list(GROUPS.items()) + list(SF_TESTLETS.items()):
        # complete cases within the group: a partial mean would quietly widen the sample
        D[g] = D[members].mean(axis=1).where(D[members].notna().all(axis=1))

    chronic = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "chronic_conditions.parquet",
                              columns=["pidp", "wave"] + [f"n_{g}" for g in
                                                          ("cvd", "metab", "resp", "msk", "cancer", "other")])
    ch = items[["pidp", "wave"]].merge(chronic, on=["pidp", "wave"], how="left")
    ngroups = [f"n_{g}" for g in ("cvd", "metab", "resp", "msk", "cancer", "other")]
    total = ch[ngroups].sum(axis=1).where(ch[ngroups].notna().all(axis=1))
    D["conditions"] = (total.clip(upper=6) / 6).to_numpy()     # the count as one 0-1 deficit, capped at six
    # the six condition groups as deficits, capped exactly as P-FULL's items
    CG_CAP = {"cvd": 2, "metab": 2, "resp": 2, "msk": 1, "cancer": 1, "other": 1}
    for g, cap in CG_CAP.items():
        D[f"cond_{g}"] = (ch[f"n_{g}"].clip(upper=cap) / cap).to_numpy()
    CG = [f"cond_{g}" for g in CG_CAP]

    SETS = {"fi_area": SF_ITEMS + list(AREAS),
            "fi_group": SF_ITEMS + list(GROUPS),
            "fi_testlet": list(SF_TESTLETS) + list(GROUPS)}
    out = items[["pidp", "wave", "age"]].copy()
    rows = []
    for name, cols in SETS.items():
        for suffix, extra in (("", []), ("_cc", ["conditions"]), ("_cg", CG)):
            use = cols + extra
            x = D[use]
            fi = x.mean(axis=1).where(x.notna().all(axis=1))
            out[f"{name}{suffix}"] = 1 - fi
            sf_w = len([c for c in use if c in SF_ITEMS or c in SF_TESTLETS]) / len(use)
            rows.append({"index": f"{name}{suffix}", "deficits": len(use), "SF-12 share of weight": sf_w,
                         "condition share": len([c for c in use if c.startswith("cond")]) / len(use),
                         "n": int(fi.notna().sum()), "distinct": int((1 - fi).nunique())})
    out.to_parquet(OUT / "frailty_variants.parquet", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"\nwrote frailty_variants.parquet, {len(SETS) * 3} indices")
    return 0


if __name__ == "__main__":
    sys.exit(main())
