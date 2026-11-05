"""Figure 3 (Exhibit 2) on PSID: Case 4 on theta, four ages four years apart, pairwise moments.

The main design of descriptives/29_paper_moments.py, run with its own fit_design and barchart and
the _paper_moments machinery (properly constrained fit, rho profiled band by band, a base age
enters when every cell rests on at least 50 people observed at both of its ages). Four ages four
years apart sit on every second PSID wave, so the design carries over unchanged.

The identification panel is _paper_moments.ident_panel's rule applied to PSID (copied here because
the original reads the UKHLS scores): one row per person-age (earliest wave kept), people whose age
sequence is implausible dropped (age falling, or rising by more than the gap in years plus one),
ages 20-90, people with at least four observed ages.

Outputs: descriptives/psid/figures/main/fig_exhibit2_main.png, artifacts/descriptives/psid/ident_corr_band.csv
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _psid_common import DESC, FIG, scores  # noqa: E402
from _paper_moments import corr_of, wide  # noqa: E402
from _style import apply_style  # noqa: E402

M29 = importlib.import_module("29_paper_moments")


def ident_panel(min_obs: int = 4) -> pd.DataFrame:
    """Copy of _paper_moments.ident_panel on the PSID scores (wave is the calendar year)."""
    d = scores().sort_values(["pidp", "wave"])
    dage = d.groupby("pidp")["age"].diff(); dwave = d.groupby("pidp")["wave"].diff()
    bad = d.loc[dage.notna() & ((dage < 0) | (dage > dwave + 1)), "pidp"].unique()
    d = d[~d["pidp"].isin(bad)]
    d = d.sort_values(["pidp", "age", "wave"]).drop_duplicates(["pidp", "age"], keep="first")
    nobs = d.groupby("pidp")["age"].transform("size")
    return d[nobs >= min_obs].reset_index(drop=True)


def main() -> int:
    apply_style()
    d = ident_panel()
    print(f"identification panel: {len(d):,} person-ages, {d['pidp'].nunique():,} people")
    spec = M29.DESIGNS["main"]
    fits = {"theta": M29.fit_design(wide(d, "theta"), spec)}
    rows = []
    for (bn, case), (m, n, f) in fits["theta"].items():
        rows.append({"band": bn, "case": case, "n": n, "corr": corr_of(f["b"]), "rho": f["rho"], "VH": f["b"][0], "CHd": f["b"][1],
                     "Vd": f["b"][2], "binds": bool(f.get("binds")), "weak": bool(f.get("weak")), "cost": f.get("cost", np.nan),
                     "corr_unconstrained": corr_of(f["unconstrained"]["b"])})
    t = pd.DataFrame(rows)
    t.to_csv(DESC / "ident_corr_band.csv", index=False)
    print(t[t["case"].isin(["case1", "case4"])].round(3).to_string(index=False))
    foot = ("PSID, " + f"{d['pidp'].nunique():,} people with at least four observed ages 20-90. "
            "Four ages four years apart (every second biennial wave), pairwise moments pooled over base ages within each band; a base age enters "
            "when every cell rests on at least 50 people observed at both of its ages.\n"
            "Properly constrained fit: V_H, V_d and noise variances non-negative and |Corr(H,d)| <= 1 imposed in the estimation, rho profiled. "
            "Where the unconstrained fit was inadmissible the title gives it and the fit cost of the constraint\nin squared error. "
            "Bars stack the fitted layers; black ticks are the observed moments. V(k): the variance at the k-th age; C(k,l): the covariance between ages k and l. "
            "The one-period error is a two-year one here.")
    M29.barchart(FIG / "fig_exhibit2_main.png", [("theta", "case4")], fits, spec, "Exhibit 2: Case 4 on theta, PSID", foot)
    print("wrote fig_exhibit2_main.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
