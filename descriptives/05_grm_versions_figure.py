"""The measure-choice panel for the GRM versions (empirics-note Figure 26).

Four reads across the GRM versions, with the UKHLS PCS and SF-6D as
references: (a) the standardised mean age path, (b) fanning of the dispersion,
(c) top-censoring mass by age, (d) the mental-health artifact at a fixed
physical profile and the spread across 1945-65 birth cohorts at ages 50-59.
Birth year comes from xwavedat (birthy, fallback doby_dv).

Outputs: docs/figures/fig_grm_versions.png +
         artifacts/descriptives/grm_versions_criteria.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _style import INK2, apply_style  # noqa: E402

from prevention_health_clustering.config import (
    ARTIFACTS_DIR,
    PROCESSED_DATA_DIR,
    ROOT_DIR,
    UKHLS_PANEL_DIR,
)

MEASURES = {
    "sf12pcs_dv": ("UKHLS PCS", "#12395B"),
    "sf6d_utility": ("SF-6D", "#3D9970"),
    "grm_theta": ("GRM original (4 testlets)", "#CC5500"),
    "theta_phys_func": ("GRM-2 phys, functioning", "#2E86C1"),
    "theta_phys_full": ("GRM-2 phys, +conditions", "#7B52AB"),
    "theta_ment": ("GRM-2 mental", "#CC79A7"),
    "theta_combined": ("GRM-2 combined", "#E69F00"),
}
AGES = np.arange(25, 81)


def main() -> int:
    apply_style()
    panel = pd.read_parquet(
        PROCESSED_DATA_DIR / "measures" / "measure_panel.parquet",
        columns=["pidp", "wave", "age", "sf12mcs_dv",
                 "sub_PF", "sub_RP", "sub_BP", "sub_GH", "sub_MH"]
        + list(MEASURES))
    panel = panel[panel["age"].notna()]
    panel["age"] = panel["age"].astype(int)
    panel = panel[panel["age"].between(20, 90)]

    xw = pd.read_csv(UKHLS_PANEL_DIR.parent.parent / "tab" / "ukhls" / "xwavedat.tab",
                     sep="\t", usecols=["pidp", "birthy"], low_memory=False)
    xw["birthy"] = pd.to_numeric(xw["birthy"], errors="coerce")
    xw.loc[xw["birthy"] < 0, "birthy"] = np.nan
    panel = panel.merge(xw, on="pidp", how="left")

    def prof(v, f):
        g = panel.groupby("age")[v]
        return g.apply(f).reindex(AGES)

    # criteria table
    young = panel[panel["age"].between(25, 34)]
    old = panel[panel["age"].between(70, 80)]
    med = panel[(panel["sub_PF"] >= 100) & (panel["sub_RP"] >= 100)
                & (panel["sub_BP"] >= 100)
                & (panel["sub_GH"].sub(60).abs() < 1e-6)]
    mdec = med["sub_MH"].quantile([0.1, 0.9])
    coh = panel[panel["age"].between(50, 59)].copy()
    coh["cohbin"] = 5 * (coh["birthy"] // 5)
    coh = coh[coh["cohbin"].between(1945, 1965)]

    rows = []
    for v, (label, _) in MEASURES.items():
        x = panel[v].dropna()
        sd = x.std()
        mx, mn = x.max(), x.min()
        art = (med.loc[med["sub_MH"] <= mdec.iloc[0], v].mean()
               - med.loc[med["sub_MH"] >= mdec.iloc[1], v].mean()) / sd
        cm = coh.groupby("cohbin")[v].mean()
        rows.append({
            "measure": v, "label": label,
            "ceiling_pct": 100 * (x >= mx - 1e-9).mean(),
            "floor_pct": 100 * (x <= mn + 1e-9).mean(),
            "age_grad_sd": (old[v].mean() - young[v].mean()) / sd,
            "fanning": old[v].std() / young[v].std(),
            "cor_MCS": x.corr(panel["sf12mcs_dv"]),
            "artifact_sd": art,
            "cohort_sd": (cm.max() - cm.min()) / sd,
        })
    crit = pd.DataFrame(rows)
    out_dir = ARTIFACTS_DIR / "descriptives"
    out_dir.mkdir(parents=True, exist_ok=True)
    crit.to_csv(out_dir / "grm_versions_criteria.csv", index=False)
    print(crit.round(3).to_string(index=False))

    # figure
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    ax = axes[0, 0]
    for v, (label, color) in MEASURES.items():
        m = prof(v, "mean")
        s = panel[v].std()
        ax.plot(AGES, (m - m.iloc[0]) / s, lw=1.9, color=color, label=label)
    ax.axhline(0, color="#bbbbbb", lw=0.7, ls=":")
    ax.set_title("(a) Mean age path, in own-sd units relative to age 25")
    ax.set_xlabel("age")
    ax.legend(fontsize=7.2, ncols=2)

    ax = axes[0, 1]
    for v, (label, color) in MEASURES.items():
        s = prof(v, "std")
        ax.plot(AGES, s / s.iloc[0], lw=1.9, color=color)
    ax.axhline(1, color="#bbbbbb", lw=0.7, ls=":")
    ax.set_title("(b) Dispersion, relative to age 25")
    ax.set_xlabel("age")

    ax = axes[1, 0]
    for v, (label, color) in MEASURES.items():
        mx = panel[v].max()
        s = prof(v, lambda z: 100 * (z.dropna() >= mx - 1e-9).mean())
        ax.plot(AGES, s, lw=1.9, color=color)
    ax.set_title("(c) Share of person-years at the exact maximum")
    ax.set_xlabel("age")
    ax.set_ylabel("%")

    ax = axes[1, 1]
    idx = np.arange(len(MEASURES))
    ax.bar(idx - 0.2, crit["artifact_sd"], width=0.38, color="#CC5500",
           label="MH artifact at fixed physical profile")
    ax.bar(idx + 0.2, crit["cohort_sd"], width=0.38, color="#8a8a8a",
           label="spread across 1945–65 cohorts, ages 50–59")
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_xticks(idx, [lab for lab, _ in MEASURES.values()],
                  rotation=18, ha="right", fontsize=7)
    ax.set_ylabel("sd units")
    ax.set_title("(d) Contamination")
    ax.legend(fontsize=7.2)

    fig.suptitle("Choosing among the GRM versions", fontweight="bold", y=0.995)
    fig.text(0.01, -0.012,
             "Panels (a)–(c) on ages 25–80. The artifact in (d) is the worst-minus-best mental-health "
             "decile gap among person-years at the identical median physical profile — negative for the "
             "mental measures by construction (they are supposed to move).",
             fontsize=7.5, color=INK2)
    fig.tight_layout()
    out = ROOT_DIR / "docs" / "figures" / "fig_grm_versions.png"
    fig.savefig(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
