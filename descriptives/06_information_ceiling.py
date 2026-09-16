"""How each GRM is constructed, and where every metric runs out of room.

Two outputs:

* fig_grm_information.png — per GRM version, the stacked item-information
  decomposition over the latent scale with the implied measurement SE: which
  items carry the measurement, and where the instrument goes dark. This is
  the construction illustration: a version IS its information profile.

* ceiling_floor.csv + a printed table — for every metric: share of
  person-years at the exact maximum and minimum, the attainable score range,
  and (for the GRMs) the EAP shrinkage and posterior sd at the all-best and
  all-worst response patterns. Tests the 'more items buy more headroom'
  hypothesis directly.
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
)
from prevention_health_clustering.measures.grm import cat_probs

TH = np.linspace(-4, 4, 161)

SPECS = {
    "GRM original": ("grm_items.csv", None, "grm_theta"),
    "P-FUNC": ("grm2_items.csv", "P-FUNC", "theta_phys_func"),
    "P-FULL": ("grm2_items.csv", "P-FULL", "theta_phys_full"),
    "MENT": ("grm2_items.csv", "MENT", "theta_ment"),
    "COMBINED": ("grm2_items.csv", "COMBINED", "theta_combined"),
}

SCORE_METRICS = ["sf12pcs_dv", "sf12mcs_dv", "PCS_phys_only", "sf6d_utility",
                 "ghq_likert", "grm_theta", "theta_phys_func",
                 "theta_phys_full", "theta_ment", "theta_combined"]


def load_items(fname, spec):
    df = pd.read_csv(PROCESSED_DATA_DIR / "measures" / fname)
    if spec is not None:
        df = df[df["spec"] == spec]
    items = {}
    for _, row in df.iterrows():
        b = row[[c for c in df.columns if c.startswith("b")]].dropna().to_numpy(float)
        items[row["item"]] = (float(row["a"]), b)
    return items


def item_information(a, b, th):
    eps = 1e-4
    pr = cat_probs(a, b, th)
    p2 = cat_probs(a, b, th + eps)
    return (((p2 - pr) / eps) ** 2 / pr).sum(axis=1)


def main() -> int:
    apply_style()
    panel = pd.read_parquet(
        PROCESSED_DATA_DIR / "measures" / "measure_panel.parquet",
        columns=SCORE_METRICS)

    # ---- information decomposition figure ----------------------------------
    ncols = 3
    nrows = int(np.ceil(len(SPECS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.5 * nrows),
                             sharex=True)
    axes = np.atleast_2d(axes)
    cmap = plt.get_cmap("viridis")
    for j, (name, (fname, spec, col)) in enumerate(SPECS.items()):
        r, c = divmod(j, ncols)
        ax = axes[r, c]
        items = load_items(fname, spec)
        info = np.column_stack(
            [item_information(a, b, TH) for a, b in items.values()])
        colors = [cmap(0.05 + 0.9 * i / max(len(items) - 1, 1))
                  for i in range(len(items))]
        ax.stackplot(TH, info.T, colors=colors, alpha=0.92, labels=list(items))
        total = info.sum(axis=1)
        ax2 = ax.twinx()
        ax2.plot(TH, 1 / np.sqrt(np.maximum(total, 1e-9)), color="#D55E00",
                 lw=1.6, ls="--")
        ax2.set_ylim(0, 1.2)
        # SE scale only on the last occupied panel of each row
        last_in_row = (j == len(SPECS) - 1) or (divmod(j + 1, ncols)[0] != r)
        if last_in_row:
            ax2.set_ylabel("measurement SE (dashed)", fontsize=8)
        else:
            ax2.set_yticks([])
        ax.set_title(f"{name} ({len(items)} items)", fontsize=10)
        if r == nrows - 1 or j >= len(SPECS) - ncols:
            ax.set_xlabel(r"latent health $\theta$")
        if c == 0:
            ax.set_ylabel("Fisher information (stacked)")
        ax.legend(fontsize=5.6, ncols=2, loc="upper right")
    for j in range(len(SPECS), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")
    fig.suptitle("What each GRM is made of: item information over the latent scale",
                 fontweight="bold", y=1.0)
    fig.text(0.01, -0.005,
             "Each band is one item's Fisher information. The dashed line is the implied measurement SE,\n"
             "1/sqrt(total information): where it rises, the instrument stops discriminating \u2014 the IRT\n"
             "reading of ceiling and floor.",
             fontsize=7.5, color=INK2, va="top")
    fig.tight_layout()
    out = ROOT_DIR / "measuring_health" / "figures" / "fig_grm_information.png"
    fig.savefig(out)
    print(f"wrote {out}")

    # ---- ceiling / floor table ---------------------------------------------
    scores = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "grm2_scores.parquet")
    old = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "grm_scores.parquet",
                          columns=["grm_theta", "grm_theta_sd"])
    sd_at = {}
    for col, sdcol, frame in (
        ("theta_phys_func", "theta_phys_func_sd", scores),
        ("theta_phys_full", "theta_phys_full_sd", scores),
        ("theta_ment", "theta_ment_sd", scores),
        ("theta_combined", "theta_combined_sd", scores),
        ("grm_theta", "grm_theta_sd", old),
    ):
        x = frame[col]
        sd_at[col] = (float(frame.loc[x.idxmax(), sdcol]),
                      float(frame.loc[x.idxmin(), sdcol]))

    rows = []
    for v in SCORE_METRICS:
        x = panel[v].dropna()
        mx, mn = x.max(), x.min()
        rows.append({
            "measure": v, "n": len(x),
            "ceiling_pct": 100 * (x >= mx - 1e-9).mean(),
            "floor_pct": 100 * (x <= mn + 1e-9).mean(),
            "range": mx - mn,
            "sd_units": (mx - mn) / x.std(),
            "psd_at_ceiling": sd_at.get(v, (np.nan, np.nan))[0],
            "psd_at_floor": sd_at.get(v, (np.nan, np.nan))[1],
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(ARTIFACTS_DIR / "descriptives" / "ceiling_floor.csv", index=False)
    print(tab.round(3).to_string(index=False))

    # ---- the floor, in depth ----------------------------------------------
    # SE by latent position: which versions keep measuring below theta = -2?
    probe = np.array([-4.0, -3.0, -2.5, -2.0, -1.0, 0.0])
    frames = {"grm_theta": old["grm_theta"],
              "theta_phys_func": scores["theta_phys_func"],
              "theta_phys_full": scores["theta_phys_full"],
              "theta_ment": scores["theta_ment"],
              "theta_combined": scores["theta_combined"]}
    floor_rows = []
    for name, (fname, spec, col) in SPECS.items():
        items = load_items(fname, spec)
        info = sum(item_information(a, b, probe) for a, b in items.values())
        se = 1 / np.sqrt(np.maximum(info, 1e-12))
        x = frames[col].dropna()
        q01 = x.quantile(0.01)
        floor_rows.append({
            "spec": name,
            **{f"se_at_{t:+.1f}": v for t, v in zip(probe, se)},
            "min_theta": x.min(),
            "pct_below_-2": 100 * (x < -2).mean(),
            "pct_below_-3": 100 * (x < -3).mean(),
            "pct_at_worst_pattern": 100 * (x <= x.min() + 1e-9).mean(),
            "distinct_in_bottom_1pct": int(x[x <= q01].nunique()),
        })
    ftab = pd.DataFrame(floor_rows)
    ftab.to_csv(ARTIFACTS_DIR / "descriptives" / "floor_depth.csv", index=False)
    print("\nfloor depth (SE by latent position; tail occupancy and clumping):")
    print(ftab.round(3).to_string(index=False))

    # resolution among the worst-off: within the P-FUNC bottom decile, how
    # many distinct positions does each richer bank resolve?
    m = scores.dropna(subset=["theta_phys_func", "theta_phys_full",
                              "theta_combined"])
    dec = m["theta_phys_func"] <= m["theta_phys_func"].quantile(0.10)
    print(f"\nP-FUNC bottom decile (n={int(dec.sum()):,}): distinct theta "
          f"FUNC {m.loc[dec, 'theta_phys_func'].nunique():,}, "
          f"FULL {m.loc[dec, 'theta_phys_full'].nunique():,}, "
          f"COMBINED {m.loc[dec, 'theta_combined'].nunique():,}")

    # the honest SF-12-family ceilings: share at the all-best item profile
    sub_cols = [f"sub_{s_}" for s_ in
                ("PF", "RP", "BP", "GH", "VT", "SF", "RE", "MH")]
    sf = pd.read_parquet(PROCESSED_DATA_DIR / "measures" /
                         "sf12_measures.parquet", columns=sub_cols).dropna()
    all8 = (sf >= 100 - 1e-9).all(axis=1).mean()
    phys4 = (sf[sub_cols[:4]] >= 100 - 1e-9).all(axis=1).mean()
    print(f"all-best profile share: 8 subscales {100 * all8:.2f}% "
          f"(PCS/MCS ceiling), physical 4 {100 * phys4:.2f}% (phys-only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
