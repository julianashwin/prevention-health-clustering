"""The ever-diagnosis accumulation check.

Three reads on how much of the physical GRM's age decline is mechanical
accumulation of old diagnoses rather than current health:

  (a) the latent age profiles mu_a from the multigroup fits — functioning-only
      (P-FUNC), with ever-diagnoses (P-FULL), and with only diagnoses made in
      the last ten years (P-REC);
  (b) how stale the ever-stock is: the share of condition flags older than
      10/20 years, by age;
  (c) the person-level signature: the gap theta_full - theta_func against the
      age of the person's oldest diagnosis, holding current functioning fixed
      by construction of the comparison.

Outputs: measuring_health/figures/fig_ever_sensitivity.png + printed summary numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _style import BLUE, GREEN, INK2, ORANGE, PURPLE, apply_style  # noqa: E402

from prevention_health_clustering.config import PROCESSED_DATA_DIR, ROOT_DIR
from prevention_health_clustering.measures.chronic import CODES

AGES = np.arange(20, 91)


def main() -> int:
    apply_style()
    md = PROCESSED_DATA_DIR / "measures"
    prof = pd.read_csv(md / "grm2_age_profiles.csv")
    scores = pd.read_parquet(md / "grm2_scores.parquet")
    chron = pd.read_parquet(md / "chronic_conditions.parquet")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))

    # (a) latent age profiles
    ax = axes[0]
    for col, lab, color in (
        ("mu_P-FUNC", "P-FUNC: functioning only", BLUE),
        ("mu_P-FULL", "P-FULL: + ever diagnoses", ORANGE),
        ("mu_P-REC", "P-REC: + diagnoses ≤ 10y old", GREEN),
    ):
        ax.plot(prof["age"], prof[col], lw=1.7, color=color, label=lab)
    ax.axhline(0, color="#bbbbbb", lw=0.7)
    ax.set_title("(a) Latent physical health by age\n(physical GRM, three diagnosis specifications)",
                 fontsize=9.5)
    ax.set_xlabel("age")
    ax.set_ylabel(r"$\mu_a$ (pooled latent N(0,1))")
    ax.legend(loc="lower left")
    for col, lab in (("mu_P-FUNC", "FUNC"), ("mu_P-FULL", "FULL"),
                     ("mu_P-REC", "REC10")):
        drop = (prof.loc[prof.age == 80, col].iloc[0]
                - prof.loc[prof.age == 30, col].iloc[0])
        print(f"mu(80) - mu(30) {lab:6s} {drop:+.3f}")

    # (b) staleness of the ever-stock by age
    ax = axes[1]
    seen = chron[chron["inventory_seen"] & chron["age"].between(20, 90)]
    recs = []
    for i in CODES:
        s = seen.loc[seen[f"ever_{i}"] == 1, ["age", f"diagage_{i}"]]
        recs.append(pd.DataFrame({
            "age": s["age"], "since": s["age"] - s[f"diagage_{i}"]}))
    since = pd.concat(recs)
    g = since.groupby("age")["since"]
    share10 = g.apply(lambda x: (x > 10).mean())
    share20 = g.apply(lambda x: (x > 20).mean())
    ax.plot(share10.index, share10, lw=1.7, color=PURPLE, label="> 10 years old")
    ax.plot(share20.index, share20, lw=1.7, color=ORANGE, label="> 20 years old")
    ax.set_ylim(0, 1)
    ax.set_title("(b) How old the ever-stock is\n(UKHLS chronic-condition diagnoses)", fontsize=9.5)
    ax.set_xlabel("age")
    ax.set_ylabel("share of condition flags")
    ax.legend(loc="upper left")

    # (c) person-level gap vs oldest diagnosis
    ax = axes[2]
    m = scores.dropna(subset=["theta_phys_full", "theta_phys_func"]).copy()
    diag_cols = [f"diagage_{i}" for i in CODES]
    ch = chron[["pidp", "wave", "age"] + diag_cols].copy()
    since_mat = ch["age"].to_numpy()[:, None] - ch[diag_cols].to_numpy()
    all_nan = np.isnan(since_mat).all(axis=1)
    oldest = np.full(len(ch), np.nan)
    oldest[~all_nan] = np.nanmax(since_mat[~all_nan], axis=1)
    ch["oldest"] = oldest
    m = m.merge(ch[["pidp", "wave", "oldest"]], on=["pidp", "wave"], how="left")
    m["gap"] = m["theta_phys_full"] - m["theta_phys_func"]
    m["bin"] = pd.cut(m["oldest"], [0, 5, 10, 15, 20, 30, 80],
                      labels=["0-5", "5-10", "10-15", "15-20", "20-30", "30+"])
    prof_gap = m.groupby("bin", observed=True)["gap"].agg(["mean", "count"])
    healthy = m[m["oldest"].isna()]["gap"]
    ax.axhline(healthy.mean(), color="#999999", lw=1.0, ls="--",
               label=f"no diagnosis ({healthy.mean():+.2f})")
    ax.bar(range(len(prof_gap)), prof_gap["mean"], color=BLUE, width=0.65)
    ax.set_xticks(range(len(prof_gap)), prof_gap.index)
    ax.set_title("(c) P-FULL minus P-FUNC, by age of oldest diagnosis",
                 fontsize=9.5)
    ax.set_xlabel("years since oldest diagnosis")
    ax.set_ylabel(r"$\theta_{full} - \theta_{func}$, mean")
    ax.legend(loc="lower left")
    print("\ngap theta_full - theta_func by oldest-diagnosis bin:")
    print(prof_gap.round(3).to_string())

    fig.tight_layout()
    out = ROOT_DIR / "measuring_health" / "figures" / "fig_ever_sensitivity.png"
    fig.savefig(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
