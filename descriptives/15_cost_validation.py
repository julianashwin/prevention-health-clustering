"""Stage 3: does the cost index behave like real NHS spending?

The index is built from self-reported utilisation priced at indicative unit
costs, so its LEVEL cannot be trusted. What can be checked is whether its
SHAPE reproduces things independently known about NHS spending. If it misses
the age profile it cannot be trusted on the health profile, which is what we
want it for.

Three external checks, plus a diagnosis of the one that strains:

  age profile    NHS spend per head is flat to about 50 and rises steeply
                 after 70, with published all-care estimates putting 85+ near
                 seven times a 50-year-old
  concentration  the top decile of spenders accounts for more than half of
                 total expenditure
  level          per-capita spend, where we expect to UNDERSHOOT badly

The instrument is not uniform across the three tiers, and that turns out to
drive the result: in-patient nights (hospd) are a true count, 0-365, while
GP and out-patient contacts are banded with a top category of "more than
ten". Only the uncensored tier can express intensity, so we report the age
gradient tier by tier and run a sensitivity on the top-band value.

Outputs: measuring_health/figures/fig_cost_validation.png,
         artifacts/descriptives/cost_validation.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _style import BLUE, GREEN, INK2, ORANGE, VERM, apply_style  # noqa: E402

from prevention_health_clustering.config import (
    ARTIFACTS_DIR, PROCESSED_DATA_DIR, ROOT_DIR)
from prevention_health_clustering.measures.cost import build_cost_index

COST = "flat_cost_total"
BANDS = ["20s", "30s", "40s", "50s", "60s", "70s", "80+"]
EDGES = [19, 29, 39, 49, 59, 69, 79, 90]
# Published all-care ratio of 85+ to 50-year-old spend. Ours omits
# prescribing, community and social care, the most age-skewed components,
# so this is an upper bound on what the index should reproduce.
PUBLISHED_AGE_RATIO = 7.0
PUBLISHED_TOP_DECILE = 0.50


def age_profile(d, col):
    m = d.groupby("ageband", observed=True)[col].mean()
    return m / m["50s"]


def main() -> int:
    apply_style()
    d = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "cost_index.parquet")
    d = d[d[COST].notna() & d["age"].notna()].copy()
    d["age"] = d["age"].astype(int)
    d = d[d["age"].between(20, 90)]
    d["ageband"] = pd.cut(d["age"], EDGES, labels=BANDS)
    print(f"{len(d):,} person-waves, {d['pidp'].nunique():,} people; "
          f"mean index £{d[COST].mean():,.0f}")

    rows = []

    # ---- 1. age profile, whole index and tier by tier ----------------------
    tiers = {"in-patient": "flat_cost_inpatient",
             "out-patient": "flat_cost_outpatient",
             "GP": "flat_cost_gp"}
    prof = pd.DataFrame({"total": age_profile(d, COST),
                         **{k: age_profile(d, v) for k, v in tiers.items()}})
    levels = pd.Series({k: d[v].mean() for k, v in tiers.items()})
    print("\nage profile, relative to the 50s:")
    print(prof.round(2).to_string())
    print("\nmean £ per person-year by tier:")
    print(levels.round(0).to_string())

    r_tot = prof.loc["80+", "total"]
    r_ip = prof.loc["80+", "in-patient"]
    print(f"\n  whole index, 80+ vs 50s: {r_tot:.1f}x")
    print(f"  in-patient tier only:    {r_ip:.1f}x")
    print(f"  published all-care:      ~{PUBLISHED_AGE_RATIO:.0f}x "
          "(includes prescribing, community and social care, which we omit)")
    rows += [{"check": "age 80+ vs 50s, whole index", "value": r_tot,
              "benchmark": PUBLISHED_AGE_RATIO, "unit": "ratio"},
             {"check": "age 80+ vs 50s, in-patient tier", "value": r_ip,
              "benchmark": PUBLISHED_AGE_RATIO, "unit": "ratio"}]

    # ---- why the flat tiers are flat: top-band censoring -------------------
    cens = pd.DataFrame({
        "GP": d.groupby("ageband", observed=True)["hl2gp"].apply(
            lambda s: (s == 4).mean()),
        "out-patient": d.groupby("ageband", observed=True)["hl2hop"].apply(
            lambda s: (s == 4).mean())})
    print("\nshare of each age band in the top ('more than ten') category:")
    print((cens * 100).round(2).to_string())
    print("  in-patient nights are a true count (0-365), never censored")

    # ---- top-band sensitivity ---------------------------------------------
    print("\nsensitivity to the value assigned to 'more than ten':")
    sens = []
    for top in (12.0, 15.0, 20.0, 30.0):
        alt = build_cost_index(d, top_band=top)
        a = d[["ageband"]].assign(c=alt["cost_total"].to_numpy())
        m = a.groupby("ageband", observed=True)["c"].mean()
        sens.append({"top_band": top, "mean": m.mean(),
                     "ratio_80_50": m["80+"] / m["50s"]})
        print(f"  top band = {top:5.1f} contacts -> 80+/50s = "
              f"{m['80+'] / m['50s']:.2f}x")
    sens = pd.DataFrame(sens)

    # ---- 2. concentration --------------------------------------------------
    c = np.sort(d[COST].to_numpy())[::-1]
    tot = c.sum()
    conc = {p: c[:int(len(c) * p / 100)].sum() / tot for p in (1, 5, 10, 20)}
    print("\nconcentration:")
    for p, v in conc.items():
        print(f"  top {p:2d}% of person-waves hold {v:.1%}")
    print(f"  published: the top decile holds more than "
          f"{PUBLISHED_TOP_DECILE:.0%}")
    zero = (d[COST] <= 0).mean()
    print(f"  {zero:.1%} of person-waves are zero, which inflates our "
          "concentration relative to\n  administrative data that also carries "
          "prescribing and community contacts")
    rows.append({"check": "top 10% share", "value": conc[10],
                 "benchmark": PUBLISHED_TOP_DECILE, "unit": "share"})

    # ---- 3. level ----------------------------------------------------------
    per_capita = d[COST].mean()
    print(f"\nlevel: £{per_capita:,.0f} per person-year, against NHS spend per "
          "head of roughly £3,000.\n  The index covers three tiers of contact "
          "only, so an undershoot of this size is\n  expected and is not "
          "informative about whether the shape is right.")
    rows.append({"check": "per person-year", "value": per_capita,
                 "benchmark": np.nan, "unit": "GBP"})

    # ---- 4. the object of interest: the health gradient --------------------
    d["hdec"] = pd.qcut(d["theta"], 10, labels=False,
                        duplicates="drop")
    g = d.groupby("hdec")[COST].mean()
    gi = d.groupby("hdec")["flat_cost_inpatient"].mean()
    grad, grad_ip = g.iloc[0] / g.iloc[-1], gi.iloc[0] / gi.iloc[-1]
    print(f"\nhealth gradient, least against most healthy decile:")
    print(f"  whole index      £{g.iloc[0]:,.0f} vs £{g.iloc[-1]:,.0f}  "
          f"{grad:.1f}x")
    print(f"  in-patient tier  £{gi.iloc[0]:,.0f} vs £{gi.iloc[-1]:,.0f}  "
          f"{grad_ip:.1f}x")
    print("  the health gradient is far steeper than the age gradient, which "
          "is the point:\n  age is a poor proxy for the health that actually "
          "drives cost")
    rows += [{"check": "health decile 1 vs 10, whole index", "value": grad,
              "benchmark": np.nan, "unit": "ratio"},
             {"check": "health decile 1 vs 10, in-patient", "value": grad_ip,
              "benchmark": np.nan, "unit": "ratio"}]

    out_csv = ARTIFACTS_DIR / "descriptives" / "cost_validation.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0))
    x = np.arange(len(BANDS))

    ax = axes[0, 0]
    for lab, col, cl in [("whole index", "total", INK2),
                         ("in-patient", "in-patient", BLUE),
                         ("out-patient", "out-patient", ORANGE),
                         ("GP", "GP", GREEN)]:
        ax.plot(x, prof[col].reindex(BANDS), "o-", color=cl, lw=2, ms=4,
                label=lab)
    ax.axhline(PUBLISHED_AGE_RATIO, ls="--", color=VERM, lw=1.5)
    ax.text(0.05, PUBLISHED_AGE_RATIO, " published all-care 85+/50 ratio",
            color=VERM, fontsize=7.5, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(BANDS)
    ax.set_ylabel("spend relative to the 50s")
    ax.set_title("(a) Age profile: steep only where the instrument counts",
                 fontsize=10)
    ax.legend(fontsize=7.5, frameon=False); ax.grid(True)

    ax = axes[0, 1]
    w = 0.38
    ax.bar(x - w / 2, cens["GP"].reindex(BANDS) * 100, w, color=GREEN,
           label="GP")
    ax.bar(x + w / 2, cens["out-patient"].reindex(BANDS) * 100, w,
           color=ORANGE, label="out-patient")
    ax.set_xticks(x); ax.set_xticklabels(BANDS)
    ax.set_ylabel("% of the age band in the top band")
    ax.set_ylim(0, 13.0)
    ax.set_title("(b) Censoring rises with age but cannot explain (a)",
                 fontsize=10)
    lo, hi = sens["ratio_80_50"].iloc[-1], sens["ratio_80_50"].iloc[1]
    ax.text(0.30, 0.985,
            f"scoring the top band at 30 contacts instead of 15 moves\n"
            f"the whole-index ratio from {hi:.2f}x to {lo:.2f}x \u2014 the wrong way,\n"
            "because it loads more weight onto the two flat tiers",
            transform=ax.transAxes, fontsize=7.6, color=INK2, va="top")
    ax.legend(fontsize=7.5, frameon=False); ax.grid(True, axis="y")

    ax = axes[1, 0]
    share = np.cumsum(c) / tot
    pct = np.arange(1, len(c) + 1) / len(c) * 100
    ax.plot(pct, share, color=VERM, lw=2)
    ax.axvline(10, ls=":", color="#888888")
    ax.axhline(PUBLISHED_TOP_DECILE, ls=":", color="#888888")
    ax.plot([10], [conc[10]], "o", color=BLUE, ms=7)
    ax.annotate(f"top 10% hold {conc[10]:.0%}\npublished: over 50%",
                (10, conc[10]), textcoords="offset points", xytext=(16, -30),
                fontsize=8)
    ax.set_xlabel("percentile of the index, highest first")
    ax.set_ylabel("cumulative share of the total")
    ax.set_title("(c) Concentration clears the published bar", fontsize=10)
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.02); ax.grid(True)

    ax = axes[1, 1]
    ax.plot(range(1, 11), g.values, "o-", color=VERM, lw=2, ms=5,
            label=f"whole index, {grad:.0f}x")
    ax.plot(range(1, 11), gi.values, "s--", color=BLUE, lw=1.8, ms=4,
            label=f"in-patient tier, {grad_ip:.0f}x")
    ax.set_xlabel("decile of physical health (1 = least healthy)")
    ax.set_ylabel("index, £ per person-year")
    ax.set_title("(d) The health gradient dwarfs the age gradient",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(True)

    fig.suptitle("Stage 3: validating the UKHLS cost proxy against what is "
                 "known about NHS spending", fontweight="bold", y=0.995)
    fig.text(0.01, -0.005,
             "Flat-rate index, waves 7-15, ages 20-90: in-patient spells and excess bed days, out-patient\n"
             "attendances and GP visits at national average unit costs, maternity excluded. In-patient nights\n"
             "(hospd) are a true count and are never censored; GP (hl2gp) and out-patient (hl2hop) contacts are\n"
             "banded, with the top band scored at 15 contacts. The level is not meant to be right; the shape is what\n"
             "the external checks test, and the binding limitation is the flat unit cost per contact, not the bands.",
             fontsize=7.4, color=INK2, va="top")
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    out = ROOT_DIR / "measuring_health" / "figures" / "fig_cost_validation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
