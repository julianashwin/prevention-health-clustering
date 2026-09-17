"""Build every SF-12-derived health measure and validate against the note.

Ports the sandbox chain 01 -> 03 -> 04 -> 05 -> 11 into one driver over the
package's measures modules. Validation targets are the figures documented in
the PCS construction note:

  * variant A reconstructs sf12pcs_dv: max |error| 0.0054 over 528,485
    complete person-years;
  * the UK norms table (wave 1, self-completion cross-sectional weights);
  * the promax inter-factor figure 0.897 as PRINTED by the sandbox — stored
    alongside the properly normalised correlation, which the sandbox's
    convention overstated (see measures/sf12.py);
  * the 6.6-point mental-health artifact at the median physical profile;
  * the SF-6D worked example / ceiling / floor.

Outputs (licensed-data derivatives, gitignored):
  data/interim/sf12_items_long.parquet     raw item extract (cached)
  data/processed/measures/sf12_measures.parquet
  data/processed/measures/uk_norms.csv, uk_coefficients.csv
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from prevention_health_clustering.cli_utils import ensure_runtime_directories
from prevention_health_clustering.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from prevention_health_clustering.measures.composites import (
    equal_weight_composite,
    farivar_composites,
)
from prevention_health_clustering.measures.items import extract_sf12_items
from prevention_health_clustering.measures.sf12 import (
    PHYS_SCALES,
    SCALES,
    build_subscales,
    derive_uk_structure,
    score_phys_only,
    score_uk_variants,
    score_us,
)
from prevention_health_clustering.measures import sf6d

# The construction note's documented figures, used as validation gates.
NOTE_MAX_ERROR = 0.0054
NOTE_COMPLETE_N = 528_485
NOTE_UK_NORMS = {
    "PF": (84.66, 29.17), "RP": (82.11, 28.09), "BP": (80.60, 29.08),
    "GH": (66.42, 29.34), "VT": (57.11, 26.33), "SF": (86.38, 25.32),
    "RE": (88.61, 21.25), "MH": (71.66, 20.87),
}
NOTE_PHI_AS_PRINTED = 0.897
NOTE_MH_ARTIFACT = 6.6


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-items", action="store_true",
                        help="re-extract items from the raw tab files")
    args = parser.parse_args(argv)
    ensure_runtime_directories()
    checks: list[tuple[str, bool, str]] = []

    # -- items ---------------------------------------------------------------
    cache = INTERIM_DATA_DIR / "sf12_items_long.parquet"
    if cache.exists() and not args.refresh_items:
        items = pd.read_parquet(cache)
        print(f"items: cached {cache} ({len(items):,} person-years)")
    else:
        print("items: extracting from raw indresp waves")
        items = extract_sf12_items()
        items.to_parquet(cache, index=False)
        print(f"items: wrote {cache} ({len(items):,} person-years)")

    item_cols = ["sf1", "sf2a", "sf2b", "sf3a", "sf3b", "sf4a",
                 "sf4b", "sf5", "sf6a", "sf6b", "sf6c", "sf7"]
    complete = items[item_cols].notna().all(axis=1)
    print(f"complete SF-12 responses: {complete.sum():,} of {len(items):,}")

    # -- subscales and variant A (US reconstruction) -------------------------
    sub = build_subscales(items)
    us = score_us(sub)
    both = complete & items["sf12pcs_dv"].notna() & items["sf12mcs_dv"].notna()
    err_p = (us.loc[both, "PCS_us"] - items.loc[both, "sf12pcs_dv"]).abs()
    checks.append((
        "A reconstructs sf12pcs_dv",
        err_p.max() <= NOTE_MAX_ERROR + 5e-4,
        f"max|err| {err_p.max():.4f} (note {NOTE_MAX_ERROR})",
    ))
    # The published scores are floored at 0: 32 person-years have
    # sf12mcs_dv == 0.0 where the algorithm goes slightly negative. With the
    # floor applied, BOTH channels reconstruct to the same precision.
    err_m = (us.loc[both, "MCS_us"].clip(lower=0)
             - items.loc[both, "sf12mcs_dv"]).abs()
    n_floor = int((us.loc[both, "MCS_us"] < 0).sum())
    checks.append((
        "A + zero floor reconstructs sf12mcs_dv",
        err_m.max() <= NOTE_MAX_ERROR + 5e-4,
        f"max|err| {err_m.max():.4f} ({n_floor} floored person-years)",
    ))
    checks.append((
        "complete person-years match note",
        int(both.sum()) == NOTE_COMPLETE_N,
        f"{both.sum():,} vs {NOTE_COMPLETE_N:,}",
    ))

    # -- UK structure from the wave-1 reference sample -----------------------
    ref_mask = (items["wave"] == 1) & complete & (items["indscus_xw"] > 0)
    ref_sub = sub.loc[ref_mask]
    ref_w = items.loc[ref_mask, "indscus_xw"]
    print(f"UK reference sample (wave 1, positive weight): {ref_mask.sum():,}")
    structure = derive_uk_structure(ref_sub, ref_w)

    print(f"\n{'scale':6} {'UK mean':>9} {'UK SD':>8} {'note mean':>10} {'note SD':>8}")
    norm_dev = 0.0
    for s in SCALES:
        m, sd = structure.norms[s]
        nm, nsd = NOTE_UK_NORMS[s]
        norm_dev = max(norm_dev, abs(m - nm), abs(sd - nsd))
        print(f"{s:6} {m:9.2f} {sd:8.2f} {nm:10.2f} {nsd:8.2f}")
    checks.append(("UK norms match note table", norm_dev <= 0.011,
                   f"max dev {norm_dev:.4f}"))

    checks.append((
        "phi as printed reproduces note",
        abs(structure.phi_raw - NOTE_PHI_AS_PRINTED) <= 0.01,
        f"raw {structure.phi_raw:.4f} vs note {NOTE_PHI_AS_PRINTED}",
    ))
    print(f"\npromax inter-factor: as-printed (unnormalised) "
          f"{structure.phi_raw:.4f}; proper correlation {structure.phi:.4f}")
    print("  published comparators: Farivar 0.62, Tucker 0.71-0.73")

    # -- variants B-E --------------------------------------------------------
    uk = score_uk_variants(sub, structure, ref_sub, ref_w)
    phys = score_phys_only(sub, structure, anchor_mask=both)

    # -- the mental-health artifact at the median physical profile -----------
    cc = pd.concat(
        [sub, us, items[["sf12pcs_dv", "sf12mcs_dv"]]], axis=1
    ).loc[both]
    med = cc[list(PHYS_SCALES)].median()
    at_median = np.ones(len(cc), dtype=bool)
    for s in PHYS_SCALES:
        at_median &= (cc[s] - med[s]).abs().to_numpy() < 0.01
    profile = cc[at_median]
    lo = profile[profile["MH"] <= profile["MH"].quantile(0.10)]
    hi = profile[profile["MH"] >= profile["MH"].quantile(0.90)]
    artifact = lo["PCS_us"].mean() - hi["PCS_us"].mean()
    checks.append((
        "6.6-point MH artifact reproduces",
        abs(artifact - NOTE_MH_ARTIFACT) <= 0.15,
        f"{artifact:+.2f} PCS points (n={at_median.sum():,} at median profile)",
    ))

    # -- SF-6D ---------------------------------------------------------------
    for name, ok, got in sf6d.validate_tariff():
        checks.append((f"SF-6D {name}", ok, f"{got:.3f}"))
    states = sf6d.classify(items)
    utility = sf6d.utility(states).where(complete)
    state_label = sf6d.state_string(states, complete)
    n_levels_ok = all(
        states.loc[complete, d].nunique() == sf6d.LEVELS[d] for d in sf6d.DIMS
    )
    checks.append(("SF-6D all dimension levels observed", n_levels_ok,
                   f"{ {d: int(states.loc[complete, d].nunique()) for d in sf6d.DIMS} }"))
    print(f"\nSF-6D utility: n={utility.notna().sum():,} mean={utility.mean():.4f} "
          f"sd={utility.std():.4f} ceiling={(utility == 1.0).mean():.1%} "
          f"floor={(utility <= 0.3451).sum():,}")
    print("  (external benchmark: UKHLS means 0.81 men / 0.79 women)")

    # -- composites ----------------------------------------------------------
    far = farivar_composites(sub, structure.norms)
    equal8 = equal_weight_composite(sub)

    # -- assemble and write --------------------------------------------------
    out = pd.concat(
        [
            items[["pidp", "wave", "wave_letter", "age",
                   "sf12pcs_dv", "sf12mcs_dv"] + item_cols],
            sub.add_prefix("sub_"),
            us, uk, phys.rename("PCS_phys_only"),
            states.add_prefix("sf6d_").where(complete),
            state_label.rename("sf6d_state"),
            utility.rename("sf6d_utility"),
            far, equal8.rename("HEALTH_equal8"),
        ],
        axis=1,
    ).sort_values(["pidp", "wave"]).reset_index(drop=True)

    measures_dir = PROCESSED_DATA_DIR / "measures"
    measures_dir.mkdir(parents=True, exist_ok=True)
    out_path = measures_dir / "sf12_measures.parquet"
    out.to_parquet(out_path, index=False)

    pd.DataFrame(
        [{"scale": s, "uk_mean": structure.norms[s][0],
          "uk_sd": structure.norms[s][1]} for s in SCALES]
    ).to_csv(measures_dir / "uk_norms.csv", index=False)
    pd.DataFrame(
        [{"scale": s,
          "uk_varimax_pcs": structure.coef_varimax[i, 0],
          "uk_varimax_mcs": structure.coef_varimax[i, 1],
          "uk_promax_pcs": structure.coef_promax[i, 0],
          "uk_promax_mcs": structure.coef_promax[i, 1],
          "load_vmax_phys": structure.loadings_varimax[i, 0],
          "load_vmax_ment": structure.loadings_varimax[i, 1]}
         for i, s in enumerate(SCALES)]
    ).to_csv(measures_dir / "uk_coefficients.csv", index=False)
    print(f"\nwrote {out_path} ({len(out):,} rows x {len(out.columns)} cols)")

    # -- summary tables mirroring the note -----------------------------------
    single = ["sf12pcs_dv", "PCS_us", "PCS_uk_norm", "PCS_uk_varimax",
              "PCS_uk_promax", "PCS_phys_only", "sf6d_utility",
              "PCS_c_farivar", "MCS_c_farivar", "HEALTH_equal8"]
    print("\nPCS variants over all person-years (note: promax mean 50.47, "
          "sd 9.74, r 0.9522):")
    stats = out[single[:6]].describe().T[["mean", "std"]]
    stats["r_vs_dv"] = [out[c].corr(out["sf12pcs_dv"]) for c in single[:6]]
    print(stats.round(4).to_string())
    print(f"\ncorr(PCS_c, MCS_c) Farivar oblique: "
          f"{out['PCS_c_farivar'].corr(out['MCS_c_farivar']):.3f} "
          f"(Farivar et al. report 0.74)")

    print("\n=== checks ===")
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= ok
        print(f"  {'PASS' if ok else 'FAIL':4s}  {name:38s} {detail}")
    print("MEASURES BUILD OK" if all_ok else "MEASURES BUILD FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
