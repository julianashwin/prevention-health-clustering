"""Fit the GRM physical-health measure and validate against the EIT-note fit.

Replicates AnalysisForEIT emp_09_grm.R with the package's own-code estimator
(measures/grm.py) on this repo's item extract. Hard gates below are the
August 2026 reference fit's figures; ``--reference-dir`` (CSVs extracted from
grm.rds) additionally compares per-person-wave theta, the TCC grid, and the
age-by-age latent distributions.

Outputs (licensed-data derivatives, gitignored):
  data/processed/measures/grm_scores.parquet   pidp, wave, age, testlets,
                                               grm_theta, grm_theta_sd, grmh
  data/processed/measures/grm_items.csv        item parameters
  data/processed/measures/grm_age_profile.csv  mu_a, sigma_a by single year

Archived: the first graded-response measure, superseded by
data_cleaning/04_build_health.py. Kept runnable: grm_scores.parquet is still
read by several descriptives, and this script is the EIT-note replication.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from prevention_health_clustering.cli_utils import ensure_runtime_directories
from prevention_health_clustering.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from prevention_health_clustering.measures.grm import (
    NCAT,
    TESTLETS,
    build_testlets,
    fit_grm,
    grmh_from_theta,
    score_eap,
    tcc,
    yens_q3,
)

# The reference fit (grm.rds, 21 Aug 2026), extracted via Rscript.
REF_N_ROWS = 498_424
REF_N_PEOPLE = 84_759
REF_A = {"GH": 1.934749, "PF": 3.540075, "RP": 3.687029, "BP": 2.375283}
REF_LOGL = -2_278_354.990593
REF_THETA_RANGE = (-2.6100, 1.3831)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, default=None,
                        help="CSVs extracted from grm.rds for full comparison")
    parser.add_argument("--skip-multigroup", action="store_true")
    args = parser.parse_args(argv)
    ensure_runtime_directories()
    checks: list[tuple[str, bool, str]] = []

    items_path = INTERIM_DATA_DIR / "sf12_items_long.parquet"
    if not items_path.exists():
        print("run data_cleaning/02_build_measures.py first (item extract missing)")
        return 1
    items = pd.read_parquet(items_path)
    d = build_testlets(items)
    print(f"calibration sample: {len(d):,} person-years, "
          f"{d['pidp'].nunique():,} people")
    checks.append(("calibration rows match reference",
                   len(d) == REF_N_ROWS, f"{len(d):,} vs {REF_N_ROWS:,}"))
    checks.append(("calibration people match reference",
                   d["pidp"].nunique() == REF_N_PEOPLE,
                   f"{d['pidp'].nunique():,} vs {REF_N_PEOPLE:,}"))

    cols = list(TESTLETS)
    pat = d[cols].value_counts().reset_index(name="N")
    print(f"distinct response patterns: {len(pat):,} of "
          f"{np.prod([NCAT[c] for c in cols]):,} possible")

    # -- pooled fit ----------------------------------------------------------
    t0 = time.time()
    fit = fit_grm(pat[cols].to_numpy(), pat["N"].to_numpy(float),
                  [NCAT[c] for c in cols], cols, verbose=True)
    print(f"pooled fit: {fit.n_iter} EM iterations, {time.time()-t0:.1f}s, "
          f"logL {fit.log_likelihood:.3f}")
    a_dev = max(abs(fit.items[c][0] - REF_A[c]) for c in cols)
    checks.append(("discriminations match reference", a_dev <= 0.005,
                   f"max |da| {a_dev:.5f}"))
    checks.append((
        "log-likelihood matches reference",
        abs(fit.log_likelihood - REF_LOGL) <= 1.0,
        f"{fit.log_likelihood:.3f} vs {REF_LOGL:.3f}",
    ))
    for c in cols:
        a, b = fit.items[c]
        print(f"  {c:3s} a = {a:5.2f}   b = "
              + " ".join(f"{v:6.2f}" for v in b))

    # -- scores --------------------------------------------------------------
    eap, psd = score_eap(fit, pat[cols].to_numpy())
    pat["grm_theta"], pat["grm_theta_sd"] = eap, psd
    d = d.merge(pat[cols + ["grm_theta", "grm_theta_sd"]], on=cols, how="left")
    d["grmh"] = grmh_from_theta(fit, d["grm_theta"].to_numpy())
    got_range = (d["grm_theta"].min(), d["grm_theta"].max())
    checks.append((
        "theta range matches reference",
        max(abs(got_range[0] - REF_THETA_RANGE[0]),
            abs(got_range[1] - REF_THETA_RANGE[1])) <= 0.005,
        f"[{got_range[0]:.4f}, {got_range[1]:.4f}] vs {REF_THETA_RANGE}",
    ))

    # Q3 off-diagonals are all negative here, and with only J = 4 items that
    # is the expected artifact (bias toward -1/(J-1) = -0.33), not dependence.
    # POSITIVE residual correlation is what would flag it — the +0.38 within
    # the PF pair that motivated the testlets.
    q3 = yens_q3(fit, pat[cols].to_numpy(), pat["N"].to_numpy(float),
                 pat["grm_theta"].to_numpy())
    off = q3[np.triu_indices(4, 1)]
    checks.append(("no positive local dependence", off.max() < 0.2,
                   f"max Q3 {off.max():+.3f}, min {off.min():+.3f}"))

    # -- multi-group by single year of age -----------------------------------
    mg_frame = None
    if not args.skip_multigroup:
        ages = np.arange(20, 91)
        d["gidx"] = d["age"].to_numpy() - 20
        pg = d.groupby(cols + ["gidx"]).size().reset_index(name="N")
        t0 = time.time()
        mg = fit_grm(pg[cols].to_numpy(), pg["N"].to_numpy(float),
                     [NCAT[c] for c in cols], cols,
                     grp=pg["gidx"].to_numpy(), verbose=True)
        print(f"multigroup fit: {mg.n_iter} EM iterations, "
              f"{time.time()-t0:.1f}s, logL {mg.log_likelihood:.3f}")
        mg_frame = pd.DataFrame({"age": ages, "mu": mg.mu, "sigma": mg.sigma})

    # -- optional full reference comparison ----------------------------------
    if args.reference_dir is not None:
        ref_items = pd.read_csv(args.reference_dir / "items.csv")
        b_dev = 0.0
        for _, row in ref_items.iterrows():
            b_ref = np.array([float(x) for x in row["b"].split(";")])
            b_dev = max(b_dev, np.abs(fit.items[row["item"]][1] - b_ref).max())
        checks.append(("thresholds match reference", b_dev <= 0.01,
                       f"max |db| {b_dev:.5f}"))

        ref_q3 = pd.read_csv(args.reference_dir / "q3.csv")
        q3_dev = np.abs(q3 - ref_q3[list(cols)].to_numpy()).max()
        checks.append(("Q3 matrix matches reference", q3_dev <= 0.01,
                       f"max dev {q3_dev:.5f}"))

        ref_tcc = pd.read_csv(args.reference_dir / "tcc.csv")
        tcc_dev = np.abs(tcc(fit, ref_tcc["th"].to_numpy())
                         - ref_tcc["tcc"].to_numpy()).max()
        checks.append(("TCC grid matches reference", tcc_dev <= 1e-3,
                       f"max dev {tcc_dev:.2e}"))

        ref_panel = pd.read_csv(args.reference_dir / "panel_theta.csv")
        m = d.merge(ref_panel[["pidp", "wave", "theta"]],
                    on=["pidp", "wave"], how="inner")
        th_dev = (m["grm_theta"] - m["theta"]).abs().max()
        checks.append((
            "person-wave theta matches reference",
            bool(len(m) == REF_N_ROWS and th_dev <= 0.005),
            f"{len(m):,} rows matched, max |dtheta| {th_dev:.5f}",
        ))

        if mg_frame is not None:
            ref_age = pd.read_csv(args.reference_dir / "age_distributions.csv")
            ma = mg_frame.merge(ref_age, on="age", suffixes=("", "_ref"))
            age_dev = max((ma["mu"] - ma["mu_ref"]).abs().max(),
                          (ma["sigma"] - ma["sigma_ref"]).abs().max())
            checks.append(("age latent distributions match", age_dev <= 0.01,
                           f"max dev {age_dev:.5f}"))

    # -- write ---------------------------------------------------------------
    measures_dir = PROCESSED_DATA_DIR / "measures"
    measures_dir.mkdir(parents=True, exist_ok=True)
    out = d[["pidp", "wave", "age"] + cols
            + ["grm_theta", "grm_theta_sd", "grmh"]]
    out.to_parquet(measures_dir / "grm_scores.parquet", index=False)
    pd.DataFrame(
        [{"item": c, "a": fit.items[c][0],
          **{f"b{i+1}": v for i, v in enumerate(fit.items[c][1])}}
         for c in cols]
    ).to_csv(measures_dir / "grm_items.csv", index=False)
    if mg_frame is not None:
        mg_frame.to_csv(measures_dir / "grm_age_profile.csv", index=False)
    print(f"\nwrote {measures_dir / 'grm_scores.parquet'} "
          f"({len(out):,} rows)")

    print("\n=== checks ===")
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= ok
        print(f"  {'PASS' if ok else 'FAIL':4s}  {name:38s} {detail}")
    print("GRM BUILD OK" if all_ok else "GRM BUILD FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
