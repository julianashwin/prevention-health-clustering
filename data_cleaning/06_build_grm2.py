"""Fit the two-dimensional GRM (physical + mental) with ever-variable
sensitivities.

Physical specifications, all sharing the four validated SF-12 testlets:
  P-FUNC   + the disdif functioning testlet (no diagnoses)
  P-FULL   + the six ever-diagnosis group items
  P-REC    as P-FULL but conditions diagnosed within the last 10 years only

Mental: MH/RE/SF testlets + 12 GHQ items + ever-depression. GHQ enters
item-level; if Yen's Q3 shows positive within-wording dependence (method
effects), the pre-specified remedy collapses the GHQ into two wording
testlets — the same cure emp_09 applied to the PF/RP pairs.

Each final spec gets the multigroup fit by single year of age (mu_a, sigma_a),
which is where the ever-accumulation question is answered: how much steeper is
the latent age profile once diagnoses enter, and how much of that steepness
is old diagnoses (P-FULL vs P-REC vs P-FUNC).

Also runs the note's invariance test on P-FULL: item parameters freed across
three age bands (20-45, 45-65, 65-91), latent distributions free throughout.

Outputs (gitignored): data/processed/measures/grm2_scores.parquet,
grm2_items.csv, grm2_age_profiles.csv, grm2_q3.csv.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from prevention_health_clustering.cli_utils import ensure_runtime_directories
from prevention_health_clustering.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from prevention_health_clustering.measures.grm import (
    fit_grm,
    grmh_from_theta,
    score_eap,
    tcc,
    yens_q3,
)
from prevention_health_clustering.measures.grm2 import (
    COMBINED_NCAT,
    GHQ_NEGATIVE,
    GHQ_POSITIVE,
    MENT_ITEMS,
    MENT_NCAT,
    MENT_TESTLET_NCAT,
    PHYS_CONDITION_ITEMS,
    PHYS_NCAT,
    PHYS_TIMED_NCAT,
    build_combined_items,
    build_mental_items,
    build_physical_items,
    build_physical_timed_items,
    collapse_ghq_wording,
)

AGES = np.arange(20, 91)


def fit_spec(name, frame, names, ncat_map, *, multigroup=True, verbose=False):
    """Pooled fit, EAP scores, Q3, and optionally the age multigroup fit."""
    cols = list(names)
    pat = frame[cols].value_counts().reset_index(name="N")
    ncat = [ncat_map[c] for c in cols]
    t0 = time.time()
    fit = fit_grm(pat[cols].to_numpy(), pat["N"].to_numpy(float), ncat, cols,
                  verbose=verbose)
    eap, psd = score_eap(fit, pat[cols].to_numpy())
    pat["theta"], pat["theta_sd"] = eap, psd
    scored = frame.merge(pat[cols + ["theta", "theta_sd"]], on=cols, how="left")
    q3 = yens_q3(fit, pat[cols].to_numpy(), pat["N"].to_numpy(float),
                 pat["theta"].to_numpy())
    identity = scored["theta"].var(ddof=0) + (scored["theta_sd"] ** 2).mean()
    print(f"[{name}] n={len(frame):,} rows, {len(pat):,} patterns, "
          f"{fit.n_iter} EM iters, {time.time()-t0:.0f}s, "
          f"logL {fit.log_likelihood:.0f}, EAP identity {identity:.3f}")
    for c in cols:
        a, b = fit.items[c]
        print(f"    {c:8s} a = {a:5.2f}   b = "
              + " ".join(f"{v:5.2f}" for v in b[: min(len(b), 8)]))
    mg = None
    if multigroup:
        t0 = time.time()
        g = frame.copy()
        g["gidx"] = g["age"].to_numpy() - 20
        pg = g.groupby(cols + ["gidx"]).size().reset_index(name="N")
        mg = fit_grm(pg[cols].to_numpy(), pg["N"].to_numpy(float), ncat, cols,
                     grp=pg["gidx"].to_numpy())
        print(f"    multigroup: {mg.n_iter} iters, {time.time()-t0:.0f}s, "
              f"logL {mg.log_likelihood:.0f}")
    return fit, scored, q3, mg, identity


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-invariance", action="store_true")
    args = parser.parse_args(argv)
    ensure_runtime_directories()
    checks: list[tuple[str, bool, str]] = []

    items = pd.read_parquet(INTERIM_DATA_DIR / "sf12_items_long.parquet")
    chronic = pd.read_parquet(
        PROCESSED_DATA_DIR / "measures" / "chronic_conditions.parquet")

    # ---------------- physical banks ----------------------------------------
    p_func = build_physical_items(items, conditions=None)
    p_full = build_physical_items(items, chronic, conditions="ever")
    p_rec = build_physical_items(items, chronic, conditions="recent10")
    print(f"physical banks: FUNC {len(p_func):,}  FULL {len(p_full):,}  "
          f"REC10 {len(p_rec):,} person-years")
    # The condition items exist only from a person's first diagnosis
    # inventory onward (76% of person-waves), so P-FULL necessarily loses the
    # pre-inventory waves; the gate protects against losses beyond that.
    checks.append((
        "P-FULL keeps the inventory-covered sample",
        len(p_full) >= 0.75 * len(p_func),
        f"{len(p_full) / len(p_func):.1%} of P-FUNC rows",
    ))

    fits, scores, q3s, mgs, ids = {}, {}, {}, {}, {}
    specs = [
        ("P-FUNC", p_func, list(PHYS_NCAT)[:5]),
        ("P-FULL", p_full, list(PHYS_NCAT)),
        ("P-REC", p_rec, list(PHYS_NCAT)),
    ]
    for name, frame, cols in specs:
        fits[name], scores[name], q3s[name], mgs[name], ids[name] = fit_spec(
            name, frame, cols, PHYS_NCAT)
        off = q3s[name][np.triu_indices(len(cols), 1)]
        checks.append((f"{name} no positive local dependence",
                       off.max() < 0.2, f"max Q3 {off.max():+.3f}"))
        checks.append((f"{name} EAP identity", abs(ids[name] - 1.0) < 0.05,
                       f"{ids[name]:.3f}"))

    # ---------------- mental bank -------------------------------------------
    ment = build_mental_items(items, chronic)
    print(f"\nmental bank: {len(ment):,} person-years")
    m_fit, m_scored, m_q3, _, m_id = fit_spec(
        "M-ITEM", ment, list(MENT_ITEMS), MENT_NCAT, multigroup=False)
    names = list(MENT_ITEMS)
    qdf = pd.DataFrame(m_q3, index=names, columns=names)
    within = []
    for grpn in (GHQ_POSITIVE, GHQ_NEGATIVE):
        for i, a in enumerate(grpn):
            for b in grpn[i + 1:]:
                within.append(qdf.loc[a, b])
    within = np.array(within)
    print(f"    GHQ within-wording Q3: mean {within.mean():+.3f}, "
          f"max {within.max():+.3f}, positive share {(within > 0).mean():.0%}")
    use_testlets = bool(within.max() > 0.2 or within.mean() > 0.05)
    print(f"    wording testlets: {'YES' if use_testlets else 'no'} "
          "(pre-specified rule: max within-wording Q3 > 0.2 "
          "or mean > 0.05)")
    if use_testlets:
        ment_final = collapse_ghq_wording(ment)
        m_names = list(MENT_TESTLET_NCAT)
        m_ncat = MENT_TESTLET_NCAT
    else:
        ment_final, m_names, m_ncat = ment, list(MENT_ITEMS), MENT_NCAT
    fits["MENT"], scores["MENT"], q3s["MENT"], mgs["MENT"], ids["MENT"] = (
        fit_spec("MENT", ment_final, m_names, m_ncat))
    off = q3s["MENT"][np.triu_indices(len(m_names), 1)]
    checks.append(("MENT no positive local dependence", off.max() < 0.2,
                   f"max Q3 {off.max():+.3f}"))
    checks.append(("MENT EAP identity", abs(ids["MENT"] - 1.0) < 0.05,
                   f"{ids['MENT']:.3f}"))

    # ---------------- P-TIMED: the estimated recency weighting --------------
    p_timed = build_physical_timed_items(items, chronic)
    fits["P-TIMED"], scores["P-TIMED"], q3s["P-TIMED"], _, ids["P-TIMED"] = (
        fit_spec("P-TIMED", p_timed, list(PHYS_TIMED_NCAT), PHYS_TIMED_NCAT,
                 multigroup=False))
    print("\n    estimated recency weighting (discrimination of recent vs "
          "stale diagnoses):")
    for g in PHYS_CONDITION_ITEMS:
        ar = fits["P-TIMED"].items[f"{g}R"][0]
        as_ = fits["P-TIMED"].items[f"{g}S"][0]
        print(f"      {g:7s} recent a = {ar:5.2f}   stale a = {as_:5.2f}   "
              f"ratio {ar / as_ if as_ > 0 else float('inf'):5.2f}")
    checks.append(("P-TIMED EAP identity", abs(ids["P-TIMED"] - 1.0) < 0.05,
                   f"{ids['P-TIMED']:.3f}"))

    # ---------------- COMBINED: every good-coverage item --------------------
    combined = build_combined_items(p_full, ment_final if use_testlets
                                    else collapse_ghq_wording(ment))
    fits["COMBINED"], scores["COMBINED"], q3s["COMBINED"], mgs["COMBINED"], \
        ids["COMBINED"] = fit_spec(
            "COMBINED", combined, list(COMBINED_NCAT), COMBINED_NCAT)
    # A single dimension over genuinely two-dimensional data leaves the
    # minority dimension's shared variance in the residuals: the mental items
    # cluster positively (MH-GHQNEG etc.). That is the RECORDED FINDING that
    # justifies the two-bank design, not a bug to gate away. The gate instead
    # requires the physical side to stay clean within the combined bank, and
    # prints the offending mental pairs for the note.
    names_c = list(COMBINED_NCAT)
    ment_side = {"MH", "RE", "SF", "GHQPOS", "GHQNEG", "DEPR"}
    q3c = q3s["COMBINED"]
    worst_pairs = []
    phys_max = -1.0
    for i, a in enumerate(names_c):
        for j in range(i + 1, len(names_c)):
            b = names_c[j]
            v = q3c[i, j]
            if a in ment_side or b in ment_side:
                worst_pairs.append((v, a, b))
            else:
                phys_max = max(phys_max, v)
    worst_pairs.sort(reverse=True)
    print("    combined-bank residual clustering (evidence of two dimensions):")
    for v, a, b in worst_pairs[:4]:
        print(f"      Q3({a}, {b}) = {v:+.3f}")
    checks.append(("COMBINED physical side locally independent",
                   phys_max < 0.25, f"max phys-phys Q3 {phys_max:+.3f}"))
    checks.append(("COMBINED mental residuals cluster (two-dim evidence)",
                   worst_pairs[0][0] > 0.2,
                   f"max mental-pair Q3 {worst_pairs[0][0]:+.3f}"))
    checks.append(("COMBINED EAP identity", abs(ids["COMBINED"] - 1.0) < 0.05,
                   f"{ids['COMBINED']:.3f}"))

    # ---------------- invariance test on P-FULL -----------------------------
    if not args.skip_invariance:
        print("\ninvariance: item parameters freed across three age bands")
        cols = list(PHYS_NCAT)
        mg_ll, mg_np = mgs["P-FULL"].log_likelihood, mgs["P-FULL"].n_parameters
        free_ll, free_np = 0.0, 0
        for lo, hi in ((20, 44), (45, 64), (65, 90)):
            s = p_full[p_full["age"].between(lo, hi)].copy()
            uages = np.sort(s["age"].unique())
            s["g2"] = s["age"].map({a: i for i, a in enumerate(uages)})
            pg = s.groupby(cols + ["g2"]).size().reset_index(name="N")
            f2 = fit_grm(pg[cols].to_numpy(), pg["N"].to_numpy(float),
                         [PHYS_NCAT[c] for c in cols], cols,
                         grp=pg["g2"].to_numpy())
            free_ll += f2.log_likelihood
            free_np += f2.n_parameters
            print(f"    band {lo}-{hi}: a = "
                  + " ".join(f"{f2.items[c][0]:5.2f}" for c in cols))
        dof = free_np - mg_np
        stat = 2 * (free_ll - mg_ll)
        print(f"    2*dlogL = {stat:.1f} on {dof} df "
              f"(per-df {stat / dof:.1f})")

    # ---------------- age-profile sensitivity: the ever question ------------
    print("\nlatent age profiles mu_a (the ever-accumulation check):")
    prof = pd.DataFrame({"age": AGES})
    for name in ("P-FUNC", "P-FULL", "P-REC", "MENT", "COMBINED"):
        prof[f"mu_{name}"] = mgs[name].mu
        prof[f"sigma_{name}"] = mgs[name].sigma
    for name in ("P-FUNC", "P-FULL", "P-REC"):
        drop = prof.loc[prof.age == 80, f"mu_{name}"].iloc[0] - \
            prof.loc[prof.age == 30, f"mu_{name}"].iloc[0]
        print(f"    {name:7s} mu(80) - mu(30) = {drop:+.3f}")

    # ---------------- assemble scores ---------------------------------------
    base = scores["P-FUNC"][["pidp", "wave", "age"]].copy()
    base["theta_phys_func"] = scores["P-FUNC"]["theta"]
    base["theta_phys_func_sd"] = scores["P-FUNC"]["theta_sd"]
    for name, col in (("P-FULL", "theta_phys_full"), ("P-REC", "theta_phys_rec10"),
                      ("MENT", "theta_ment"), ("COMBINED", "theta_combined"),
                      ("P-TIMED", "theta_phys_timed")):
        s = scores[name][["pidp", "wave", "theta", "theta_sd"]].rename(
            columns={"theta": col, "theta_sd": f"{col}_sd"})
        base = base.merge(s, on=["pidp", "wave"], how="outer")
    base["grmh_phys_full"] = grmh_from_theta(
        fits["P-FULL"], base["theta_phys_full"].to_numpy())
    base["grmh_phys_func"] = grmh_from_theta(
        fits["P-FUNC"], base["theta_phys_func"].to_numpy())
    base["grmh_ment"] = grmh_from_theta(fits["MENT"], base["theta_ment"].to_numpy())
    base["grmh_combined"] = grmh_from_theta(
        fits["COMBINED"], base["theta_combined"].to_numpy())

    # correlations with the original 4-testlet GRM
    old = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "grm_scores.parquet")
    m = base.merge(old[["pidp", "wave", "grm_theta"]], on=["pidp", "wave"],
                   how="inner")
    for col, lo in (("theta_phys_func", 0.95), ("theta_phys_full", 0.90)):
        r = m[col].corr(m["grm_theta"])
        checks.append((f"{col} tracks the original GRM", r >= lo,
                       f"r = {r:.4f}"))
    r_pm = base["theta_phys_full"].corr(base["theta_ment"])
    print(f"\ncorr(theta_phys_full, theta_ment) = {r_pm:.3f}")

    # ---------------- write -------------------------------------------------
    out_dir = PROCESSED_DATA_DIR / "measures"
    base.to_parquet(out_dir / "grm2_scores.parquet", index=False)
    rows = []
    for name, fit in fits.items():
        for c in fit.items:
            a, b = fit.items[c]
            rows.append({"spec": name, "item": c, "a": a,
                         **{f"b{i+1}": v for i, v in enumerate(b)}})
    pd.DataFrame(rows).to_csv(out_dir / "grm2_items.csv", index=False)
    prof.to_csv(out_dir / "grm2_age_profiles.csv", index=False)
    qrows = []
    for name in q3s:
        cols_n = list(fits[name].items)
        for i, a in enumerate(cols_n):
            for j, b in enumerate(cols_n):
                if j > i:
                    qrows.append({"spec": name, "item_a": a, "item_b": b,
                                  "q3": q3s[name][i, j]})
    pd.DataFrame(qrows).to_csv(out_dir / "grm2_q3.csv", index=False)
    print(f"wrote grm2_scores.parquet ({len(base):,} rows), grm2_items.csv, "
          "grm2_age_profiles.csv, grm2_q3.csv")

    print("\n=== checks ===")
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= ok
        print(f"  {'PASS' if ok else 'FAIL':4s}  {name:44s} {detail}")
    print("GRM2 BUILD OK" if all_ok else "GRM2 BUILD FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
