"""Build the project's health measure and its reported variants.

The bank, its coding rules and the deficit index are defined in
``measures/health.py``; this script fits, scores, validates and writes. The
graded-response model is fitted pooled, with the latent distribution
identified as N(0, 1), and again multigroup by single year of age to recover
the latent age profile. Scores come from the pooled fit under a pooled prior,
so a theta means the same thing at every age.

Variants written, all monotone in the same answers:
  theta   expected a posteriori latent health, standard normal in the pool
  h       theta through the expected-score curve, 0 to 1
  fi10    the ten-deficit index, 1 = no deficits
  ws      the weighted sum that reproduces h, standardised (appendix twin)

Outputs (gitignored), data/processed/measures/:
  health_measure.parquet        person-wave scores
  health_measure_items.csv      discriminations and thresholds
  health_measure_q3.csv         Yen's Q3 between testlets
  health_measure_age_profile.csv  mu_a and sigma_a by single year of age
  health_measure_weights.csv    the appendix weighted sum

Checks, all gates: no positive local dependence, the EAP identity, the
variants agreeing in rank, and — while the archived search outputs are still
on disk — reproduction of its P-LIM3+CC scores to 1e-9.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from prevention_health_clustering.cli_utils import ensure_runtime_directories
from prevention_health_clustering.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from prevention_health_clustering.measures.grm import (
    fit_grm,
    grmh_from_theta,
    score_eap,
    yens_q3,
)
from prevention_health_clustering.measures.health import (
    HEALTH_ITEMS,
    build_deficit_index,
    build_health_items,
    projection_weights,
)

AGES = np.arange(20, 91)


def main() -> int:
    ensure_runtime_directories()
    checks: list[tuple[str, bool, str]] = []
    out_dir = PROCESSED_DATA_DIR / "measures"
    items = pd.read_parquet(INTERIM_DATA_DIR / "sf12_items_long.parquet")
    chronic = pd.read_parquet(out_dir / "chronic_conditions.parquet")

    bank, ncat = build_health_items(items, chronic)
    names = list(HEALTH_ITEMS)
    print(f"sample: {len(bank):,} person-waves, {bank['pidp'].nunique():,} people, ages "
          f"{bank['age'].min()}-{bank['age'].max()}")
    for c in names:
        share = bank[c].value_counts(normalize=True).sort_index(ascending=False)
        print(f"  {c:9s} {ncat[c]} categories, worst last: "
              + " / ".join(f"{100 * v:.2f}" for v in share.values))

    # ---------------- fit and score ----------------------------------------
    pat = bank[names].value_counts().reset_index(name="N")
    Y, w = pat[names].to_numpy(), pat["N"].to_numpy(float)
    t0 = time.time()
    fit = fit_grm(Y, w, [ncat[c] for c in names], names)
    eap, psd = score_eap(fit, Y)
    q3 = yens_q3(fit, Y, w, eap)
    pat["theta"], pat["theta_sd"] = eap, psd
    scored = bank.merge(pat[names + ["theta", "theta_sd"]], on=names, how="left")
    scored["h"] = grmh_from_theta(fit, scored["theta"].to_numpy())
    print(f"\ngraded-response fit: {len(pat):,} answer patterns, {fit.n_iter} EM iterations, "
          f"{time.time() - t0:.0f}s, log-likelihood {fit.log_likelihood:,.0f}")
    for c in names:
        a, b = fit.items[c]
        print(f"  {c:9s} a = {a:5.2f}   thresholds " + " ".join(f"{v:5.2f}" for v in b))

    # ---------------- the latent age profile --------------------------------
    t0 = time.time()
    g = bank.assign(gidx=bank["age"].to_numpy() - AGES[0])
    pg = g.groupby(names + ["gidx"]).size().reset_index(name="N")
    mg = fit_grm(pg[names].to_numpy(), pg["N"].to_numpy(float), [ncat[c] for c in names], names,
                 grp=pg["gidx"].to_numpy())
    print(f"multigroup by single year of age: {mg.n_iter} iterations, {time.time() - t0:.0f}s; "
          f"mu(80) - mu(30) = {mg.mu[60] - mg.mu[10]:+.3f}")
    profile = pd.DataFrame({"age": AGES, "mu": mg.mu, "sigma": mg.sigma})

    # ---------------- the reported variants ---------------------------------
    fi = build_deficit_index(items, chronic)
    scored = scored.merge(fi, on=["pidp", "wave"], how="left")
    weights, r2 = projection_weights(scored, scored["h"].to_numpy())
    X = scored[names].to_numpy(float)
    ws = ((X - X.mean(axis=0)) / X.std(axis=0)) @ weights.to_numpy()
    scored["ws"] = (ws - ws.mean()) / ws.std()
    print("\nthe appendix weighted sum: h regressed on the standardised codes, "
          f"R-squared {r2:.3f}")
    print("  " + "  ".join(f"{k} {100 * v:.0f}%" for k, v in weights.items()))
    r_h_fi = spearmanr(scored["h"], scored["fi10"]).statistic
    r_h_ws = spearmanr(scored["h"], scored["ws"]).statistic
    print(f"rank correlation with h: ten-deficit index {r_h_fi:.3f}, weighted sum {r_h_ws:.3f}")

    # ---------------- checks -------------------------------------------------
    off = q3[np.triu_indices(len(names), 1)]
    identity = scored["theta"].var(ddof=0) + (scored["theta_sd"] ** 2).mean()
    checks.append(("no positive local dependence", off.max() < 0.2, f"max Q3 {off.max():+.3f}"))
    checks.append(("EAP identity", abs(identity - 1) < 0.05, f"{identity:.3f}"))
    checks.append(("h is monotone in theta", bool(
        (np.diff(scored.sort_values("theta")["h"].to_numpy()) >= -1e-12).all()), "by construction"))
    checks.append(("the variants agree in rank", min(r_h_fi, r_h_ws) > 0.95,
                   f"fi10 {r_h_fi:.3f}, ws {r_h_ws:.3f}"))
    checks.append(("no deficit index missing on the fitted sample",
                   bool(scored["fi10"].notna().all()), f"{scored['fi10'].isna().sum()} missing"))
    archived = out_dir / "limitation_scores.parquet"
    if archived.exists():
        old = pd.read_parquet(archived, columns=["pidp", "wave", "theta_plim3cc_grm", "h_plim3cc_grm"]).dropna()
        m = scored.merge(old, on=["pidp", "wave"], how="inner")
        gap = max((m["theta"] - m["theta_plim3cc_grm"]).abs().max(),
                  (m["h"] - m["h_plim3cc_grm"]).abs().max())
        checks.append(("reproduces the search's P-LIM3+CC scores",
                       len(m) == len(scored) and gap < 1e-9, f"{len(m):,} rows, max |diff| {gap:.1e}"))

    # ---------------- write --------------------------------------------------
    cols = ["pidp", "wave", "age", "theta", "theta_sd", "h", "fi10", "ws"]
    scored[cols].to_parquet(out_dir / "health_measure.parquet", index=False)
    pd.DataFrame([{"item": c, "categories": ncat[c], "a": fit.items[c][0],
                   **{f"b{i + 1}": v for i, v in enumerate(fit.items[c][1])}} for c in names]
                 ).to_csv(out_dir / "health_measure_items.csv", index=False)
    pd.DataFrame([{"item_a": names[i], "item_b": names[j], "q3": q3[i, j]}
                  for i in range(len(names)) for j in range(i + 1, len(names))]
                 ).to_csv(out_dir / "health_measure_q3.csv", index=False)
    profile.to_csv(out_dir / "health_measure_age_profile.csv", index=False)
    weights.rename("weight").rename_axis("item").reset_index().assign(r_squared=r2).to_csv(
        out_dir / "health_measure_weights.csv", index=False)
    print(f"\nwrote health_measure.parquet ({len(scored):,} rows) and four csv files")

    print("\n=== checks ===")
    ok_all = True
    for name, ok, detail in checks:
        ok_all &= ok
        print(f"  {'PASS' if ok else 'FAIL':4s}  {name:44s} {detail}")
    print("HEALTH MEASURE OK" if ok_all else "HEALTH MEASURE FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
