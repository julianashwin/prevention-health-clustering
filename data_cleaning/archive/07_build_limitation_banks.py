"""Fit the SF-12 + limitation banks under the graded-response model and the
generalised partial credit model.

Banks, each the four validated SF-12 testlets (GH, PF, RP, BP) plus:
  P-FUNC    FUNC: the five physical-function areas, capped 0/1/2/3+
  P-LIM1    LIM_PF: the same five areas, uncapped 0-5
  P-LIM     LIM_PF + LIM_SC (continence, hearing, sight)
  P-LIM4    LIM_MOB (mobility, lifting), LIM_DEX (dexterity, coordination,
            personal care), LIM_CONT (continence), LIM_SENS (hearing, sight)
  P-LIM3    LIM_FL (mobility, lifting, dexterity, coordination), LIM_SELF
            (continence, personal care), LIM_SENS
  P-LIM3+O  P-LIM3 + LIM_OTHER (other health problem or disability)

Condition banks, graded-response model only: each limitation bank except P-FUNC
plus either +CC, COND, the count of the sixteen ever-diagnosed physical
conditions (top category by the merge rule), or +CG, P-FULL's six group items.
They need a condition inventory, so they use P-FULL's person-waves; P-FUNC+CG
is fitted as a check that the coding reproduces P-FULL exactly.

Coding (measures.grm2.build_limitation_items): no long-standing illness counts
as no limitation in every wave; a top count category under 0.5% of
person-waves merges into the one below, decided on the common sample before
fitting. All banks share P-FUNC's person-waves.

Each bank is fitted pooled (latent N(0, 1)) under both models and scored by
EAP under the pooled prior; h is theta through the expected-score curve.
Q3 and the EAP identity are reported, never used to change a bank.

Outputs (gitignored), data/processed/measures/:
  limitation_bank_items.parquet   every testlet code, person-wave
  limitation_scores.parquet       theta, theta_sd, h per bank and model
  limitation_items.csv            a and thresholds / steps
  limitation_q3.csv, limitation_fit_summary.csv, limitation_categories.csv,
  limitation_information.csv      item information on the quadrature grid

Archived: the measure search itself — the six limitation banks under both
models and the ten condition banks. P-LIM3+CC is what the search chose, and
it is now built by data_cleaning/04_build_health.py; this script remains the
way to reproduce the comparison the choice rested on.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from prevention_health_clustering.cli_utils import ensure_runtime_directories
from prevention_health_clustering.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from prevention_health_clustering.measures import gpcm, grm
from prevention_health_clustering.measures.grm import TESTLETS
from prevention_health_clustering.measures.grm2 import (
    CONDITION_CODINGS,
    PHYS_CONDITION_ITEMS,
    add_condition_items,
    LIMITATION_BANKS,
    LIMITATION_GROUPS,
    build_limitation_items,
)

SLUG = {"P-FUNC": "pfunc", "P-LIM1": "plim1", "P-LIM": "plim",
        "P-LIM4": "plim4", "P-LIM3": "plim3", "P-LIM3+O": "plim3o"}
MODELS = {
    "grm": (grm.fit_grm, grm.score_eap, grm.yens_q3, grm.grmh_from_theta, grm.test_information),
    "gpcm": (gpcm.fit_gpcm, gpcm.score_eap, gpcm.yens_q3, gpcm.h_from_theta, gpcm.test_information),
}


def main() -> int:
    ensure_runtime_directories()
    checks: list[tuple[str, bool, str]] = []
    items = pd.read_parquet(INTERIM_DATA_DIR / "sf12_items_long.parquet")
    bank_items, ncat = build_limitation_items(items)
    n_obs = len(bank_items)
    print(f"common sample: {n_obs:,} person-waves, {bank_items['pidp'].nunique():,} people")

    cat_rows = []
    for g in ["FUNC", *LIMITATION_GROUPS]:
        share = bank_items[g].value_counts(normalize=True).sort_index(ascending=False)
        for i, (code, v) in enumerate(share.items()):
            cat_rows.append({"testlet": g, "difficulties": i, "code": int(code), "share": float(v)})
        print(f"  {g:10s} {ncat[g]} categories: "
              + " / ".join(f"{100 * v:.2f}" for v in share.values) + "  (% none first)")

    scores = bank_items[["pidp", "wave", "age"]].copy()
    item_rows, q3_rows, fit_rows, info_rows = [], [], [], []
    for bank, lims in LIMITATION_BANKS.items():
        names = [*TESTLETS, *lims]
        k = [ncat[c] for c in names]
        pat = bank_items[names].value_counts().reset_index(name="N")
        Y, w = pat[names].to_numpy(), pat["N"].to_numpy(float)
        for model, (fit_fn, score_fn, q3_fn, h_fn, info_fn) in MODELS.items():
            t0 = time.time()
            fit = fit_fn(Y, w, k, names)
            eap, psd = score_fn(fit, Y)
            q3 = q3_fn(fit, Y, w, eap)
            pat["theta"], pat["theta_sd"] = eap, psd
            m = bank_items[names].merge(pat[names + ["theta", "theta_sd"]], on=names, how="left")
            col = f"{SLUG[bank]}_{model}"
            scores[f"theta_{col}"] = m["theta"].to_numpy()
            scores[f"theta_sd_{col}"] = m["theta_sd"].to_numpy()
            scores[f"h_{col}"] = h_fn(fit, m["theta"].to_numpy())
            identity = m["theta"].var(ddof=0) + (m["theta_sd"] ** 2).mean()
            off = q3[np.triu_indices(len(names), 1)]
            ll, npar = fit.log_likelihood, fit.n_parameters
            fit_rows.append({"bank": bank, "model": model, "n": n_obs, "patterns": len(pat),
                             "testlets": len(names), "log_likelihood": ll, "n_parameters": npar,
                             "aic": 2 * npar - 2 * ll, "bic": npar * np.log(n_obs) - 2 * ll,
                             "eap_identity": identity, "max_q3": off.max(),
                             "em_iterations": fit.n_iter, "seconds": time.time() - t0})
            print(f"[{bank} {model}] {len(pat):,} patterns, {fit.n_iter} EM iters, "
                  f"{time.time() - t0:.0f}s, logL {ll:.0f}, identity {identity:.3f}, "
                  f"max Q3 {off.max():+.3f}", flush=True)
            for c in names:
                a, b = fit.items[c]
                item_rows.append({"bank": bank, "model": model, "item": c, "a": a,
                                  **{f"b{i + 1}": v for i, v in enumerate(b)}})
                print(f"    {c:10s} a = {a:5.2f}   b = " + " ".join(f"{v:5.2f}" for v in b))
            for i, ia in enumerate(names):
                for j in range(i + 1, len(names)):
                    q3_rows.append({"bank": bank, "model": model, "item_a": ia,
                                    "item_b": names[j], "q3": q3[i, j]})
            info = info_fn(fit, fit.th)
            for j, c in enumerate(names):
                info_rows.append(pd.DataFrame({"bank": bank, "model": model, "item": c,
                                               "theta": fit.th, "information": info[:, j]}))
            checks.append((f"{bank} {model} EAP identity", abs(identity - 1) < 0.05,
                           f"{identity:.3f}"))
            if model == "gpcm":
                T = gpcm.weighted_sum(fit, Y)
                o = np.argsort(T, kind="stable")
                steps, ties = np.diff(eap[o]), np.isclose(np.diff(T[o]), 0.0)
                checks.append((f"{bank} gpcm EAP monotone in weighted sum",
                               bool((steps[~ties] > -1e-9).all()),
                               f"min step {steps[~ties].min():+.2e}"))
        r = scores[f"theta_{SLUG[bank]}_grm"].corr(scores[f"theta_{SLUG[bank]}_gpcm"])
        print(f"  corr(theta GRM, theta GPCM) = {r:.4f}")

    # ---------------- condition banks, graded-response model -----------------
    chronic = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "chronic_conditions.parquet")
    cond_items, cond_ncat = add_condition_items(bank_items, chronic)
    ncat.update(cond_ncat)
    cmask = cond_items["COND"].notna()
    n_cond = int(cmask.sum())
    print(f"\ncondition sample: {n_cond:,} person-waves, "
          f"{cond_items.loc[cmask, 'pidp'].nunique():,} people; COND top category "
          f"{cond_ncat['COND'] - 1}+")
    for g in ["COND", *PHYS_CONDITION_ITEMS]:
        share = cond_items.loc[cmask, g].value_counts(normalize=True).sort_index(ascending=False)
        for i, (code, v) in enumerate(share.items()):
            cat_rows.append({"testlet": g, "difficulties": i, "code": int(code), "share": float(v)})
        print(f"  {g:10s} {ncat[g]} categories: "
              + " / ".join(f"{100 * v:.2f}" for v in share.values) + "  (% none first)")
    specs = [(f"{bank}+{code}", SLUG[bank] + code.lower(), [*TESTLETS, *lims, *cond])
             for bank, lims in LIMITATION_BANKS.items() if bank != "P-FUNC"
             for code, cond in CONDITION_CODINGS.items()]
    specs.append(("P-FUNC+CG", None, [*TESTLETS, "FUNC", *CONDITION_CODINGS["CG"]]))
    for name, slug, names in specs:
        frame = cond_items.loc[cmask, ["pidp", "wave", *names]].copy()
        frame[names] = frame[names].astype(int)
        k = [ncat[c] for c in names]
        pat = frame[names].value_counts().reset_index(name="N")
        Y, w = pat[names].to_numpy(), pat["N"].to_numpy(float)
        t0 = time.time()
        fit = grm.fit_grm(Y, w, k, names)
        eap, psd = grm.score_eap(fit, Y)
        q3 = grm.yens_q3(fit, Y, w, eap)
        pat["theta"], pat["theta_sd"] = eap, psd
        m = frame.merge(pat[names + ["theta", "theta_sd"]], on=names, how="left")
        identity = m["theta"].var(ddof=0) + (m["theta_sd"] ** 2).mean()
        off = q3[np.triu_indices(len(names), 1)]
        print(f"[{name} grm] {len(pat):,} patterns, {fit.n_iter} EM iters, {time.time() - t0:.0f}s, "
              f"logL {fit.log_likelihood:.0f}, identity {identity:.3f}, max Q3 {off.max():+.3f}", flush=True)
        for c in names:
            a, b = fit.items[c]
            print(f"    {c:10s} a = {a:5.2f}   b = " + " ".join(f"{v:5.2f}" for v in b))
        checks.append((f"{name} grm EAP identity", abs(identity - 1) < 0.05, f"{identity:.3f}"))
        if slug is None:
            old = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "grm2_scores.parquet",
                                  columns=["pidp", "wave", "theta_phys_full"]).dropna()
            mm = m[["pidp", "wave", "theta"]].merge(old, on=["pidp", "wave"], how="inner")
            gap = (mm["theta"] - mm["theta_phys_full"]).abs().max()
            checks.append(("P-FUNC+CG GRM reproduces P-FULL", len(mm) == n_cond == len(old) and gap < 1e-4,
                           f"{len(mm):,} rows, max |diff| {gap:.2e}"))
            continue
        col = f"{slug}_grm"
        add = m[["pidp", "wave"]].assign(**{f"theta_{col}": m["theta"].to_numpy(),
                                             f"theta_sd_{col}": m["theta_sd"].to_numpy(),
                                             f"h_{col}": grm.grmh_from_theta(fit, m["theta"].to_numpy())})
        scores = scores.merge(add, on=["pidp", "wave"], how="left")
        ll, npar = fit.log_likelihood, fit.n_parameters
        fit_rows.append({"bank": name, "model": "grm", "n": n_cond, "patterns": len(pat),
                         "testlets": len(names), "log_likelihood": ll, "n_parameters": npar,
                         "aic": 2 * npar - 2 * ll, "bic": npar * np.log(n_cond) - 2 * ll,
                         "eap_identity": identity, "max_q3": off.max(),
                         "em_iterations": fit.n_iter, "seconds": time.time() - t0})
        for c in names:
            a, b = fit.items[c]
            item_rows.append({"bank": name, "model": "grm", "item": c, "a": a,
                              **{f"b{i + 1}": v for i, v in enumerate(b)}})
        for i, ia in enumerate(names):
            for j in range(i + 1, len(names)):
                q3_rows.append({"bank": name, "model": "grm", "item_a": ia, "item_b": names[j], "q3": q3[i, j]})
        info = grm.test_information(fit, fit.th)
        for j, c in enumerate(names):
            info_rows.append(pd.DataFrame({"bank": name, "model": "grm", "item": c,
                                           "theta": fit.th, "information": info[:, j]}))

    # the refitted P-FUNC GRM must reproduce the 06 build
    old = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "grm2_scores.parquet",
                          columns=["pidp", "wave", "theta_phys_func"])
    mm = scores.merge(old, on=["pidp", "wave"], how="inner")
    gap = (mm["theta_pfunc_grm"] - mm["theta_phys_func"]).abs().max()
    checks.append(("P-FUNC GRM reproduces grm2_scores", len(mm) == n_obs and gap < 1e-4,
                   f"{len(mm):,} rows, max |diff| {gap:.2e}"))

    out = PROCESSED_DATA_DIR / "measures"
    cond_items.to_parquet(out / "limitation_bank_items.parquet", index=False)
    scores.to_parquet(out / "limitation_scores.parquet", index=False)
    pd.DataFrame(item_rows).to_csv(out / "limitation_items.csv", index=False)
    pd.DataFrame(q3_rows).to_csv(out / "limitation_q3.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(out / "limitation_fit_summary.csv", index=False)
    pd.DataFrame(cat_rows).to_csv(out / "limitation_categories.csv", index=False)
    pd.concat(info_rows).to_csv(out / "limitation_information.csv", index=False)
    print("wrote limitation_bank_items.parquet, limitation_scores.parquet and five csv files")

    print("\n=== checks ===")
    ok_all = True
    for name, ok, detail in checks:
        ok_all &= ok
        print(f"  {'PASS' if ok else 'FAIL':4s}  {name:44s} {detail}")
    print("LIMITATION BANKS OK" if ok_all else "LIMITATION BANKS FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
