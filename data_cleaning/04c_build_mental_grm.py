"""Fit the mental-health GRM and score it: the multidimensional model's second channel.

The bank is the three SF-12 mental testlets (MH, RE, SF) and the twelve GHQ-12
items, without the ever-depression diagnosis (DEPR). GHQ enters item-level; if
Yen's Q3 shows positive within-wording dependence (method effects) the
pre-specified remedy collapses the GHQ into two wording testlets. That decision
is taken, as it always was, on the bank WITH the diagnosis item (the pooled
M-ITEM fit), and then applied to the diagnosis-free bank that is scored: DEPR is
one weak item but it carries the condition inventory's BHPS exclusion into the
bank, costing 22% of person-waves, so the diagnosis-free bank is the one the
paper uses.

Reads data/interim/sf12_items_long.parquet (data_cleaning/02_build_measures.py)
and data/processed/measures/chronic_conditions.parquet (03_build_chronic.py).
Writes data/processed/measures/mental_grm.parquet: pidp, wave, age,
theta_ment_nodepr, theta_ment_nodepr_sd, one row per scored person-wave; and
mental_grm_items.csv with the item parameters.

    PYTHONPATH=src .venv/bin/python data_cleaning/04c_build_mental_grm.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from prevention_health_clustering.cli_utils import ensure_runtime_directories
from prevention_health_clustering.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from prevention_health_clustering.measures.grm import fit_grm, score_eap, yens_q3
from prevention_health_clustering.measures.grm2 import (
    GHQ_NEGATIVE,
    GHQ_POSITIVE,
    MENT_ITEMS,
    MENT_NCAT,
    MENT_TESTLET_NCAT,
    build_mental_items,
    collapse_ghq_wording,
)

OUT_DIR = PROCESSED_DATA_DIR / "measures"


def fit_bank(name, frame, names, ncat_map, *, verbose=False):
    """Pooled GRM fit on the response patterns, EAP scores on every row, Yen's Q3."""
    cols = list(names)
    pat = frame[cols].value_counts().reset_index(name="N")
    ncat = [ncat_map[c] for c in cols]
    t0 = time.time()
    fit = fit_grm(pat[cols].to_numpy(), pat["N"].to_numpy(float), ncat, cols, verbose=verbose)
    eap, psd = score_eap(fit, pat[cols].to_numpy())
    pat["theta"], pat["theta_sd"] = eap, psd
    scored = frame.merge(pat[cols + ["theta", "theta_sd"]], on=cols, how="left")
    q3 = yens_q3(fit, pat[cols].to_numpy(), pat["N"].to_numpy(float), pat["theta"].to_numpy())
    identity = scored["theta"].var(ddof=0) + (scored["theta_sd"] ** 2).mean()
    print(f"[{name}] n={len(frame):,} rows, {len(pat):,} patterns, {fit.n_iter} EM iters, "
          f"{time.time() - t0:.0f}s, logL {fit.log_likelihood:.0f}, EAP identity {identity:.3f}")
    for c in cols:
        a, b = fit.items[c]
        print(f"    {c:8s} a = {a:5.2f}   b = " + " ".join(f"{v:5.2f}" for v in b[: min(len(b), 8)]))
    return fit, scored, q3, identity


def main() -> int:
    ensure_runtime_directories()
    items = pd.read_parquet(INTERIM_DATA_DIR / "sf12_items_long.parquet")
    chronic = pd.read_parquet(OUT_DIR / "chronic_conditions.parquet")
    checks: list[tuple[str, bool, str]] = []

    # the wording-testlet decision, on the bank with the diagnosis item
    ment = build_mental_items(items, chronic)
    print(f"mental bank with DEPR: {len(ment):,} person-years")
    _, _, m_q3, _ = fit_bank("M-ITEM", ment, list(MENT_ITEMS), MENT_NCAT)
    names = list(MENT_ITEMS)
    qdf = pd.DataFrame(m_q3, index=names, columns=names)
    within = np.array([qdf.loc[a, b] for grp in (GHQ_POSITIVE, GHQ_NEGATIVE)
                       for i, a in enumerate(grp) for b in grp[i + 1:]])
    print(f"    GHQ within-wording Q3: mean {within.mean():+.3f}, max {within.max():+.3f}, "
          f"positive share {(within > 0).mean():.0%}")
    use_testlets = bool(within.max() > 0.2 or within.mean() > 0.05)
    print(f"    wording testlets: {'YES' if use_testlets else 'no'} "
          "(pre-specified rule: max within-wording Q3 > 0.2 or mean > 0.05)")
    m_names = list(MENT_TESTLET_NCAT) if use_testlets else list(MENT_ITEMS)
    m_ncat = MENT_TESTLET_NCAT if use_testlets else MENT_NCAT

    # the diagnosis-free bank that is scored
    ment_nd_raw = build_mental_items(items, chronic, include_depr=False)
    ment_nd = (collapse_ghq_wording(ment_nd_raw.assign(DEPR=1)).drop(columns=["DEPR"])
               if use_testlets else ment_nd_raw)
    nd_names = [c for c in m_names if c != "DEPR"]
    nd_ncat = {k: v for k, v in m_ncat.items() if k != "DEPR"}
    print(f"\nmental bank without DEPR: {len(ment_nd):,} person-years")
    fit, scored, q3, identity = fit_bank("MENT-ND", ment_nd, nd_names, nd_ncat)
    off = q3[np.triu_indices(len(nd_names), 1)]
    checks.append(("MENT-ND no positive local dependence", off.max() < 0.2, f"max Q3 {off.max():+.3f}"))
    checks.append(("MENT-ND EAP identity", abs(identity - 1.0) < 0.05, f"{identity:.3f}"))
    checks.append(("MENT-ND recovers the excluded sample", len(scored) > 1.2 * len(ment), f"{len(scored):,} vs {len(ment):,} rows"))

    out = (scored[["pidp", "wave", "age", "theta", "theta_sd"]]
           .rename(columns={"theta": "theta_ment_nodepr", "theta_sd": "theta_ment_nodepr_sd"})
           .sort_values(["pidp", "wave"]).reset_index(drop=True))
    out.to_parquet(OUT_DIR / "mental_grm.parquet", index=False)
    pd.DataFrame([{"item": c, "a": fit.items[c][0], **{f"b{i + 1}": v for i, v in enumerate(fit.items[c][1])}}
                  for c in nd_names]).to_csv(OUT_DIR / "mental_grm_items.csv", index=False)

    print("\nchecks")
    ok = True
    for name, passed, detail in checks:
        ok &= passed
        print(f"  [{'ok' if passed else 'FAIL'}] {name}: {detail}")
    print(f"wrote {OUT_DIR / 'mental_grm.parquet'} ({len(out):,} rows)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
