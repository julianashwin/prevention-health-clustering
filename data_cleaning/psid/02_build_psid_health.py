"""Build the PSID physical health measure, theta and h, the way 04_build_health.py builds the UKHLS one.

The five testlets PSID asks in every wave from 2003, coded so 1 is the worst category, with the
repo's own merge rule (a top count category holding under 0.5% of person-waves merges into the one
below, grm2.top_category) and reversal (grm2._reverse_count):

    GH        srh: excellent ... poor, reversed                               5 categories
    RP        worklim_ord: none, a little, somewhat, a lot or cannot work     4
    LIM_FL    mobility area: difficulty walking, with a bed or chair, or going outside
    LIM_SELF  personal-care area: difficulty bathing, dressing, eating or using the toilet
    COND      the nine ever-diagnosed physical conditions, cumulated as the UKHLS code does
              (cond_count_phys: once reported, always counted)

The graded-response model is the repo's own (measures/grm.py), unchanged: fitted pooled over answer
patterns with the latent distribution N(0, 1), and again multigroup by single year of age for the
latent age profile. theta is the EAP under the pooled prior, h its expected-score curve on [0, 1].
Nothing in src/ or in the UKHLS scripts is edited; the functions are imported.

A check version on 2017-2023 adds the two items PSID asks only from 2017, where the UKHLS testlets
put them: PF (trouble walking several blocks or climbing a few flights of stairs) as its own testlet,
and lifting as a second area of LIM_FL (lim_fl_17). It is not used downstream.

Sample: Reference Persons and Spouses, 2003-2023, ages 20-90, complete on the five testlets.

Outputs (gitignored), data/processed/psid/measures/:
    psid_health_measure.parquet          pid, wave, age, theta, theta_sd, h, ws, self_report
    psid_health_measure_items.csv        discriminations and thresholds
    psid_health_measure_q3.csv           Yen's Q3 between testlets
    psid_health_measure_age_profile.csv  mu_a and sigma_a by single year of age
    psid_health_measure_weights.csv      the weighted sum that reproduces h
    psid_health_measure_2017.parquet     the six-testlet check on 2017-2023: theta6, theta6_sd, h6, theta
    psid_health_measure_items_2017.csv   its discriminations and thresholds

    python data_cleaning/psid/02_build_psid_health.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from prevention_health_clustering.measures.grm import fit_grm, grmh_from_theta, score_eap, yens_q3
from prevention_health_clustering.measures.grm2 import _reverse_count, top_category

FIRST_WAVE = 2003
AGE_RANGE = (20, 90)
AGES = np.arange(AGE_RANGE[0], AGE_RANGE[1] + 1)
ITEMS = ("GH", "RP", "LIM_FL", "LIM_SELF", "COND")
ITEMS_2017 = ("GH", "PF", "RP", "LIM_FL", "LIM_SELF", "COND")


def count_item(count: pd.Series) -> tuple[pd.Series, int]:
    """A 0-based deficit count capped by the merge rule and reversed so 1 = worst."""
    k = top_category(count) + 1
    return _reverse_count(count, k), k


def build_bank(panel: pd.DataFrame, with_2017_items: bool = False) -> tuple[pd.DataFrame, dict[str, int]]:
    """Person-waves with the testlets, complete cases, 1 = worst."""
    names = ITEMS_2017 if with_2017_items else ITEMS
    d = panel.copy()
    d["GH"] = 6 - d["srh"]                                   # srh 1 excellent ... 5 poor
    d["RP"] = _reverse_count(d["worklim_ord"], 4)            # 0 none ... 3 a lot / cannot work
    fl = "lim_fl_17" if with_2017_items else "lim_fl"
    if with_2017_items:
        d["PF"] = _reverse_count(d["mob_walk_climb"], 2)     # 1 = has trouble
    raw = {"LIM_FL": fl, "LIM_SELF": "lim_self", "COND": "cond_count_phys"}
    need = ["srh", "worklim_ord", *raw.values()] + (["mob_walk_climb"] if with_2017_items else [])
    d = d.dropna(subset=need)
    ncat = {"GH": 5, "RP": 4, "PF": 2}
    for item, col in raw.items():                            # capped on the complete-case sample
        d[item], ncat[item] = count_item(d[col])
    for c in names:
        d[c] = d[c].astype(int)
    keep = ["pid", "wave", "age", "self_report", *names]
    return d[keep].reset_index(drop=True), {c: ncat[c] for c in names}


def fit_and_score(bank: pd.DataFrame, ncat: dict[str, int], names: tuple[str, ...]):
    pat = bank[list(names)].value_counts().reset_index(name="N")
    Y, w = pat[list(names)].to_numpy(), pat["N"].to_numpy(float)
    fit = fit_grm(Y, w, [ncat[c] for c in names], list(names))
    eap, psd = score_eap(fit, Y)
    q3 = yens_q3(fit, Y, w, eap)
    pat["theta"], pat["theta_sd"] = eap, psd
    scored = bank.merge(pat[list(names) + ["theta", "theta_sd"]], on=list(names), how="left")
    scored["h"] = grmh_from_theta(fit, scored["theta"].to_numpy())
    return fit, scored, q3, len(pat)


def projection_weights(codes: pd.DataFrame, score: np.ndarray, names) -> tuple[pd.Series, float]:
    """measures/health.projection_weights on the PSID testlets: the score on the standardised codes."""
    X = codes[list(names)].to_numpy(float)
    Z = np.column_stack([np.ones(len(X)), (X - X.mean(axis=0)) / X.std(axis=0)])
    b, *_ = np.linalg.lstsq(Z, score, rcond=None)
    r2 = 1 - ((score - Z @ b) ** 2).sum() / ((score - score.mean()) ** 2).sum()
    return pd.Series(b[1:] / b[1:].sum(), index=list(names)), float(r2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", default="data/processed/psid/psid_panel.parquet")
    ap.add_argument("--out", default="data/processed/psid/measures")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = ["pid", "wave", "age", "role", "self_report", "srh", "worklim_ord", "lim_fl", "lim_fl_17",
            "lim_self", "cond_count_phys", "mob_walk_climb"]
    panel = pd.read_parquet(args.panel, columns=cols)
    panel = panel[(panel["wave"] >= FIRST_WAVE) & panel["age"].between(*AGE_RANGE)]
    panel["age"] = panel["age"].astype(int)
    checks: list[tuple[str, bool, str]] = []

    bank, ncat = build_bank(panel)
    names = list(ITEMS)
    print(f"sample: {len(bank):,} person-waves, {bank['pid'].nunique():,} people, ages "
          f"{bank['age'].min()}-{bank['age'].max()}, waves {bank['wave'].min()}-{bank['wave'].max()} "
          f"({len(bank) / len(panel):.1%} of RP and SP rows in that window)")
    for c in names:
        share = bank[c].value_counts(normalize=True).sort_index(ascending=False)
        print(f"  {c:9s} {ncat[c]} categories, best first: " + " / ".join(f"{100 * v:.2f}" for v in share.values))

    # ---------------- fit and score ----------------------------------------
    t0 = time.time()
    fit, scored, q3, npat = fit_and_score(bank, ncat, ITEMS)
    print(f"\ngraded-response fit: {npat:,} answer patterns, {fit.n_iter} EM iterations, "
          f"{time.time() - t0:.0f}s, log-likelihood {fit.log_likelihood:,.0f}")
    for c in names:
        a, b = fit.items[c]
        print(f"  {c:9s} a = {a:5.2f}   thresholds " + " ".join(f"{v:5.2f}" for v in b))
    top = scored["theta"].max()
    print(f"best answer on every testlet: {(scored['theta'] == top).mean():.1%} of person-waves, "
          f"theta {top:.2f}")

    # ---------------- the latent age profile --------------------------------
    t0 = time.time()
    g = bank.assign(gidx=bank["age"].to_numpy() - AGES[0])
    pg = g.groupby(names + ["gidx"]).size().reset_index(name="N")
    mg = fit_grm(pg[names].to_numpy(), pg["N"].to_numpy(float), [ncat[c] for c in names], names,
                 grp=pg["gidx"].to_numpy())
    print(f"multigroup by single year of age: {mg.n_iter} iterations, {time.time() - t0:.0f}s; "
          f"mu(80) - mu(30) = {mg.mu[60] - mg.mu[10]:+.3f}")
    n_age = bank.groupby("age").size().reindex(AGES, fill_value=0)
    profile = pd.DataFrame({"age": AGES, "mu": mg.mu, "sigma": mg.sigma, "n": n_age.to_numpy()})

    # ---------------- the weighted sum --------------------------------------
    weights, r2 = projection_weights(scored, scored["h"].to_numpy(), names)
    X = scored[names].to_numpy(float)
    ws = ((X - X.mean(axis=0)) / X.std(axis=0)) @ weights.to_numpy()
    scored["ws"] = (ws - ws.mean()) / ws.std()
    r_h_ws = spearmanr(scored["h"], scored["ws"]).statistic
    print(f"\nweighted sum: h on the standardised codes, R-squared {r2:.3f}; "
          + "  ".join(f"{k} {100 * v:.0f}%" for k, v in weights.items()))

    # ---------------- checks (the UKHLS gates) -------------------------------
    off = q3[np.triu_indices(len(names), 1)]
    identity = scored["theta"].var(ddof=0) + (scored["theta_sd"] ** 2).mean()
    checks.append(("no positive local dependence", off.max() < 0.2, f"max Q3 {off.max():+.3f}"))
    checks.append(("EAP identity", abs(identity - 1) < 0.05, f"{identity:.3f}"))
    checks.append(("h is monotone in theta", bool(
        (np.diff(scored.sort_values("theta")["h"].to_numpy()) >= -1e-12).all()), "by construction"))
    checks.append(("the weighted sum agrees with h in rank", r_h_ws > 0.95, f"ws {r_h_ws:.3f}"))

    # ---------------- the six-testlet check, 2017-2023 ----------------------
    p17 = panel[panel["wave"] >= 2017]
    bank6, ncat6 = build_bank(p17, with_2017_items=True)
    fit6, scored6, q36, npat6 = fit_and_score(bank6, ncat6, ITEMS_2017)
    off6 = q36[np.triu_indices(len(ITEMS_2017), 1)]
    print(f"\nsix-testlet check, 2017-2023: {len(bank6):,} person-waves, {npat6:,} patterns, largest Q3 "
          f"{off6.max():+.3f}")
    for c in ITEMS_2017:
        a, b = fit6.items[c]
        print(f"  {c:9s} {ncat6[c]} categories, a = {a:5.2f}   thresholds " + " ".join(f"{v:5.2f}" for v in b))
    both = scored6.rename(columns={"theta": "theta6", "theta_sd": "theta6_sd", "h": "h6"})[
        ["pid", "wave", "theta6", "theta6_sd", "h6"]].merge(scored[["pid", "wave", "theta"]], on=["pid", "wave"])
    print(f"  correlation with the five-testlet theta on the same rows: Pearson "
          f"{np.corrcoef(both['theta'], both['theta6'])[0, 1]:.3f}, Spearman "
          f"{spearmanr(both['theta'], both['theta6']).statistic:.3f}")

    # ---------------- write --------------------------------------------------
    keep = ["pid", "wave", "age", "theta", "theta_sd", "h", "ws", "self_report"]
    scored[keep].to_parquet(out_dir / "psid_health_measure.parquet", index=False)
    pd.DataFrame([{"item": c, "categories": ncat[c], "a": fit.items[c][0],
                   **{f"b{i + 1}": v for i, v in enumerate(fit.items[c][1])}} for c in names]
                 ).to_csv(out_dir / "psid_health_measure_items.csv", index=False)
    pd.DataFrame([{"item_a": names[i], "item_b": names[j], "q3": q3[i, j]}
                  for i in range(len(names)) for j in range(i + 1, len(names))]
                 ).to_csv(out_dir / "psid_health_measure_q3.csv", index=False)
    profile.to_csv(out_dir / "psid_health_measure_age_profile.csv", index=False)
    weights.rename("weight").rename_axis("item").reset_index().assign(r_squared=r2).to_csv(
        out_dir / "psid_health_measure_weights.csv", index=False)
    both.to_parquet(out_dir / "psid_health_measure_2017.parquet", index=False)
    pd.DataFrame([{"item": c, "categories": ncat6[c], "a": fit6.items[c][0],
                   **{f"b{i + 1}": v for i, v in enumerate(fit6.items[c][1])}} for c in ITEMS_2017]
                 ).to_csv(out_dir / "psid_health_measure_items_2017.csv", index=False)
    print(f"\nwrote {out_dir}/psid_health_measure.parquet ({len(scored):,} rows) and six more files")

    print("\n=== checks ===")
    ok_all = True
    for name, ok, detail in checks:
        ok_all &= ok
        print(f"  {'PASS' if ok else 'FAIL':4s}  {name:44s} {detail}")
    print("PSID HEALTH MEASURE OK" if ok_all else "PSID HEALTH MEASURE FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
