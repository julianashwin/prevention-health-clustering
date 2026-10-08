"""Remaining life expectancy and healthy life expectancy by class, K = 3 AR(1) + "spike".

Classes and hazards come from the multidimensional fit theta-ssm-mort
(artifacts/multidim-health/; theta + the mental GRM + Gompertz-Makeham mortality,
AR(1) plus "spike"), the K = 3 AR(1) + "spike" fit that carries a class-specific
hazard. Its classes agree with the single-channel theta classes for 77% of people
(39_paper_multidim.py).

Attrition. The panel records a death only when the survey learns of it, so the
fitted hazards understate mortality. Every class hazard is scaled by one common
age-varying factor m(a) = exp(alpha + beta (a - 65)), with alpha and beta chosen so
that the composition-weighted population life expectancy at 65 and at 90 matches
the ONS national life tables for the UK, 2020 to 2022 (both sexes combined with
the contract's female share): the class differences are the fit's, the level is
the UK's. The hazards are extrapolated to age 110.

Healthy life expectancy. Healthy = general health reported good, very good or
excellent (the SF-12 general-health item, as in the ONS definition). Each class's
probability of being healthy at each age is the posterior-weighted share of its
person-waves observed at that age (five-year rolling mean, held at the last value
beyond 89); HLE from age r is the survival-weighted sum of those probabilities.

Outputs: paper/figures/fig_life_expectancy.png, paper/tables/tab_life_expectancy.tex,
artifacts/descriptives/paper_life_expectancy.csv.

    PYTHONPATH=src .venv/bin/python descriptives/44_paper_life_expectancy.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import DESC, FIG, TAB  # noqa: E402
from _style import CLUSTER, INK2, apply_style  # noqa: E402

from prevention_health_clustering.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR  # noqa: E402

M39 = importlib.import_module("39_paper_multidim")
K = 3
TAG = "theta-ssm-mort"
AGE_MAX = 110
AGES = np.arange(20, AGE_MAX + 1)
R_GRID = np.arange(20, 86)
REPORT_AGES = (20, 40, 65, 80)
# ONS national life tables, UK, 2020 to 2022 (period): remaining life expectancy at 65 and at 90,
# males and females; combined below with the contract's female share. Life expectancy at birth
# (78.6 / 82.6) is reported as a check only, since the model starts at 20.
ONS = {"e65": (18.3, 20.8), "e90": (3.8, 4.5), "e0": (78.6, 82.6)}
HEALTHY_MAX = 3            # sf1: 1 excellent ... 5 poor; healthy = good or better


def class_hazards(f) -> np.ndarray:
    """Yearly Gompertz-Makeham hazard by class on AGES, from the fit's posterior means."""
    return np.vstack([f["makeham"] + np.exp(f["log_b"][k] + f["slope"][k] * (AGES - 55)) for k in range(K)])


def remaining(hz: np.ndarray, p: np.ndarray | None = None) -> np.ndarray:
    """Remaining (healthy) life expectancy from every age r on AGES, for one hazard row.

    Survival from r is exp(-cumulative hazard); the integral uses the trapezoid rule on
    yearly ages, weighted by ``p`` (the probability of being healthy at each age) for HLE."""
    p = np.ones_like(hz) if p is None else p
    H = np.concatenate([[0.0], np.cumsum(0.5 * (hz[1:] + hz[:-1]))])     # cumulative hazard from AGES[0]
    out = np.full(len(AGES), np.nan)
    for i in range(len(AGES)):
        S = np.exp(-(H[i:] - H[i]))
        y = S * p[i:]
        out[i] = np.sum(0.5 * (y[1:] + y[:-1]))
    return out


def baseline(f, contract=None) -> dict:
    """The calibrated class hazards, the composition by age, the ONS targets, LE and HLE by class
    and the healthy share by class and age: everything the life-expectancy figure draws, exposed
    so that 48_paper_prevention_le.py can shift the same objects under prevention. ``f`` needs
    share, post, comp, log_b, slope and makeham; ``contract`` is the rows the posteriors belong
    to (the multidim contract by default)."""
    contract = M39.CONTRACT if contract is None else contract
    hz0 = class_hazards(f)
    comp = f["comp"].reindex(columns=[f"class{k + 1}" for k in range(K)])
    comp = comp.reindex(range(20, AGE_MAX + 1)).ffill()                    # hold the 89 composition beyond
    W = comp.to_numpy()                                                     # rows: AGES, cols: classes

    # the population's sex mix sets the ONS targets
    long = pd.read_csv(contract / "long.csv", usecols=["pidp", "wave", "age"])
    sex = pd.read_csv(PROCESSED_DATA_DIR / "ukhls_indresp_processed.csv", usecols=["pidp", "sex"]).groupby("pidp")["sex"].first()
    fem = float((sex.reindex(long["pidp"].unique()) == 2).mean())
    target = {k: (1 - fem) * m + fem * w for k, (m, w) in ONS.items()}
    print(f"female share {fem:.3f}; ONS targets (both sexes, this mix): e65 {target['e65']:.2f}, e90 {target['e90']:.2f}; e0 {target['e0']:.1f}")

    def pop_le(params):
        a, b = params
        m = np.exp(a + b * (AGES - 65))
        E = np.vstack([remaining(hz0[k] * m) for k in range(K)])          # K x AGES
        return (W * E.T).sum(axis=1), E, m
    def gap(params):
        e, _, _ = pop_le(params)
        return [e[65 - 20] - target["e65"], e[90 - 20] - target["e90"]]
    e_raw, E_raw, _ = pop_le((0.0, 0.0))
    (alpha, beta), info, ok, msg = fsolve(gap, (0.3, 0.0), full_output=True)
    e_pop, E, m = pop_le((alpha, beta))
    print(f"uncalibrated: population e20 {e_raw[0]:.1f}, e65 {e_raw[45]:.1f}, e90 {e_raw[70]:.1f}")
    print(f"calibrated multiplier m(a) = exp({alpha:.3f} + {beta:.4f} (a - 65)): m(20) {m[0]:.2f}, m(65) {m[45]:.2f}, m(90) {m[70]:.2f}; "
          f"population e20 {e_pop[0]:.1f} (ONS e0 - 20 = {target['e0'] - 20:.1f}), e65 {e_pop[45]:.1f}, e90 {e_pop[70]:.1f}; solver {msg.strip()}")

    # healthy share by class and age, posterior weighted
    items = pd.read_parquet(INTERIM_DATA_DIR / "sf12_items_long.parquet", columns=["pidp", "wave", "sf1"])
    rows = long.merge(items, on=["pidp", "wave"], how="left").merge(f["post"], on="pidp")
    rows["healthy"] = (rows["sf1"] <= HEALTHY_MAX).astype(float)
    P = np.zeros((K, len(AGES)))
    for k in range(K):
        w = rows[f"class{k + 1}"]
        g = pd.DataFrame({"age": rows["age"], "wy": w * rows["healthy"], "w": w}).groupby("age").sum()
        p = (g["wy"] / g["w"]).reindex(range(20, 90)).rolling(5, center=True, min_periods=1).mean()
        P[k] = p.reindex(range(20, AGE_MAX + 1)).ffill().to_numpy()
    HE = np.vstack([remaining(hz0[k] * m, P[k]) for k in range(K)])
    he_pop = (W * HE.T).sum(axis=1)
    p_pop = (W * P.T).sum(axis=1)
    return {"hz": hz0 * m, "hz0": hz0, "m": m, "alpha": alpha, "beta": beta, "W": W, "E": E, "HE": HE, "P": P,
            "e_pop": e_pop, "he_pop": he_pop, "p_pop": p_pop, "target": target, "fem": fem, "rows": rows}


def main() -> int:
    apply_style()
    f = M39.load_fit(TAG, "theta", None)
    B = baseline(f)
    W, E, HE, P, e_pop, he_pop, p_pop, m, target, fem = (B[k] for k in ("W", "E", "HE", "P", "e_pop", "he_pop", "p_pop", "m", "target", "fem"))

    # ---- table and csv
    recs = []
    for i, r in enumerate(AGES):
        for k in range(K):
            recs.append({"age": r, "class": k + 1, "le": E[k, i], "hle": HE[k, i], "p_healthy": P[k, i], "share_at_age": W[i, k]})
        recs.append({"age": r, "class": 0, "le": e_pop[i], "hle": he_pop[i], "p_healthy": p_pop[i], "share_at_age": 1.0})
    t = pd.DataFrame(recs); t.to_csv(DESC / "paper_life_expectancy.csv", index=False)
    with open(TAB / "tab_life_expectancy.tex", "w") as fh:
        fh.write("\\begin{tabular}{l" + "rr" * len(REPORT_AGES) + "}\n\\toprule\n")
        fh.write(" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{from age {r}}}" for r in REPORT_AGES) + " \\\\\n")
        fh.write(" ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(REPORT_AGES))) + "\n & " + " & ".join(["LE & HLE"] * len(REPORT_AGES)) + " \\\\\n\\midrule\n")
        for k in list(range(1, K + 1)) + [0]:
            lab = f"class {k} ({f['share'][k - 1]:.0%})" if k else "population"
            cells = []
            for r in REPORT_AGES:
                q = t[(t["age"] == r) & (t["class"] == k)].iloc[0]
                cells.append(f"{q['le']:.1f} & {q['hle']:.1f}")
            fh.write(lab + " & " + " & ".join(cells) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    for r in REPORT_AGES:
        i = r - 20
        print(f"from {r}: LE " + " / ".join(f"{E[k, i]:.1f}" for k in range(K)) + f" (pop {e_pop[i]:.1f}); HLE "
              + " / ".join(f"{HE[k, i]:.1f}" for k in range(K)) + f" (pop {he_pop[i]:.1f})")

    # ---- figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"wspace": 0.28})
    sel = slice(0, len(R_GRID))
    for k in range(K):
        axes[0].plot(R_GRID, E[k, sel], color=CLUSTER[k], lw=1.0 + 3.5 * f["share"][k], label=f"class {k + 1} ({f['share'][k]:.0%})")
        axes[1].plot(R_GRID, HE[k, sel], color=CLUSTER[k], lw=1.0 + 3.5 * f["share"][k])
        axes[2].plot(R_GRID, HE[k, sel] / E[k, sel], color=CLUSTER[k], lw=1.0 + 3.5 * f["share"][k])
    axes[0].plot(R_GRID, e_pop[sel], color=INK2, lw=1.4, ls="--", label="population (composition-weighted)")
    axes[0].plot([65], [target["e65"]], marker="o", color="#d62728", ms=6, lw=0, label="ONS 2020-22 at 65 (matched here and at 90)")
    axes[1].plot(R_GRID, he_pop[sel], color=INK2, lw=1.4, ls="--")
    axes[2].plot(R_GRID, he_pop[sel] / e_pop[sel], color=INK2, lw=1.4, ls="--")
    axes[0].set_title("(a) remaining life expectancy, years", loc="left", fontsize=10)
    axes[1].set_title("(b) remaining healthy life expectancy, years", loc="left", fontsize=10)
    axes[2].set_title("(c) healthy share of remaining life", loc="left", fontsize=10)
    axes[2].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlim(R_GRID[0], R_GRID[-1]); ax.grid(True, axis="y"); ax.set_xlabel("age $r$")
    axes[0].legend(fontsize=7.5, loc="upper right")
    import textwrap
    footer = (f"K = 3 AR(1) + \"spike\" classes of the multidimensional fit (theta, the mental GRM and Gompertz-Makeham mortality; {f['n_person']:,} people, {f['events']:,} recorded deaths), "
             "class 1 worst health. Every class hazard is scaled by one common age-varying factor, chosen so that the composition-weighted population life expectancy at 65 and at 90\n"
             f"matches the ONS national life tables for the UK, 2020 to 2022 (both sexes, {fem:.0%} female as in the sample: {target['e65']:.1f} and {target['e90']:.1f} years), "
             f"which corrects for deaths the panel does not record; the factor is {m[0]:.2f} at 20, {m[45]:.2f} at 65 and {m[70]:.2f} at 90, and the population life expectancy at 20 comes out at {e_pop[0]:.1f} "
             f"against {target['e0'] - 20:.1f} implied by the ONS figure at birth.\nHealthy = general health good, very good or excellent; each class's probability of being healthy at each age is the posterior-weighted share of its person-waves "
             "observed there (five-year rolling mean, held beyond 89). Hazards extrapolated to 110. The population line weights the classes by their posterior composition among the people observed at each age.")
    fig.text(0.01, -0.04, "\n".join(textwrap.wrap(footer.replace("\n", " "), 230)), fontsize=7.2, color=INK2, va="top")
    fig.savefig(FIG / "fig_life_expectancy.png", bbox_inches="tight")
    print(f"wrote fig_life_expectancy.png and tab_life_expectancy.tex to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
