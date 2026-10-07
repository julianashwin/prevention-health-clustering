"""What targeted prevention does to life expectancy, lifespan inequality, health inequality and the
education, income and regional gradients.

Classes, paths and hazards are the multidimensional K = 3 AR(1) + "spike" fit with mortality
(theta-ssm-mort), calibrated to the ONS life tables exactly as in 44_paper_life_expectancy.py,
so the baseline here is Table tab:le. Prevention is the paper's instrument: from start age r,
intensity mu scales every future mean change of the type by (1 - mu), so the class mean path
gains D_j(a) = mu (pbar_j(r-1) - pbar_j(a)) for a >= r (held at its value at 90 beyond; the
paths stop there). Two links turn the gain in theta into years:

  mortality  the hazard of class j at age a is multiplied by exp(gamma(a) D_j(a)), with gamma(a)
             the within-age log-odds gradient of death before the next wave in theta (a logit
             of death on theta, a quadratic in age and theta x age over the UKHLS person-waves
             at risk, waves 1-14, as in 37_paper_cost_mortality_age.py);
  health     the class's probability of being healthy (general health good or better) at age a
             rises by the posterior-weighted mean over its person-waves at that age of
             sigma(x_i + b_theta(a) D_j(a)) - sigma(x_i), from a logit of healthy on the same
             design; the baseline share is the observed one of 44, so Table tab:le is unchanged.

Scenarios at each start age r: baseline; targeted, the worst class treated at mu = 1 (its mean
health held at its level at r - 1) and the others untouched; uniform with the same budget, every
class treated at mu = pi_1(r), the worst class's share among the people observed at r; and
uniform at mu = 1, the upper bound. Lifespan inequality is the standard deviation of age at death
from r over the composition-weighted mixture (and the class-3-minus-class-1 gap in LE); health
inequality the same gap in HLE. The gradients are the composition channel: each education,
income-tertile and region group's LE (HLE) at r is its posterior class composition among the
person-waves within five years of r times the class LE (HLE) under the scenario. Within-class
differences by group are not modelled, so a gradient here is the part the types carry.

Outputs: paper/figures/fig_prevention_le.png, paper/tables/tab_prevention_le.tex,
artifacts/descriptives/paper_prevention_le{,_groups}.csv.

    PYTHONPATH=src .venv/bin/python descriptives/48_paper_prevention_le.py
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import DESC, FIG, TAB, read_scores  # noqa: E402
from _style import CLUSTER, GREEN, INK2, ORANGE, PURPLE, VERM, apply_style  # noqa: E402

from prevention_health_clustering.config import PROCESSED_DATA_DIR  # noqa: E402

M37 = importlib.import_module("37_paper_cost_mortality_age")
M39 = importlib.import_module("39_paper_multidim")
M44 = importlib.import_module("44_paper_life_expectancy")
K = M44.K
AGES = M44.AGES                      # 20..110, the life-table grid
R_GRID = np.arange(25, 86)
R_REPORT = (30, 50, 70)
R_GRAD = 50
PATH_MAX = 90                        # the class paths run to 90
SCEN = ["baseline", "targeted", "uniform (same budget)", "uniform (mu = 1)"]
SCOL = {"baseline": INK2, "targeted": VERM, "uniform (same budget)": PURPLE, "uniform (mu = 1)": GREEN}
SLS = {"baseline": "-", "targeted": "-", "uniform (same budget)": "--", "uniform (mu = 1)": ":"}
EDUC_ORDER = ["other/none", "GCSE", "A-level", "degree or higher"]
REGION = {1: "North East", 2: "North West", 3: "Yorkshire & Humber", 4: "East Midlands", 5: "West Midlands", 6: "East of England",
          7: "London", 8: "South East", 9: "South West", 10: "Wales", 11: "Scotland", 12: "Northern Ireland"}


def design(theta: np.ndarray, age: np.ndarray) -> np.ndarray:
    a = (age - 55.0) / 10.0
    return np.column_stack([np.ones(len(theta)), theta, a, a ** 2, theta * a])


def slope(b: np.ndarray, age: np.ndarray) -> np.ndarray:
    """d logit / d theta at each age under the design above."""
    return b[1] + b[4] * (age - 55.0) / 10.0


def mortality_gradient() -> tuple[np.ndarray, int, int]:
    """Logit of death before the next wave on theta, age and theta x age; as 37's at-risk sample."""
    hm = read_scores(["theta"]); hm = hm[hm["age"].between(20, 90)]
    panel = pd.read_csv(PROCESSED_DATA_DIR / "ukhls_indresp_processed.csv", usecols=["pidp", "wave", "dcsedw_dv"])
    death_wave = panel.groupby("pidp")["dcsedw_dv"].max() - 18
    m = hm[hm["wave"] <= 14].merge(death_wave.rename("dw"), left_on="pidp", right_index=True, how="left")
    m = m[~(m["dw"] <= m["wave"])].assign(died=lambda x: (x["dw"] == x["wave"] + 1).astype(float)).reset_index(drop=True)
    b, V = M37.logit(design(m["theta"].to_numpy(float), m["age"].to_numpy(float)), m["died"].to_numpy(float))
    return b, len(m), int(m["died"].sum())


def gains(paths: np.ndarray, r: int, mu: np.ndarray) -> np.ndarray:
    """D_j(a) on AGES for prevention from r at intensity mu[j]: mu (pbar(r-1) - pbar(a)) for a >= r, held beyond 90."""
    D = np.zeros((K, len(AGES)))
    for k in range(K):
        p = paths[k]                                            # on 20..90
        gap = p[r - 1 - 20] - p[np.minimum(AGES, PATH_MAX) - 20]
        D[k] = np.where(AGES >= r, mu[k] * gap, 0.0)
    return D


def death_distribution(hz_row: np.ndarray, i: int) -> tuple[float, np.ndarray]:
    """From AGES[i]: the probability of dying in each yearly interval (mass beyond 110 put at 110)."""
    H = np.concatenate([[0.0], np.cumsum(0.5 * (hz_row[1:] + hz_row[:-1]))])
    S = np.exp(-(H[i:] - H[i]))
    d = np.append(-np.diff(S), S[-1])
    return AGES[i], d


def main() -> int:
    apply_style()
    f = M39.load_fit(M44.TAG, "theta", None)
    B = M44.baseline(f)
    hz, W, E0, HE0, P0, rows = (B[k] for k in ("hz", "W", "E", "HE", "P", "rows"))
    paths = f["paths"][0]                                              # physical channel, raw theta, 20..90
    share = f["share"]

    # the two links
    b_m, n_risk, n_deaths = mortality_gradient()
    g_m = slope(b_m, AGES.astype(float))
    print(f"mortality logit: {n_risk:,} person-waves at risk, {n_deaths:,} deaths; log-odds per unit theta "
          f"{slope(b_m, np.array([30.0]))[0]:.2f} at 30, {slope(b_m, np.array([55.0]))[0]:.2f} at 55, {slope(b_m, np.array([80.0]))[0]:.2f} at 80")
    long = pd.read_csv(M39.CONTRACT / "long.csv", usecols=["pidp", "wave", "age", "theta"])
    hr = rows.merge(long, on=["pidp", "wave", "age"]).dropna(subset=["sf1"]).reset_index(drop=True)
    b_h, _ = M37.logit(design(hr["theta"].to_numpy(float), hr["age"].to_numpy(float)), hr["healthy"].to_numpy(float))
    print(f"healthy logit: {len(hr):,} person-waves; log-odds per unit theta {slope(b_h, np.array([30.0]))[0]:.2f} at 30, {slope(b_h, np.array([80.0]))[0]:.2f} at 80")
    x_h = design(hr["theta"].to_numpy(float), hr["age"].to_numpy(float)) @ b_h
    p_h = 1 / (1 + np.exp(-x_h))
    age_idx = np.minimum(hr["age"].to_numpy(int), PATH_MAX) - 20                     # row's age on AGES
    Wrows = hr[[f"class{k + 1}" for k in range(K)]].to_numpy()
    b_h_age = slope(b_h, hr["age"].to_numpy(float))
    g_by_age = pd.DataFrame({"age": hr["age"]}).groupby("age").indices

    def healthy_shift(D: np.ndarray) -> np.ndarray:
        """Change in the healthy share by class and age from the gains D (K x AGES)."""
        dP = np.zeros((K, len(AGES)))
        for k in range(K):
            dp = 1 / (1 + np.exp(-(x_h + b_h_age * D[k][age_idx]))) - p_h
            w = Wrows[:, k]
            num = np.bincount(age_idx, weights=w * dp, minlength=len(AGES)); den = np.bincount(age_idx, weights=w, minlength=len(AGES))
            with np.errstate(invalid="ignore", divide="ignore"):
                s = np.where(den > 0, num / den, np.nan)
            dP[k] = pd.Series(s).ffill().fillna(0).to_numpy()
            dP[k][AGES > PATH_MAX] = dP[k][PATH_MAX - 20]                    # hold beyond 90 like the paths
        return dP

    def scenario(r: int, name: str) -> dict:
        i = r - 20
        if name == "baseline":
            mu = np.zeros(K)
        elif name == "targeted":
            mu = np.array([1.0] + [0.0] * (K - 1))
        elif name == "uniform (same budget)":
            mu = np.full(K, W[i, 0])
        else:
            mu = np.ones(K)
        D = gains(paths, r, mu)
        hz_s = hz * np.exp(g_m[None, :] * D)
        P_s = np.clip(P0 + healthy_shift(D), 0, 1) if mu.any() else P0
        E = np.array([M44.remaining(hz_s[k])[i] for k in range(K)])
        HE = np.array([M44.remaining(hz_s[k], P_s[k])[i] for k in range(K)])
        # the distribution of age at death from r, mixture over the composition at r
        ages_d = AGES[i:] + 0.5
        mix = sum(W[i, k] * death_distribution(hz_s[k], i)[1] for k in range(K))
        mean_d = float((mix * ages_d).sum()); sd_d = float(np.sqrt((mix * (ages_d - mean_d) ** 2).sum()))
        sd_cls = []
        for k in range(K):
            d = death_distribution(hz_s[k], i)[1]; mk = (d * ages_d).sum(); sd_cls.append(float(np.sqrt((d * (ages_d - mk) ** 2).sum())))
        return {"r": r, "scenario": name, "mu": mu, "E": E, "HE": HE, "pop_le": float((W[i] * E).sum()), "pop_hle": float((W[i] * HE).sum()),
                "sd_death": sd_d, "sd_death_class": np.array(sd_cls), "gap_le": float(E[K - 1] - E[0]), "gap_hle": float(HE[K - 1] - HE[0])}

    res = {(r, s): scenario(int(r), s) for r in R_GRID for s in SCEN}
    # each class treated on its own at mu = 1, for panels (a), (b)
    own = {}
    for r in R_GRID:
        i = r - 20; D = gains(paths, int(r), np.ones(K)); hz_s = hz * np.exp(g_m[None, :] * D); P_s = np.clip(P0 + healthy_shift(D), 0, 1)
        own[r] = (np.array([M44.remaining(hz_s[k])[i] for k in range(K)]), np.array([M44.remaining(hz_s[k], P_s[k])[i] for k in range(K)]))

    # ---- the gradients: composition by group within five years of r
    obs = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "observables_person.parquet").set_index("pidp")
    panel = pd.read_csv(PROCESSED_DATA_DIR / "ukhls_indresp_processed.csv", usecols=["pidp", "gor_dv"]).dropna()
    region = panel.groupby("pidp")["gor_dv"].agg(lambda z: z.mode().iloc[0]).astype(int)
    people = rows.drop_duplicates("pidp")[["pidp"]].join(obs[["educ_group", "income_rank_mean"]], on="pidp")
    people["income_tertile"] = pd.qcut(people["income_rank_mean"], 3, labels=["bottom third", "middle third", "top third"])
    people["region"] = people["pidp"].map(region).map(REGION)
    people.loc[~people["educ_group"].isin(EDUC_ORDER), "educ_group"] = np.nan
    gr = rows.merge(people[["pidp", "educ_group", "income_tertile", "region"]], on="pidp")
    cols = [f"class{k + 1}" for k in range(K)]
    GROUPS = {"education": ("educ_group", EDUC_ORDER), "income": ("income_tertile", ["bottom third", "middle third", "top third"]),
              "region": ("region", list(REGION.values()))}
    grec = []
    for r in R_REPORT:
        near = gr[gr["age"].between(r - 5, r + 5)]
        for gname, (col, order) in GROUPS.items():
            comp = near.groupby(col, observed=True)[cols].mean().reindex(order)
            n = near.groupby(col, observed=True)["pidp"].nunique().reindex(order)
            for s in SCEN:
                q = res[(r, s)]
                for g in order:
                    pi = comp.loc[g].to_numpy()
                    grec.append({"r": r, "gradient": gname, "group": g, "scenario": s, "n_people": int(n[g]), "share_worst": pi[0],
                                 "le": float((pi * q["E"]).sum()), "hle": float((pi * q["HE"]).sum())})
    G = pd.DataFrame(grec); G.to_csv(DESC / "paper_prevention_le_groups.csv", index=False)

    def gradient_stats(r: int, s: str) -> dict:
        q = G[(G["r"] == r) & (G["scenario"] == s)]
        ed = q[q["gradient"] == "education"].set_index("group"); inc = q[q["gradient"] == "income"].set_index("group"); rg = q[q["gradient"] == "region"]
        return {"educ_gap_le": ed.loc["degree or higher", "le"] - ed.loc["other/none", "le"], "educ_gap_hle": ed.loc["degree or higher", "hle"] - ed.loc["other/none", "hle"],
                "income_gap_le": inc.loc["top third", "le"] - inc.loc["bottom third", "le"], "income_gap_hle": inc.loc["top third", "hle"] - inc.loc["bottom third", "hle"],
                "region_range_le": rg["le"].max() - rg["le"].min(), "region_range_hle": rg["hle"].max() - rg["hle"].min()}

    # ---- csv and table
    recs = []
    for (r, s), q in res.items():
        rec = {"r": r, "scenario": s, "mu_class1": q["mu"][0], "mu_class3": q["mu"][-1], "pop_le": q["pop_le"], "pop_hle": q["pop_hle"], "sd_death": q["sd_death"],
               "gap_le": q["gap_le"], "gap_hle": q["gap_hle"]}
        for k in range(K):
            rec[f"le_{k + 1}"] = q["E"][k]; rec[f"hle_{k + 1}"] = q["HE"][k]; rec[f"sd_death_{k + 1}"] = q["sd_death_class"][k]
        if r in R_REPORT:
            rec.update(gradient_stats(r, s))
        recs.append(rec)
    T = pd.DataFrame(recs); T.to_csv(DESC / "paper_prevention_le.csv", index=False)
    with open(TAB / "tab_prevention_le.tex", "w") as fh:
        fh.write("\\begin{tabular}{llrrrrrrrr}\n\\toprule\n")
        fh.write(" & & \\multicolumn{2}{c}{population} & \\multicolumn{3}{c}{inequality} & \\multicolumn{3}{c}{gradient in LE, years} \\\\\n")
        fh.write("\\cmidrule(lr){3-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}\n")
        fh.write("from & policy & LE & HLE & sd(death age) & LE gap & HLE gap & education & income & region \\\\\n\\midrule\n")
        for r in R_REPORT:
            for s in SCEN:
                q = T[(T["r"] == r) & (T["scenario"] == s)].iloc[0]
                lab = {"baseline": "none", "targeted": "worst class, $\\mu=1$", "uniform (same budget)": f"everyone, $\\mu={q['mu_class3']:.2f}$",
                       "uniform (mu = 1)": "everyone, $\\mu=1$"}[s]
                fh.write(f"{r if s == 'baseline' else ''} & {lab} & {q['pop_le']:.1f} & {q['pop_hle']:.1f} & {q['sd_death']:.1f} & {q['gap_le']:.1f} & {q['gap_hle']:.1f} & "
                         f"{q['educ_gap_le']:.1f} & {q['income_gap_le']:.1f} & {q['region_range_le']:.1f} \\\\\n")
            if r != R_REPORT[-1]:
                fh.write("\\addlinespace\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    for r in R_REPORT:
        for s in SCEN:
            q = T[(T["r"] == r) & (T["scenario"] == s)].iloc[0]
            print(f"from {r} {s:22s} LE {q['pop_le']:5.1f} HLE {q['pop_hle']:5.1f} sd {q['sd_death']:4.1f} | gap LE {q['gap_le']:4.1f} HLE {q['gap_hle']:4.1f} | "
                  f"educ {q['educ_gap_le']:4.1f} income {q['income_gap_le']:4.1f} region {q['region_range_le']:4.1f} | class LE " + "/".join(f"{q[f'le_{k + 1}']:.1f}" for k in range(K)))

    # ---- figure
    fig = plt.figure(figsize=(17, 10.2))
    gs_top = fig.add_gridspec(1, 4, top=0.95, bottom=0.60, wspace=0.30)
    gs_bot = fig.add_gridspec(1, 4, top=0.46, bottom=0.10, wspace=0.95, width_ratios=[1, 1, 1.15, 0.85])
    axes = np.array([[fig.add_subplot(gs_top[0, j]) for j in range(4)], [fig.add_subplot(gs_bot[0, j]) for j in range(4)]])
    base = {r: res[(r, "baseline")] for r in R_GRID}
    for k in range(K):
        axes[0, 0].plot(R_GRID, [base[r]["E"][k] for r in R_GRID], color=CLUSTER[k], lw=1.0 + 3.5 * share[k], label=f"class {k + 1} ({share[k]:.0%})")
        axes[0, 0].plot(R_GRID, [own[r][0][k] for r in R_GRID], color=CLUSTER[k], lw=1.2, ls="--")
        axes[0, 1].plot(R_GRID, [base[r]["HE"][k] for r in R_GRID], color=CLUSTER[k], lw=1.0 + 3.5 * share[k])
        axes[0, 1].plot(R_GRID, [own[r][1][k] for r in R_GRID], color=CLUSTER[k], lw=1.2, ls="--")
    axes[0, 0].plot([], [], color=INK2, lw=1.2, ls="--", label="treated at $r$, $\\mu=1$")
    axes[0, 0].set_title("(a) remaining LE at $r$ (dashed: treated, $\\mu=1$)", loc="left", fontsize=9.5)
    axes[0, 1].set_title("(b) remaining healthy LE at $r$", loc="left", fontsize=9.5)
    axes[0, 0].legend(fontsize=7.5, loc="upper right")
    for s in SCEN:
        axes[0, 2].plot(R_GRID, [res[(r, s)]["gap_le"] for r in R_GRID], color=SCOL[s], ls=SLS[s], lw=1.8, label=s)
        axes[0, 3].plot(R_GRID, [res[(r, s)]["sd_death"] for r in R_GRID], color=SCOL[s], ls=SLS[s], lw=1.8, label=s)
    axes[0, 2].set_title("(c) LE gap, class 3 minus class 1, years", loc="left", fontsize=9.5)
    axes[0, 3].set_title("(d) sd of age at death from $r$, years", loc="left", fontsize=9.5)
    axes[0, 2].set_ylim(0, 20); axes[0, 3].set_ylim(3, 15)
    axes[0, 2].legend(fontsize=7, loc="upper right"); axes[0, 3].legend(fontsize=7, loc="lower left")
    for ax in axes[0]:
        ax.set_xlim(R_GRID[0], R_GRID[-1]); ax.grid(True, axis="y"); ax.set_xlabel("start age $r$")
    # the gradients at R_GRAD: dot plots, LE and HLE by group under the scenarios
    Gr = G[G["r"] == R_GRAD]
    for j, (gname, (col, order)) in enumerate(GROUPS.items()):
        ax = axes[1, j]; q = Gr[Gr["gradient"] == gname]
        if gname == "region":
            order = q[q["scenario"] == "baseline"].sort_values("le")["group"].tolist()
        y = np.arange(len(order))
        handles = []
        for s, mk, ms in (("baseline", "o", 6), ("uniform (same budget)", "s", 5), ("targeted", "D", 5), ("uniform (mu = 1)", "^", 5)):
            qq = q[q["scenario"] == s].set_index("group").reindex(order)
            h1, = ax.plot(qq["le"], y, marker=mk, ms=ms, lw=0, color=SCOL[s], label=f"LE, {s}")
            h2, = ax.plot(qq["hle"], y, marker=mk, ms=ms, lw=0, color=SCOL[s], mfc="white", label=f"HLE, {s}")
            handles += [h1, h2]
        qb = q[q["scenario"] == "baseline"].set_index("group").reindex(order)
        ax.set_yticks(y); ax.set_yticklabels([f"{g} ({qb.loc[g, 'share_worst']:.0%} worst)" for g in order], fontsize=7.5 if gname == "region" else 8.5)
        ax.invert_yaxis(); ax.grid(True, axis="x"); ax.set_xlabel(f"years from {R_GRAD} (filled LE, open HLE)")
        ax.set_title(f"({'efg'[j]}) by {gname} at $r={R_GRAD}$", loc="left", fontsize=9.5)
    fig.legend(handles, [h.get_label() for h in handles], fontsize=7.5, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), frameon=False)
    # (h) the gradients themselves under each scenario
    ax = axes[1, 3]; stats = {s: gradient_stats(R_GRAD, s) for s in SCEN}
    names = [("educ_gap_le", "education"), ("income_gap_le", "income"), ("region_range_le", "region")]
    xg = np.arange(len(names)); wb = 0.2
    for i, s in enumerate(SCEN):
        ax.bar(xg + (i - 1.5) * wb, [stats[s][n] for n, _ in names], width=wb * 0.92, color=SCOL[s], label=s)
    ax.set_xticks(xg); ax.set_xticklabels([n for _, n in names], fontsize=8.5); ax.grid(True, axis="y")
    ax.set_title(f"(h) LE gradients at $r={R_GRAD}$, years", loc="left", fontsize=9.5); ax.legend(fontsize=7, loc="upper left")
    q50 = {s: res[(R_GRAD, s)] for s in SCEN}
    footer = (f"The K = 3 AR(1) + \"spike\" classes of the multidimensional fit (theta, the mental GRM and Gompertz-Makeham mortality; {f['n_person']:,} people), class 1 worst health; hazards calibrated to the ONS "
              "life tables as in the life expectancy figure, so the baseline is that figure's. Prevention from r at intensity mu holds back a share mu of the class's mean change in theta after r - 1 "
              "(the paper's instrument; the gain is held at its value at 90 beyond). Mortality responds through the within-age gradient of death before the next wave in theta "
              f"(a logit on theta, a quadratic in age and theta x age over {n_risk:,} person-waves at risk, {n_deaths:,} deaths: log-odds {slope(b_m, np.array([55.0]))[0]:.2f} per unit of theta at 55); "
              "the healthy share (general health good or better) through the same design fitted to the healthy indicator, applied to each class's person-waves at each age. "
              f"Policies: targeted, class 1 at mu = 1 (its mean health held at the level of r - 1); uniform with the same budget, everyone at mu = pi_1(r) (class 1's share at r, {W[R_GRAD - 20, 0]:.2f} at {R_GRAD}); "
              "uniform at mu = 1. (a), (b) dashed: each class treated on its own at mu = 1. (c) class 3 minus class 1. (d) the standard deviation of age at death over the composition-weighted mixture. "
              f"(e)-(g) each group's LE and HLE at {R_GRAD} is its posterior class composition among the person-waves within five years of {R_GRAD} times the class values, so the gradients are the part the types carry "
              "(within-class differences by group are not modelled); the label gives the group's worst-class share; regions sorted by baseline LE. (h) LE of the top minus the bottom group: degree or higher minus other/none, "
              "top minus bottom income third, and the range across the twelve regions. The within-age mortality gradient is at or below the one the fit's own class hazards imply between types at every age.")
    fig.text(0.01, -0.04, "\n".join(textwrap.wrap(footer, 250)), fontsize=7.2, color=INK2, va="top")
    fig.savefig(FIG / "fig_prevention_le.png", bbox_inches="tight")
    print(f"wrote fig_prevention_le.png and tab_prevention_le.tex to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
