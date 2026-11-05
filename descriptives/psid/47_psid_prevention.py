"""Figure 14 on PSID: returns to prevention, the bill, when the saving arrives, and targeting, on K-means types.

descriptives/47_paper_prevention_main.py on PSID, with the K = 3 partial K-means types on theta in
place of the AR(1) plus "spike" classes. The objects and panels are 42_paper_returns_bayes.py's own
functions (returns, savings, draw_saving_mu, draw_bill, draw_timing_profiles), fed PSID rows:
  cost curve   the pooled cubic in theta on the PSID in-patient cost index in pounds, uncapped
               (31_paper_appendix_extras.fit_cost_curve, the curve drawn in Figure 1)
  typology     the K-means types: one-hot weights, paths the smoothed type means (smooth_path,
               25 members), shares the type shares, as 42's class_set("kmeans")
  (a) S_j(r; 0.1) by type and uniform; (b) the bill C(r) split into mean path, between types and
  within, with the exact saving at mu = 1; (c) the per-age saving at mu = 0.1 from starting at 25
  and 60; (d) targeting at mu = 0.1 on the K-means types refitted with each person's first two rows
  held out (22_psid_kmeans.py theta_first2held, 43_psid_type_from_initial.py): 46_paper_targeting's
  compute() logic copied, with the rules age; + education, income, sex; + theta(t-1);
  + theta(t-1), theta(t-2); and the estimated type (the fitted-window type, which also scores the
  rules). The model-posterior rules of the paper need a mixture and are dropped.
Pounds to 90, no survival weighting, no discounting; the PSID index prices hospital nights only.

Outputs: descriptives/psid/figures/main/fig_prevention.png, artifacts/descriptives/psid/returns_kmeans.csv,
targeting_mu0.1.csv
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from _psid_common import DESC, FIG, K, OBS, contract, cost_rows, hash_test, load_labels, load_trajectories, scores  # noqa: E402
from _paper_common import smooth_path  # noqa: E402
from _style import CLUSTER, GREEN, INK2, ORANGE, PURPLE, apply_style  # noqa: E402

M31 = importlib.import_module("31_paper_appendix_extras")
M42 = importlib.import_module("42_paper_returns_bayes")
M43P = importlib.import_module("43_psid_type_from_initial")
MU = 0.1
Q_GRID = np.round(np.arange(0.05, 1.0001, 0.05), 2)
AGE_RULE, OBS_RULE = "age", "+ education, income, sex"
R1, R12 = r"+ $\theta_{t-1}$", r"+ $\theta_{t-1},\ \theta_{t-2}$"
ORACLE = "the estimated type"


def cost_curve():
    cs = scores().merge(cost_rows(), on=["pidp", "wave"]).dropna(subset=["theta", "cost"])
    coef, dcoef, deg = M31.fit_cost_curve(cs["theta"].to_numpy(float), cs["cost"].to_numpy(float))
    cprime = lambda z: np.polynomial.polynomial.polyval(z, dcoef)  # noqa: E731
    cprime.coef = np.asarray(dcoef, float)
    cfun = lambda z: np.maximum(np.polynomial.polynomial.polyval(z, coef), 0.0)  # noqa: E731
    print(f"PSID cost curve on theta: degree {deg}, coef {np.round(coef, 1)}, {len(cs):,} person-waves")
    return cfun, cprime, deg, coef


def kmeans_set() -> dict:
    """Copy of 42's class_set('kmeans') on the PSID contract."""
    long = contract().sort_values(["pidp", "age"]).reset_index(drop=True)
    L = load_labels("theta")
    long = long[long["pidp"].isin(L.index)].reset_index(drop=True)
    W = np.eye(K)[long["pidp"].map(L).to_numpy().astype(int)]
    traj = load_trajectories("theta")
    paths = np.vstack([smooth_path(q["age"].to_numpy(), q["mean"].to_numpy(), q["count"].to_numpy(), min_count=25).to_numpy()
                       for c in range(K) for q in [traj[traj["cluster"] == c]]])
    return {"rows": long, "W": W, "paths": paths, "shares": np.array([(L == c).mean() for c in range(K)]), "name": "K-means types", "kind": "type"}


def targeting(R: pd.DataFrame, mu: float = MU) -> dict:
    """46_paper_targeting.compute() on the PSID K-means first-two-rows-held-out types."""
    df = M43P.typed().dropna(subset=OBS).reset_index(drop=True)     # education missing for a few people
    test = hash_test(df["pidp"])
    r_i = df["age1"].clip(R.index.min(), R.index.max()).astype(int).to_numpy()
    Pj = np.column_stack([(mu * R[f"P_{k + 1}"] - mu ** 2 * R[f"Q2_{k + 1}"] + mu ** 3 * R[f"Q3_{k + 1}"]).loc[r_i].to_numpy() for k in range(K)])
    W = np.eye(K)[df["type"].to_numpy()]
    realised = (W * Pj).sum(axis=1)
    SETS = {AGE_RULE: [], OBS_RULE: OBS, R1: ["y1", "y1a"], R12: ["y0", "y0a", "y1", "y1a"]}
    probs = {}
    for name, cols in SETS.items():
        X = np.column_stack([df["a0"], df["a0"] ** 2] + [df[c].to_numpy(float) for c in cols])
        probs[name] = LogisticRegression(max_iter=2000, C=1e4).fit(X[~test], df["type"].to_numpy()[~test]).predict_proba(X)
    probs[ORACLE] = W
    rows, uniform = [], float(realised[test].mean())
    rules = list(SETS) + [ORACLE]
    for name in rules:
        e = (probs[name] * Pj).sum(axis=1)[test]; val = realised[test]; order = np.argsort(-e)
        for q in Q_GRID:
            n = max(1, int(round(q * len(order))))
            rows.append({"rule": name, "q": q, "n_treated": n, "return_per_treated": float(val[order[:n]].mean()),
                         "share_of_total": float(val[order[:n]].sum() / val.sum())})
    t = pd.DataFrame(rows); t["uniform"] = uniform
    t.to_csv(DESC / f"targeting_mu{mu:g}.csv", index=False)
    piv = t.pivot(index="q", columns="rule", values="return_per_treated")
    print(f"targeting: {len(df):,} people ({test.sum():,} scored); uniform £{uniform:,.0f}")
    print(piv.loc[[0.1, 0.25, 0.5]].round(0).to_string())
    cols = {AGE_RULE: INK2, OBS_RULE: ORANGE, R1: CLUSTER[1], R12: PURPLE, ORACLE: GREEN}
    return {"t": t, "piv": piv, "uniform": uniform, "rules": rules, "cols": cols, "n": len(df), "MODEL": "none", "ORACLE": ORACLE, "mu": mu}


def draw_targeting(ax, res):
    """46_paper_targeting.draw_targeting's panel (a), copied (that module reads its mode from argv at import)."""
    piv, uniform, cols = res["piv"], res["uniform"], res["cols"]
    for name in res["rules"]:
        ax.plot(piv.index, piv[name] / 1000, color=cols[name], lw=2.2 if name == ORACLE else 1.6, ls="--" if name == ORACLE else "-", label=name)
    ax.axhline(uniform / 1000, color=INK2, lw=1.0, ls=":", label="uniform: everyone treated")
    ax.set_xlabel("share of people treated $q$ (the top $q$ by predicted saving)")
    ax.set_ylabel("saving per treated person (£000)")
    ax.set_title(f"(d) targeting at $\\mu = {res['mu']:g}$", loc="left", fontsize=9.5)
    ax.grid(True, axis="y"); ax.legend(fontsize=7, loc="upper right")


def main() -> int:
    apply_style()
    cfun, cprime, deg, coef = cost_curve()
    cs = kmeans_set(); cs["deg"] = deg
    R = M42.returns(cs["rows"], "theta", cs["W"], cs["paths"], cs["shares"], cfun, cprime)
    R.to_csv(DESC / "returns_kmeans.csv")
    sv = M42.savings(cs["rows"], "theta", cs["W"], cs["paths"], cs["shares"], cprime, mu=MU)
    res = targeting(R)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2), gridspec_kw={"hspace": 0.36, "wspace": 0.24})
    M42.draw_saving_mu(axes[0, 0], R, cs, MU, "a", scale=1000, ylabel="total saving per person (£000)")
    M42.draw_bill(axes[0, 1], R, cs, "b", exact=True, mean_channel=False, scale=1000, ylabel="£000 per person")
    axes[0, 0].set_xlabel("start age $r$"); axes[0, 1].set_xlabel("start age $r$")
    M42.draw_timing_profiles(axes[1, 0], sv, cs, r_list=(25, 60), letter="c", uniform=False, mu=MU)
    draw_targeting(axes[1, 1], res)
    footer = (f"PSID. The K = 3 partial K-means types on theta with the cost curve estimated in pounds on theta (a degree-{deg} polynomial on the PSID in-patient cost index: hospital nights "
              "priced at the UKHLS index's NHS rates, waves 2003-2023, uncapped). Savings are exact for the cubic. "
              "(a) The treatment cost avoided by prevention at mu = 0.1 (a tenth of each type's future decline held back) from start age r, by type, line width proportional to the type share, dashed the uniform policy. "
              "(b) C(r), the expected treatment cost from r to 90 on the observed cross-section, split into the cost at the mean path, what the type differences add and what realised deviations add, "
              "with the saving from uniform prevention at mu = 1. (c) The saving at mu = 0.1 in each year of age a from prevention started at 25 (solid) and 60 (dashed), by type. "
              "Pounds to 90, no survival weighting, no discounting. "
              f"(d) Targeting at mu = 0.1, on the K-means types refitted with each person's first two rows held out ({res['n']:,} people): each rule predicts a person's saving from what is known at t-1 "
              "(a multinomial logit of the type on a quadratic in age plus the set shown, fitted on a 70% split), treats the top q of the other 30% by predicted saving and is scored by the realised "
              "saving, S_j(r; 0.1) of the person's fitted-window type at their age at t-1. Dashed, targeting on the estimated type; dotted, everyone treated.")
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    box = fig.transFigure.inverted().transform(Bbox.union([ax.get_tightbbox(rend) for ax in axes.flat]))
    fig.text(box[0, 0], box[0, 1] - 0.012, "\n".join(textwrap.wrap(footer, 190)), fontsize=7.2, color=INK2, va="top")
    fig.savefig(FIG / "fig_prevention.png", bbox_inches="tight")
    show = R.loc[[25, 50, 70]]
    s01 = pd.DataFrame({f"S_{k + 1}(0.1)": MU * show[f"P_{k + 1}"] - MU ** 2 * show[f"Q2_{k + 1}"] + MU ** 3 * show[f"Q3_{k + 1}"] for k in range(K)})
    s01["uniform(0.1)"] = s01.to_numpy() @ cs["shares"]
    print(pd.concat([show[["C", "C_mean", "C_between", "C_within", "S", "S_over_C"]], s01], axis=1).round(2).to_string())
    print("shares", np.round(cs["shares"], 3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
