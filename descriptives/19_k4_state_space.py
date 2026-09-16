"""Four classes under the state-space specification, set against three.

The K=3 state-space fits (AR(1) latent state plus one-period measurement
error) are written up in section 9 of the health measures note. This script
produces the same material at K=4 for the two physical measures that were
refitted, P-FULL and P-FUNC, so the two class counts can be read side by side:

  specification table   shares, persistence, noise, convergence
  trajectories          class mean paths, with per-age class composition
  spread                level gaps and declines, as in section 9
  signal share          persistent-state share of observation variance

and two things section 9 did not need but a change of K does:

  correspondence        which K=3 class the fourth class is carved from. The
                        K=3 and K=4 fits of each measure run on the SAME
                        contract, so the same people sit in the same order and
                        their class assignments can be cross-tabulated
                        person by person.
  in-sample fit         marginal log likelihood at the posterior mean, BIC,
                        and classification entropy. With ~39,000-50,000 people
                        BIC is expected to favour more classes almost
                        regardless, so the entropy and the correspondence
                        matter more for whether the class is real.

Class posteriors are recomputed from posterior-mean parameters with the same
Kalman filter as the Stan ar_mode 2 branch. Everything here is K-general;
11_class_composition.py hard-codes K=3 and feeds downstream scripts, so it is
left alone.

Outputs: measuring_health/figures/fig_k4_trajectories.png
         measuring_health/figures/fig_k4_structure.png
         artifacts/descriptives/k4_classes.csv
         artifacts/descriptives/k4_fit.csv
         artifacts/descriptives/k4_correspondence.csv
         artifacts/descriptives/k4_composition_by_age.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).parent))
from _style import INK, INK2, SURFACE, apply_style  # noqa: E402

from prevention_health_clustering.config import (
    ARTIFACTS_DIR, PROCESSED_DATA_DIR, ROOT_DIR)
from prevention_health_clustering.models.registry import get_model
from prevention_health_clustering.runner.fit import build_payload

FIG_DIR = ROOT_DIR / "measuring_health" / "figures"
CONTRACTS = PROCESSED_DATA_DIR / "contracts"
AGES = np.arange(20, 91)
A = (AGES - 55) / 10.0

# Validated ordinal ramps (dataviz validate_palette.js --ordinal, light surface).
# The K=4 ramp shares both endpoints with the note's K=3 ramp, so the worst and
# best classes carry the same colour at either class count.
RAMP = {3: ["#08306b", "#3182bd", "#6baed6"],
        4: ["#08306b", "#1c5a9e", "#3d8bc9", "#6baed6"]}
# Validated categorical pair for K identity (CVD dE 24.7), plus marker shape.
K_COLOR = {3: "#2a78d6", 4: "#eb6834"}
K_MARK = {3: "o", 4: "s"}

MEASURES = {
    "P-FULL": {"contract": "physgrm_lifecycle_20_89_minobs3_v1",
               3: ("physgrm-full-ssm", ARTIFACTS_DIR / "ssm" / "physfull-ssm"),
               4: ("physgrm-full-ssm-k4", ARTIFACTS_DIR / "k4" / "physfull-ssm-k4")},
    "P-FUNC": {"contract": "physfunc_lifecycle_20_89_minobs3_v1",
               3: ("physgrm-func-ssm", ARTIFACTS_DIR / "ssm" / "physfunc-ssm"),
               4: ("physgrm-func-ssm-k4", ARTIFACTS_DIR / "k4" / "physfunc-ssm-k4")},
}


def load_params(path: Path, K: int) -> dict:
    j = json.loads((path / "run_summary.json").read_text())
    p = j["params"]
    g = lambda n: p[n]["mean"]
    return {
        "theta": np.array([g(f"theta[{k}]") for k in range(1, K + 1)]),
        "coef": np.array([[g(f"coef[1,{k},{q}]") for q in (1, 2, 3)]
                          for k in range(1, K + 1)]),
        "rho": np.array([g(f"rho[{k}]") for k in range(1, K + 1)]),
        "sigma": g("sigma[1,1]"),
        "sigma_meas": g("sigma_meas[1]"),
        "rhat": j["max_structural_rhat"],
        "worst": j["worst_param"],
        "wall": j["wall_hours"],
        "n_person": j["n_person"],
    }


def kalman_person_lp(Y, X, coef, sigma, sigma_meas, rho, gap, fs, fe):
    """(persons, K) log density under an AR(1) state plus i.i.d. noise.

    Mirrors the ar_mode 2 branch of mixture_gaussian_panel.stan, vectorised
    over people and classes and looped over the within-person time index.
    """
    K = coef.shape[0]
    n_person = len(fs)
    lens = fe - fs + 1
    T = int(lens.max())
    idx = np.full((n_person, T), -1, dtype=np.int64)
    for i in range(n_person):
        idx[i, :lens[i]] = np.arange(fs[i] - 1, fe[i])
    mask = idx >= 0
    safe = np.where(mask, idx, 0)
    mu = (X @ coef.T)[safe]                   # (persons, T, K)
    Yp, gp = Y[safe], gap[safe]
    v_stat = sigma ** 2 / (1 - rho ** 2)
    a = np.zeros((n_person, K))
    Pv = np.tile(v_stat, (n_person, 1))
    total = np.zeros((n_person, K))
    for t in range(T):
        m = mask[:, t][:, None]
        if t > 0:
            g = gp[:, t][:, None]
            rg = rho[None, :] ** g
            a = rg * a
            Pv = rg ** 2 * Pv + v_stat[None, :] * np.maximum(
                1 - rho[None, :] ** (2 * g), 1e-9)
        F = Pv + sigma_meas ** 2
        v = Yp[:, t][:, None] - mu[:, t, :] - a
        total += np.where(m, -0.5 * (np.log(2 * np.pi * F) + v ** 2 / F), 0.0)
        Kg = Pv / F
        a = np.where(m, a + Kg * v, a)
        Pv = np.where(m, Pv - Kg * Pv, Pv)
    return total


def analyse(measure: str, K: int, pl, prm: dict) -> dict:
    d = pl.data
    X, Y = np.asarray(d["X"]), np.asarray(d["y"][0])
    fs, fe = np.asarray(d["fit_start"]), np.asarray(d["fit_end"])
    lp = kalman_person_lp(Y, X, prm["coef"], prm["sigma"], prm["sigma_meas"],
                          prm["rho"], np.asarray(d["age_gap"]), fs, fe)
    joint = np.log(prm["theta"])[None, :] + lp
    ll_person = logsumexp(joint, axis=1)
    w = np.exp(joint - ll_person[:, None])

    # composition of the person-waves observed at each age
    n_person = len(fs)
    person = np.zeros(d["N_obs"], dtype=np.int64)
    for i in range(n_person):
        person[fs[i] - 1:fe[i]] = i
    ages = np.round(X[:, 1] * 10 + 55).astype(int)
    comp = (pd.DataFrame(w[person], columns=[f"class{k + 1}" for k in range(K)])
            .assign(age=ages).groupby("age").mean().reset_index())

    # relative entropy: 1 = perfectly separated classes, 0 = no separation
    ent = -(w * np.log(np.clip(w, 1e-300, None))).sum()
    rel_entropy = 1 - ent / (n_person * np.log(K))
    n_par = 5 * K + 1       # (K-1) shares + K intercepts + 2K slopes + sigma + K rho + sigma_meas
    ll = float(ll_person.sum())
    return {"measure": measure, "K": K, "prm": prm, "w": w,
            "modal": w.argmax(axis=1), "ids": np.asarray(pl.person_ids),
            "comp": comp, "ll": ll, "n_par": n_par,
            "bic": -2 * ll + n_par * np.log(n_person),
            "rel_entropy": float(rel_entropy),
            "mean_max_post": float(w.max(axis=1).mean()),
            "n_person": n_person}


def class_table(r: dict) -> pd.DataFrame:
    prm, K = r["prm"], r["K"]
    mu = np.column_stack([prm["coef"][k, 0] + prm["coef"][k, 1] * A
                          + prm["coef"][k, 2] * A ** 2 for k in range(K)])
    v = prm["sigma"] ** 2 / (1 - prm["rho"] ** 2)
    share = v / (v + prm["sigma_meas"] ** 2)
    rows = []
    for k in range(K):
        rows.append({"measure": r["measure"], "K": K, "class": k + 1,
                     "share": prm["theta"][k], "alpha": prm["coef"][k, 0],
                     "beta": prm["coef"][k, 1], "gamma": prm["coef"][k, 2],
                     "rho": prm["rho"][k], "signal_share": share[k],
                     "level_30": mu[AGES == 30, k][0],
                     "level_80": mu[AGES == 80, k][0],
                     "drop_30_80": mu[AGES == 80, k][0] - mu[AGES == 30, k][0]})
    return pd.DataFrame(rows)


def main() -> int:
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = ARTIFACTS_DIR / "descriptives"
    results = {}
    for measure, cfg in MEASURES.items():
        long = pd.read_csv(CONTRACTS / cfg["contract"] / "long.csv")
        for K in (3, 4):
            model, path = cfg[K]
            spec = get_model(model)
            assert spec.n_classes == K, (model, spec.n_classes)
            pl = build_payload(spec, long)
            prm = load_params(path, K)
            r = analyse(measure, K, pl, prm)
            results[(measure, K)] = r
            print(f"{measure} K={K}: {r['n_person']:,} people, "
                  f"R-hat {prm['rhat']:.4f}, LL {r['ll']:,.1f}, "
                  f"rel. entropy {r['rel_entropy']:.3f}")

    # ---- tables ---------------------------------------------------------
    classes = pd.concat([class_table(r) for r in results.values()],
                        ignore_index=True)
    classes.to_csv(out_dir / "k4_classes.csv", index=False)
    print("\nclasses:")
    print(classes.round(3).to_string(index=False))

    fit_rows = []
    for measure in MEASURES:
        r3, r4 = results[(measure, 3)], results[(measure, 4)]
        assert np.array_equal(r3["ids"], r4["ids"]), \
            f"{measure}: K=3 and K=4 payloads are not on the same people"
        for r in (r3, r4):
            p = r["prm"]
            fit_rows.append({
                "measure": measure, "K": r["K"], "n_person": r["n_person"],
                "max_rhat": p["rhat"], "worst_param": p["worst"],
                "wall_hours": p["wall"], "sigma": p["sigma"],
                "sigma_meas": p["sigma_meas"], "loglik": r["ll"],
                "n_par": r["n_par"], "bic": r["bic"],
                "rel_entropy": r["rel_entropy"],
                "mean_max_posterior": r["mean_max_post"]})
    fit = pd.DataFrame(fit_rows)
    fit.to_csv(out_dir / "k4_fit.csv", index=False)
    print("\nfit:")
    print(fit.round(4).to_string(index=False))
    for measure in MEASURES:
        f3 = fit[(fit.measure == measure) & (fit.K == 3)].iloc[0]
        f4 = fit[(fit.measure == measure) & (fit.K == 4)].iloc[0]
        print(f"  {measure}: K=4 gains {f4.loglik - f3.loglik:,.1f} log-lik "
              f"for {int(f4.n_par - f3.n_par)} parameters; "
              f"BIC change {f4.bic - f3.bic:+,.1f} (negative favours K=4)")

    corr_rows, corr = [], {}
    for measure in MEASURES:
        r3, r4 = results[(measure, 3)], results[(measure, 4)]
        ct = pd.crosstab(r3["modal"] + 1, r4["modal"] + 1)
        ct = ct.reindex(index=range(1, 4), columns=range(1, 5), fill_value=0)
        rowpct = ct.div(ct.sum(axis=1), axis=0) * 100
        corr[measure] = (ct, rowpct)
        # posterior-weighted version, as a check on hard assignment
        soft = r3["w"].T @ r4["w"]
        soft_pct = soft / soft.sum(axis=1, keepdims=True) * 100
        print(f"\n{measure}: K=3 class (rows) -> K=4 class (cols), % of row, "
              f"modal assignment")
        print(rowpct.round(1).to_string())
        print(f"  posterior-weighted, same layout:\n"
              + pd.DataFrame(soft_pct, index=range(1, 4),
                             columns=range(1, 5)).round(1).to_string())
        for i in range(1, 4):
            for j in range(1, 5):
                corr_rows.append({"measure": measure, "k3_class": i,
                                  "k4_class": j, "count": int(ct.loc[i, j]),
                                  "row_pct_modal": rowpct.loc[i, j],
                                  "row_pct_soft": soft_pct[i - 1, j - 1]})
    pd.DataFrame(corr_rows).to_csv(out_dir / "k4_correspondence.csv",
                                   index=False)

    comp_rows = []
    for (measure, K), r in results.items():
        c = r["comp"].copy()
        c.insert(0, "K", K)
        c.insert(0, "measure", measure)
        comp_rows.append(c)
    pd.concat(comp_rows, ignore_index=True).to_csv(
        out_dir / "k4_composition_by_age.csv", index=False)

    # ---- figure 1: trajectories with composition strips -------------------
    fig = plt.figure(figsize=(12.6, 8.6))
    gs = fig.add_gridspec(4, 2, height_ratios=[3.2, 0.7, 3.2, 0.7],
                          hspace=0.42, wspace=0.16)
    for row, measure in enumerate(MEASURES):
        for col, K in enumerate((3, 4)):
            r = results[(measure, K)]
            prm = r["prm"]
            ax = fig.add_subplot(gs[2 * row, col])
            axc = fig.add_subplot(gs[2 * row + 1, col], sharex=ax)
            for k in range(K):
                mu = (prm["coef"][k, 0] + prm["coef"][k, 1] * A
                      + prm["coef"][k, 2] * A ** 2)
                ax.plot(AGES, mu, color=RAMP[K][k],
                        lw=0.8 + 4.0 * prm["theta"][k],
                        label=f"class {k + 1}: {prm['theta'][k]:.0%}")
            ax.axhline(0, color="#cccccc", lw=0.7, ls=":")
            ax.set_title(f"{measure}, K={K}", fontsize=9.5)
            ax.legend(fontsize=7, loc="lower left", frameon=False)
            ax.grid(True, axis="y")
            ax.set_ylim(-2.2, 1.3)
            ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel("standardised units", fontsize=8.5)
            c = r["comp"].sort_values("age")
            axc.stackplot(c["age"], *[c[f"class{k + 1}"] for k in range(K)],
                          colors=RAMP[K], edgecolor=SURFACE, linewidth=0.8)
            axc.set_ylim(0, 1)
            axc.set_yticks([])
            axc.set_xlim(20, 90)
            axc.spines["left"].set_visible(False)
            if row == 1:
                axc.set_xlabel("age")
    fig.suptitle("Class trajectories under the state-space specification: "
                 "three classes against four", fontweight="bold", y=0.995)
    fig.text(0.01, 0.005,
             "AR(1) latent health state plus one-period measurement error; physical GRM in its P-FULL (functioning plus\n"
             "diagnosis groups) and P-FUNC (functioning only) forms, each standardised over its fitted sample. Line width is\n"
             "proportional to the class share; class 1 is worst health. The strip beneath each panel is the class composition\n"
             "of the person-waves observed at each age. The K=3 and K=4 fits of a measure run on the same people.",
             fontsize=7.4, color=INK2, va="bottom")
    fig.tight_layout(rect=[0, 0.07, 1, 0.98])
    p1 = FIG_DIR / "fig_k4_trajectories.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)

    # ---- figure 2: where the fourth class comes from, and what it is -----
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6),
                             gridspec_kw={"width_ratios": [1.15, 1, 1]})
    for row, measure in enumerate(MEASURES):
        ct, rowpct = corr[measure]
        ax = axes[row, 0]
        ax.imshow(rowpct.to_numpy(), cmap="Greys", vmin=0, vmax=100,
                  aspect="auto")
        for i in range(3):
            for j in range(4):
                val = rowpct.iloc[i, j]
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8.5, color="white" if val > 55 else INK)
        ax.set_xticks(range(4), [f"{j}" for j in range(1, 5)])
        ax.set_yticks(range(3), [f"{i}" for i in range(1, 4)])
        ax.set_xlabel("class at K=4")
        ax.set_ylabel("class at K=3")
        ax.set_title(f"({'ab'[row]}) {measure}: K=3 class -> K=4 class,\n"
                     "% of each K=3 class", fontsize=9.5)
        ax.grid(False)

        for col, (key, ylab, ttl) in enumerate(
                (("rho", "persistence rho", "persistence"),
                 ("signal_share", "persistent share of variance",
                  "signal share")), start=1):
            ax = axes[row, col]
            for K in (3, 4):
                t = classes[(classes.measure == measure) & (classes.K == K)]
                ax.plot(t["alpha"], t[key], color=K_COLOR[K], lw=2,
                        marker=K_MARK[K], ms=8, label=f"K={K}",
                        markeredgecolor=SURFACE, markeredgewidth=1.2)
            ax.set_xlabel("class intercept (level at age 55)")
            ax.set_ylabel(ylab)
            ax.set_title(f"({'cdef'[2 * row + col - 1]}) {measure}: {ttl} "
                         "by class level", fontsize=9.5)
            ax.grid(True)
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Where the fourth class comes from, and what it is",
                 fontweight="bold", y=0.995)
    fig.text(0.01, 0.005,
             "Left: modal class assignment of the same people at K=3 (rows) and K=4 (columns), as a percentage of each K=3 class.\n"
             "Middle and right: each class's persistence and persistent-state share of observation variance, plotted against the\n"
             "class's level so the two class counts share an axis; the rightmost point at K=4 is the added class.",
             fontsize=7.4, color=INK2, va="bottom")
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    p2 = FIG_DIR / "fig_k4_structure.png"
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    print(f"\nwrote {p1}\nwrote {p2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
