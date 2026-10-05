"""Figure: the multidimensional fits on the paper's measure (physical, mental, mortality).

Four K = 3 fits from clustering/runs/multidim_health_queue.py on the multidim
health contract (38,181 people): the physical measure (theta or h) and the
mental GRM theta as Gaussian channels, under AR(1) latent state plus a
one-period "spike" (persistence by class and channel) or independent
residuals, with a Gompertz-Makeham mortality hazard by class. One row per
fit, four columns:
  (1) the physical class paths in the variant's units, with the observed
      class means dotted (posterior-weighted mean at each age)
  (2) the mental class paths in mental-theta units, observed means dotted
  (3) the yearly mortality hazard by class, log scale, with the
      posterior-weighted crude death rate by five-year band dotted
  (4) the posterior class composition of the person-waves observed at each age
Posterior means are read from the chain csvs; a fit whose chains split into
modes is read over the chains that share the better mode (CHAINS), as the
K = 5 fit on h is in 36_paper_trajectories.py.

Per-person class posteriors are recomputed here (Kalman filter per channel
plus the Gompertz survival likelihood, at the posterior mean) and cached.

Outputs: paper/figures/fig_multidim_trajectories.png
         artifacts/descriptives/paper_multidim.csv (class rows)
         artifacts/descriptives/paper_multidim_posteriors_{tag}.parquet
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import AGES, DESC, FIG, LABEL, MAX_AGE, MIN_AGE, kalman_person_lp  # noqa: E402
from _style import CLUSTER, INK2, apply_style  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR  # noqa: E402

MD = ARTIFACTS_DIR / "multidim-health"
CONTRACT = PROCESSED_DATA_DIR / "contracts" / "multidim_health_20_89_minobs3_v1"
K = 3
A = (AGES - 55) / 10.0
# (tag, physical variant, specification label, chains read)
FITS = [
    ("theta-ssm-mort", "theta", 'AR(1) + "spike"', None),
    ("h-ssm-mort", "h", 'AR(1) + "spike"', [1, 2, 3]),
    ("theta-base-mort", "theta", "independent residuals", None),
    ("h-base-mort", "h", "independent residuals", None),
]
# --cohort: the theta AR(1) + "spike" fit with birth-decade level shifts on both channels, common to the
# classes; class paths are drawn at the 1950s decade and the likelihood nets the shifts out of the scores.
FITS_COHORT = [("theta-ssm-mort-cohort", "theta", 'AR(1) + "spike", cohort shifts', None)]
COHORT_REF = 1950
# --k: the theta AR(1) + "spike" fit at K = 3, 4 and 5 (clustering/runs/multidim_health_k45_queue.py), one row each
FITS_K = [("theta-ssm-mort", "theta", 'AR(1) + "spike", K = 3', None, 3),
          ("theta-ssm-mort-k4", "theta", 'AR(1) + "spike", K = 4', None, 4),
          ("theta-ssm-mort-k5", "theta", 'AR(1) + "spike", K = 5', None, 5)]
PALETTE = {3: CLUSTER, 4: ["#08306b", "#2171b5", "#4292c6", "#9ecae1"],
           5: ["#08306b", "#08519c", "#2171b5", "#4292c6", "#9ecae1"]}     # worst -> best, as 36_paper_trajectories.py
# h-ssm-mort: chains 1-3 agree to three decimals; chain 4 sits in the mode with the
# class 2 and 3 intercepts pressed together on the ordering constraint, about 2,950
# log-posterior units lower, and is dropped.
BANDS = [(20, 34), (35, 44), (45, 54), (55, 64), (65, 74), (75, 84), (85, 90)]
MIN_PERSON_YEARS = 300.0


def chain_means(fit_dir: Path, chains) -> tuple[dict, float, list]:
    files = sorted((fit_dir / "chains").glob("*_[0-9].csv"))
    if chains is not None:
        files = [f for f in files if int(f.stem.rsplit("_", 1)[1]) in set(chains)]
    dfs = [pd.read_csv(f, comment="#") for f in files]
    struct = [c for c in dfs[0].columns
              if re.match(r"^(theta|coef|sigma|rho|sigma_meas|log_b|gomp_slope|makeham|cohort_effect)\.", c)]
    pooled = pd.concat(dfs)
    means, worst = {c: float(pooled[c].mean()) for c in struct}, 0.0
    for c in struct:
        halves = [x[: len(x) // 2] for x in (d[c].to_numpy() for d in dfs)] + [x[len(x) // 2:] for x in (d[c].to_numpy() for d in dfs)]
        n = len(halves[0]); mu = np.array([h.mean() for h in halves]); vs = np.array([h.var(ddof=1) for h in halves])
        W = vs.mean(); B = n * mu.var(ddof=1)
        if W > 0:
            worst = max(worst, float(np.sqrt(((n - 1) / n * W + B / n) / W)))
    return means, worst, [float(d["lp__"].mean()) for d in dfs]


def gompertz_person_lp(d: dict, log_b, slope, makeham, kc: int = K) -> np.ndarray:
    age = np.asarray(d["mort_age"], float); gap = np.asarray(d["mort_gap"], float)
    obs = np.asarray(d["mort_obs"]) == 1; y = np.asarray(d["mort_y"]) == 1
    fs, fe = np.asarray(d["fit_start"]), np.asarray(d["full_end"])
    out = np.zeros((len(fs), kc))
    for k in range(kc):
        H = makeham * gap + np.exp(log_b[k] - np.log(slope[k]) + slope[k] * age) * np.expm1(slope[k] * gap)
        ll = np.where(y, np.log(-np.expm1(-H)), -H) * obs
        person = np.repeat(np.arange(len(fs)), fe - fs + 1)
        out[:, k] = np.bincount(person, weights=ll, minlength=len(fs))
    return out


def load_fit(tag: str, v: str, chains, kc: int = K):
    fit_dir = MD / tag
    d = json.load(open(fit_dir / "stan_data.json"))
    summ = json.load(open(fit_dir / "run_summary.json"))
    m, rhat, lps = chain_means(fit_dir, chains)
    mom = summ["channel_moments"]; ch = summ["channels"]
    theta = np.array([m[f"theta.{k}"] for k in range(1, kc + 1)])
    coef = np.array([[[m[f"coef.{c}.{k}.{p}"] for p in (1, 2, 3)] for k in range(1, kc + 1)] for c in (1, 2)])
    sigma = np.array([m[f"sigma.1.{c}"] for c in (1, 2)])
    ar = d["ar_mode"]
    rho = np.array([[m[f"rho.{k}.{c}"] for c in (1, 2)] for k in range(1, kc + 1)]) if ar else np.zeros((kc, 2))
    smeas = np.array([m[f"sigma_meas.{c}"] for c in (1, 2)]) if ar == 2 else np.zeros(2)
    log_b = np.array([m[f"log_b.{k}"] for k in range(1, kc + 1)])
    slope = np.array([m[f"gomp_slope.{k}"] for k in range(1, kc + 1)])
    makeham = m["makeham.1"]
    X = np.asarray(d["X"]); Y = np.asarray(d["y"])
    # cohort shifts: net them out of the scores so the class likelihood is as Stan computes it,
    # and draw the class paths at the reference decade
    n_coh = d.get("N_cohort", 1); ce = np.zeros((2, n_coh)); ref_shift = np.zeros(2); Y_raw = Y
    if n_coh > 1:
        for c in range(2):
            ce[c, 1:] = [m[f"cohort_effect.{c + 1}.{j}"] for j in range(2, n_coh + 1)]
        cid = np.asarray(d["cohort_id"]) - 1
        Y_raw, Y = Y, Y - ce[:, cid]
        ref_shift = ce[:, summ["cohort_decades"].index(COHORT_REF)]
    fs, fe = np.asarray(d["fit_start"]), np.asarray(d["fit_end"])
    person = np.repeat(np.arange(len(fs)), fe - fs + 1)
    per = np.zeros((len(fs), kc))
    for c in range(2):
        if ar == 2:
            per += kalman_person_lp(Y[c], X, coef[c], sigma[c], smeas[c], rho[:, c], np.asarray(d["age_gap"], float), fs, fe)
        else:
            mu = X @ coef[c].T
            lp = -0.5 * np.log(2 * np.pi) - np.log(sigma[c]) - 0.5 * ((Y[c][:, None] - mu) / sigma[c]) ** 2
            np.add.at(per, person, lp)
    per += gompertz_person_lp(d, log_b, slope, makeham, kc)
    un = np.log(theta)[None, :] + per
    w = np.exp(un - un.max(axis=1, keepdims=True)); w /= w.sum(axis=1, keepdims=True)
    long = pd.read_csv(CONTRACT / "long.csv").sort_values(["pidp", "age"]).reset_index(drop=True)
    assert len(long) == len(Y[0])
    z = (long[ch[0]].to_numpy() - mom[ch[0]]["mean"]) / mom[ch[0]]["sd"]
    assert np.abs(z - Y_raw[0]).max() < 1e-6, "payload rows do not match the contract"
    pids = long.groupby("pidp", sort=False)["pidp"].first().to_numpy()
    cols = [f"class{k + 1}" for k in range(kc)]
    post = pd.DataFrame(w, columns=cols).assign(pidp=pids)
    adj = [(ce[c, cid] - ref_shift[c]) * mom[ch[c]]["sd"] if n_coh > 1 else 0.0 for c in range(2)]
    rows = pd.DataFrame(w[person], columns=cols).assign(age=long["age"].to_numpy(), y0=long[ch[0]].to_numpy() - adj[0],
                                                         y1=long[ch[1]].to_numpy() - adj[1],
                                                         dead=np.asarray(d["mort_y"], float), gap=np.asarray(d["mort_gap"], float))
    comp = rows.groupby("age")[cols].mean()
    observed = {}
    for j, yc in enumerate(("y0", "y1")):
        obs = {}
        for c in cols:
            g = pd.DataFrame({"age": rows["age"], "wy": rows[c] * rows[yc], "w": rows[c]}).groupby("age").sum()
            obs[c] = (g["wy"] / g["w"]).where(g["w"] >= 25)
        observed[j] = pd.DataFrame(obs)
    crude = {}
    for c in cols:
        rec = []
        for lo, hi in BANDS:
            s = rows[rows["age"].between(lo, hi)]
            py = (s[c] * s["gap"]).sum(); ev = (s[c] * s["dead"]).sum()
            rec.append(((lo + hi) / 2, ev / py if py >= MIN_PERSON_YEARS else np.nan))
        crude[c] = rec
    paths = [np.vstack([mom[ch[c]]["mean"] + mom[ch[c]]["sd"] * (b[0] + ref_shift[c] + b[1] * A + b[2] * A ** 2) for b in coef[c]]) for c in range(2)]
    hazard = np.vstack([makeham + np.exp(log_b[k] + slope[k] * (AGES - 55)) for k in range(kc)])
    sig_share = np.array([[sigma[c] ** 2 / (1 - rho[k, c] ** 2) / (sigma[c] ** 2 / (1 - rho[k, c] ** 2) + smeas[c] ** 2)
                           for c in range(2)] for k in range(kc)]) if ar == 2 else np.full((kc, 2), np.nan)
    return {"tag": tag, "v": v, "ar": ar, "K": kc, "share": theta, "rho": rho, "sigma": sigma, "sigma_meas": smeas, "signal": sig_share,
            "log_b": log_b, "slope": slope, "makeham": makeham, "paths": paths, "hazard": hazard, "post": post,
            "cohort_shift": ce, "cohort_decades": summ.get("cohort_decades", []),
            "comp": comp, "observed": observed, "crude": crude, "rhat": rhat, "lps": lps, "chains": chains,
            "wall": summ["wall_hours"], "n_person": d["N_person"], "events": int(np.sum(d["mort_y"]))}


def agreement(fit) -> dict:
    """Agreement of the physical typology with the single-channel K = 3 fit on the same measure (K = 3 fits only)."""
    tag = "ssm" if fit["ar"] == 2 else "base"
    f = DESC / f"paper_bayes_{tag}_posteriors.parquet"
    if not f.exists() or fit["K"] != K:
        return {}
    uni = pd.read_parquet(f); uni = uni[uni["variant"] == fit["v"]]
    cols = [f"class{k + 1}" for k in range(fit["K"])]
    m = fit["post"].merge(uni[["pidp"] + cols], on="pidp", suffixes=("", "_uni"))
    a = m[cols].to_numpy().argmax(1); b = m[[c + "_uni" for c in cols]].to_numpy().argmax(1)
    return {"ari_vs_univariate": adjusted_rand_score(a, b), "same_class": float((a == b).mean()), "n_compared": len(m)}


def main() -> int:
    apply_style()
    # the paper's figure is theta alone (AR(1) + "spike", then independent residuals); --h the appendix version
    V_ONLY = "h" if "--h" in sys.argv else "theta"
    COHORT = "--cohort" in sys.argv
    KMODE = "--k" in sys.argv
    if KMODE:
        FITS_V = [f[:4] for f in FITS_K if (MD / f[0] / "run_summary.json").exists()]
        fits = [load_fit(tag, v, ch, kc) for tag, v, _, ch, kc in FITS_K if (MD / tag / "run_summary.json").exists()]
    else:
        FITS_V = FITS_COHORT if COHORT else [f for f in FITS if f[1] == V_ONLY]
        fits = [load_fit(tag, v, ch) for tag, v, _, ch in FITS_V]
    fig, axes = plt.subplots(len(FITS_V), 4, figsize=(15, 3.3 * len(FITS_V)), squeeze=False,
                             gridspec_kw={"wspace": 0.28, "hspace": 0.35, "width_ratios": [1, 1, 1, 0.8]})
    rows = []
    for r, ((tag, v, spec, _), f) in enumerate(zip(FITS_V, fits)):
        Kc = f["K"]; CLUSTER = PALETTE[Kc]
        for c in range(2):
            ax = axes[r, c]
            for k in range(Kc):
                lab = f"{k + 1}: {f['share'][k]:.0%}" + (f", $\\rho$ {f['rho'][k, c]:.2f}" if f["ar"] else "")
                ax.plot(AGES, f["paths"][c][k], color=CLUSTER[k], lw=1.0 + 3.5 * f["share"][k], label=lab)
                obs = f["observed"][c][f"class{k + 1}"]
                ax.plot(obs.index, obs.to_numpy(), color=CLUSTER[k], lw=1.0, ls=":")
            ax.legend(fontsize=6.8, loc="lower left" if c == 0 else ("upper right" if COHORT else "lower right"), handlelength=1.6)
            ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
            if r == 0:
                ax.set_title(["physical channel", "mental channel"][c], fontsize=10, loc="left")
            ax.set_ylabel(f"{LABEL[v]}, {spec}" if c == 0 else r"mental $\theta$", fontsize=9)
        ax = axes[r, 2]
        for k in range(Kc):
            ax.plot(AGES, f["hazard"][k], color=CLUSTER[k], lw=1.0 + 3.5 * f["share"][k],
                    label=f"{k + 1}: slope {f['slope'][k]:.3f}")
            xs, ys = zip(*f["crude"][f"class{k + 1}"])
            ax.plot(xs, ys, color=CLUSTER[k], lw=0, marker="o", ms=3.5, mfc="none")
        ax.set_yscale("log"); ax.set_ylim(1e-4, 0.3); ax.grid(True, axis="y"); ax.set_xlim(MIN_AGE, MAX_AGE)
        ax.legend(fontsize=6.8, loc="upper left", handlelength=1.6)
        ax.set_ylabel("yearly hazard", fontsize=9)
        if r == 0:
            ax.set_title("mortality: Gompertz-Makeham hazard", fontsize=10, loc="left")
        ax = axes[r, 3]
        comp = f["comp"]
        ax.stackplot(comp.index, *[comp[c] for c in comp.columns], colors=CLUSTER, alpha=0.9)
        ax.set_ylim(0, 1); ax.set_xlim(MIN_AGE, MAX_AGE); ax.set_ylabel("share observed", fontsize=9)
        if r == 0:
            ax.set_title("class composition by age", fontsize=10, loc="left")
        for ax in axes[r]:
            if r == len(FITS_V) - 1:
                ax.set_xlabel("age")
        agree = agreement(f)
        for k in range(Kc):
            rows.append({"fit": tag, "variant": v, "spec": spec, "class": k + 1, "share": f["share"][k],
                         "rho_phys": f["rho"][k, 0], "rho_ment": f["rho"][k, 1],
                         "signal_phys": f["signal"][k, 0], "signal_ment": f["signal"][k, 1],
                         "sigma_phys": f["sigma"][0], "sigma_ment": f["sigma"][1],
                         "sigma_meas_phys": f["sigma_meas"][0], "sigma_meas_ment": f["sigma_meas"][1],
                         "phys_20": f["paths"][0][k][0], "phys_55": f["paths"][0][k][35], "phys_90": f["paths"][0][k][-1],
                         "ment_20": f["paths"][1][k][0], "ment_55": f["paths"][1][k][35], "ment_90": f["paths"][1][k][-1],
                         "log_b_55": f["log_b"][k], "gomp_slope": f["slope"][k], "makeham": f["makeham"],
                         "hazard_55": f["hazard"][k][35], "hazard_85": f["hazard"][k][65],
                         "rhat": f["rhat"], "chains": "all" if f["chains"] is None else ",".join(map(str, f["chains"])),
                         "wall_hours": f["wall"], **agree})
        f["post"].to_parquet(DESC / f"paper_multidim_posteriors_{tag}.parquet", index=False)
        if COHORT:
            print(f"{tag}: birth-decade shifts (oldest = 0), decades {f['cohort_decades']}")
            for c, lab in enumerate(("physical", "mental")):
                print(f"  {lab}: " + " ".join(f"{x:+.3f}" for x in f["cohort_shift"][c]))
        print(f"{tag}: rhat {f['rhat']:.4f} over chains {f['chains'] or 'all'}, lp by chain {np.round(f['lps'], 1)}; "
              f"shares {np.round(f['share'], 3)}; rho phys {np.round(f['rho'][:, 0], 2)} ment {np.round(f['rho'][:, 1], 2)}; "
              f"hazard ratio 1/{f['K']} at 55 {f['hazard'][0][35] / f['hazard'][-1][35]:.1f}x, at 85 {f['hazard'][0][65] / f['hazard'][-1][65]:.1f}x; "
              + (f"ARI vs single-channel {agree['ari_vs_univariate']:.2f}, same class {agree['same_class']:.0%}" if agree else ""))
    n = fits[0]["n_person"]; ev = fits[0]["events"]
    fig.text(0.01, 0.005,
             f"Multidimensional K = 3 mixtures on the multidim health contract ({n:,} people, the health contract's people with a mental score on every row; "
             f"{ev:,} deaths). Rows: the physical measure and the stochastic specification. Columns: the physical and mental class paths at the posterior mean\n"
             "in each channel's units (dotted: the observed class mean at each age, posterior-weighted), the class-specific Gompertz-Makeham yearly hazard "
             "(circles: the posterior-weighted crude death rate by five-year band), and the posterior class composition of the person-waves observed at each age.\n"
             "Under AR(1) + \"spike\" the legend gives each class's persistence on that channel; the innovation and \"spike\" scales are common to the classes. "
             "The Makeham constant is below 0.0001 in every fit."
             + (" Class paths and observed class means are at the 1950s birth decade, every score net of its estimated decade shift." if COHORT else "")
             + (" Rows: the class count K = 3, 4 and 5 under AR(1) + \"spike\"; classes ordered worst to best." if KMODE else "")
             + (" h + AR(1) + \"spike\" is read over the three chains that share a mode; the fourth sat in the mode "
             "with the class 2 and 3 intercepts pressed together on the ordering constraint, about 2,950 log-posterior units lower." if V_ONLY == "h" else ""),
             fontsize=7.2, color=INK2, va="top")
    fig.subplots_adjust(top=0.93, bottom=0.17 if len(FITS_V) > 1 else 0.30, left=0.05, right=0.99)
    sfx = "_by_K" if KMODE else "_cohort" if COHORT else ("" if V_ONLY == "theta" else "_h")
    fig.savefig(FIG / f"fig_multidim_trajectories{sfx}.png")
    tab = pd.DataFrame(rows); tab.to_csv(DESC / f"paper_multidim{sfx}.csv", index=False)
    print(tab[["fit", "class", "share", "rho_phys", "rho_ment", "signal_phys", "signal_ment", "gomp_slope", "hazard_55", "hazard_85"]].round(3).to_string(index=False))
    print(f"wrote fig_multidim_trajectories{sfx}.png to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
