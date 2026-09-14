"""Can the K=4 state-space model recover a four-class truth?

Section 10 of the health measures note fits four classes to P-FULL and P-FUNC
and reads the fourth -- 7.1% of people, highest level, lowest persistence and
signal share -- as a probable measurement-ceiling artefact. Two things bear
on that reading which the real-data fits cannot show:

  1. IDENTIFICATION. If the data really came from these four classes, would
     the model recover them, including a small, weakly persistent top class?
     If it cannot, the fitted class is suspect for estimation reasons before
     any substantive reading. This is also the check that caught a mode split
     in the multidimensional model, so per-chain agreement is reported, not
     just R-hat.

  2. CALIBRATION OF THE SEPARATION ARGUMENT. Section 10 counts it against the
     fourth class that classification certainty falls at K=4 (mean max
     posterior .668). What certainty does a TRUE four-class population of this
     shape produce? Computing it at the true parameters separates the overlap
     intrinsic to the configuration from anything the estimation adds. If a
     true K=4 world also sits near .67, low certainty is not evidence against
     a fourth class.

Truth is the fitted P-FULL K=4 posterior mean, on its standardised scale,
with class-specific persistence. No ceiling is imposed, so this tests the
model, not the ceiling reading. Only the state-space model is fitted; the
AR(1) contrast of the K=3 recovery is not needed for either question.

    PYTHONPATH=src .venv/bin/python \
        clustering/validation/synthetic_recovery_ssm_k4.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from prevention_health_clustering.config import ARTIFACTS_DIR
from prevention_health_clustering.models.registry import get_model
from prevention_health_clustering.runner.fit import build_payload, fit_model

TRUTH_FIT = ARTIFACTS_DIR / "k4" / "physfull-ssm-k4" / "run_summary.json"
CHANNEL = "theta_phys_full"
AGE_CENTER, AGE_SCALE = 55, 10.0


def load_truth(path: Path, K: int = 4) -> dict:
    p = json.loads(path.read_text())["params"]
    g = lambda n: p[n]["mean"]
    theta = np.array([g(f"theta[{k}]") for k in range(1, K + 1)])
    return {
        "theta": theta / theta.sum(),
        "coef": np.array([[g(f"coef[1,{k},{q}]") for q in (1, 2, 3)]
                          for k in range(1, K + 1)]),
        "rho": np.array([g(f"rho[{k}]") for k in range(1, K + 1)]),
        "sigma": g("sigma[1,1]"),
        "sigma_meas": g("sigma_meas[1]"),
    }


def simulate(truth: dict, n_persons: int, seed: int, min_age: int = 20,
             max_age: int = 89, mean_obs: int = 9):
    rng = np.random.default_rng(seed)
    K = len(truth["theta"])
    classes = rng.choice(K, size=n_persons, p=truth["theta"])
    rows = []
    for person, k in enumerate(classes):
        n_obs = int(np.clip(rng.poisson(mean_obs - 5) + 5, 5, 15))
        start = rng.integers(min_age, max_age - n_obs + 2)
        ages = np.arange(start, start + n_obs)
        a = (ages - AGE_CENTER) / AGE_SCALE
        c = truth["coef"][k]
        mu = c[0] + c[1] * a + c[2] * a ** 2
        rho = truth["rho"][k]
        u = np.empty(n_obs)
        u[0] = rng.normal(0.0, truth["sigma"] / np.sqrt(1 - rho ** 2))
        for t in range(1, n_obs):
            u[t] = rho * u[t - 1] + rng.normal(0.0, truth["sigma"])
        y = mu + u + rng.normal(0.0, truth["sigma_meas"], size=n_obs)
        for age, val in zip(ages, y):
            rows.append((person, int(age), int(age) - AGE_CENTER, 1, 1970, val))
    frame = pd.DataFrame(rows, columns=["pidp", "age", "age_c", "wave",
                                        "birthy", CHANNEL])
    return frame, pd.Series(classes, name="true_class")


def to_payload_scale(truth: dict, m: float, sd: float) -> dict:
    """The runner standardises y, so express truth on that scale exactly."""
    t = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v
         for k, v in truth.items()}
    t["coef"] = truth["coef"].copy()
    t["coef"][:, 0] = (truth["coef"][:, 0] - m) / sd
    t["coef"][:, 1:] = truth["coef"][:, 1:] / sd
    t["sigma"] = truth["sigma"] / sd
    t["sigma_meas"] = truth["sigma_meas"] / sd
    return t


def kalman_person_lp(Y, X, coef, sigma, sigma_meas, rho, gap, fs, fe):
    """(persons, K) log density under an AR(1) state plus i.i.d. noise."""
    K, n_person = coef.shape[0], len(fs)
    lens = fe - fs + 1
    T = int(lens.max())
    idx = np.full((n_person, T), -1, dtype=np.int64)
    for i in range(n_person):
        idx[i, :lens[i]] = np.arange(fs[i] - 1, fe[i])
    mask = idx >= 0
    safe = np.where(mask, idx, 0)
    mu = (X @ coef.T)[safe]
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


def classification(pl, prm: dict, true_by_pidp: pd.Series):
    d = pl.data
    lp = kalman_person_lp(np.asarray(d["y"][0]), np.asarray(d["X"]),
                          prm["coef"], prm["sigma"], prm["sigma_meas"],
                          prm["rho"], np.asarray(d["age_gap"]),
                          np.asarray(d["fit_start"]), np.asarray(d["fit_end"]))
    joint = np.log(prm["theta"])[None, :] + lp
    w = np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
    K = w.shape[1]
    ent = -(w * np.log(np.clip(w, 1e-300, None))).sum()
    true = true_by_pidp.reindex(np.asarray(pl.person_ids)).to_numpy()
    modal = w.argmax(axis=1)
    conf = pd.crosstab(pd.Series(true + 1, name="true"),
                       pd.Series(modal + 1, name="assigned"))
    conf = conf.reindex(index=range(1, K + 1), columns=range(1, K + 1),
                        fill_value=0)
    return {"mean_max_posterior": float(w.max(axis=1).mean()),
            "rel_entropy": float(1 - ent / (len(w) * np.log(K))),
            "accuracy": float((modal == true).mean()),
            "confusion_row_pct": (conf.div(conf.sum(axis=1), axis=0) * 100)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persons", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--sampling", type=int, default=600)
    ap.add_argument("--output", type=Path,
                    default=ARTIFACTS_DIR / "recovery-ssm-k4")
    ap.add_argument("--from-csv", action="store_true",
                    help="Skip sampling and analyse the chains already in "
                         "OUTPUT/chains. The simulated panel is rebuilt "
                         "exactly from the seed.")
    a = ap.parse_args(argv)
    a.output.mkdir(parents=True, exist_ok=True)
    K = 4

    truth = load_truth(TRUTH_FIT, K)
    frame, true_class = simulate(truth, a.persons, a.seed)
    true_by_pidp = pd.Series(true_class.to_numpy(), index=range(a.persons))
    realised = np.bincount(true_class, minlength=K) / a.persons
    print(f"simulated {a.persons:,} people, {len(frame):,} rows; realised "
          f"class shares {np.round(realised, 3).tolist()}")

    v = frame[CHANNEL].to_numpy()
    m, sd = float(np.mean(v)), float(np.std(v, ddof=1))
    tp = to_payload_scale(truth, m, sd)
    print(f"standardisation: mean {m:.4f}, sd {sd:.4f}")

    spec = get_model("physgrm-full-ssm-k4")
    assert spec.n_classes == K and spec.ar_mode == 2
    pl = build_payload(spec, frame)

    # Certainty a TRUE four-class world of this shape produces, before any
    # estimation: the calibration for section 10's separation argument.
    at_truth = classification(pl, tp, true_by_pidp)
    print(f"\nat the TRUE parameters: mean max posterior "
          f"{at_truth['mean_max_posterior']:.3f}, relative entropy "
          f"{at_truth['rel_entropy']:.3f}, modal accuracy "
          f"{at_truth['accuracy']:.3f}")

    if a.from_csv:
        import cmdstanpy
        files = sorted(str(f) for f in (a.output / "chains").glob("*.csv"))
        print(f"re-analysing {len(files)} existing chains, no sampling")
        fit = cmdstanpy.from_csv(files)
    else:
        fit = fit_model(spec, pl, output_dir=a.output, threads_per_chain=1,
                        chains=a.chains, parallel_chains=a.chains,
                        iter_warmup=a.warmup, iter_sampling=a.sampling,
                        seed=a.seed)
    s = fit.summary()
    dr = fit.draws(concat_chains=False)          # (draws, chains, params)
    cols = fit.column_names
    col = cols.index          # cmdstanpy reports bracket names, e.g. theta[1]

    names, rows = [], []
    for k in range(1, K + 1):
        names += [(f"theta[{k}]", tp["theta"][k - 1]),
                  (f"coef[1,{k},1]", tp["coef"][k - 1, 0]),
                  (f"coef[1,{k},2]", tp["coef"][k - 1, 1]),
                  (f"coef[1,{k},3]", tp["coef"][k - 1, 2]),
                  (f"rho[{k}]", tp["rho"][k - 1])]
    names += [("sigma[1,1]", tp["sigma"]), ("sigma_meas[1]", tp["sigma_meas"])]
    for n, true in names:
        x = dr[:, :, col(n)]
        flat = x.reshape(-1)
        lo, hi = np.quantile(flat, [0.05, 0.95])
        per_chain = x.mean(axis=0)
        rows.append({"param": n, "truth": true, "mean": flat.mean(),
                     "sd": flat.std(), "q05": lo, "q95": hi,
                     "covered_90": bool(lo <= true <= hi),
                     "error": flat.mean() - true,
                     "chain_spread": float(per_chain.max() - per_chain.min()),
                     "rhat": float(s.loc[n, "R_hat"])})
    rec = pd.DataFrame(rows)
    rec.to_csv(a.output / "recovery_k4.csv", index=False)
    print("\n=== recovery (payload scale) ===")
    print(rec.round(4).to_string(index=False))

    prm = {"theta": np.array([rec.set_index("param").loc[f"theta[{k}]", "mean"]
                              for k in range(1, K + 1)]),
           "coef": np.array([[rec.set_index("param").loc[f"coef[1,{k},{q}]", "mean"]
                              for q in (1, 2, 3)] for k in range(1, K + 1)]),
           "rho": np.array([rec.set_index("param").loc[f"rho[{k}]", "mean"]
                            for k in range(1, K + 1)]),
           "sigma": rec.set_index("param").loc["sigma[1,1]", "mean"],
           "sigma_meas": rec.set_index("param").loc["sigma_meas[1]", "mean"]}
    prm["theta"] = prm["theta"] / prm["theta"].sum()
    at_fit = classification(pl, prm, true_by_pidp)

    div = int(fit.method_variables()["divergent__"].sum())
    core = [n for n, _ in names]
    max_rhat = float(s.loc[core, "R_hat"].max())
    rho_spread = float(rec[rec.param.str.startswith("rho")]["chain_spread"].max())
    alpha = prm["coef"][:, 0]
    covered = int(rec["covered_90"].sum())
    print(f"\nmax R-hat {max_rhat:.4f}   divergences {div}   largest "
          f"per-chain spread in rho {rho_spread:.4f}")
    print(f"truth inside the 90% interval for {covered} of {len(rec)} "
          "parameters (about 90% expected)")
    print(f"recovered class intercepts {np.round(alpha, 3).tolist()} against "
          f"true {np.round(tp['coef'][:, 0], 3).tolist()}")
    print(f"\nat the ESTIMATED parameters: mean max posterior "
          f"{at_fit['mean_max_posterior']:.3f}, relative entropy "
          f"{at_fit['rel_entropy']:.3f}, modal accuracy {at_fit['accuracy']:.3f}")
    print("\ntrue class (rows) -> assigned class (cols), % of true class, "
          "estimated parameters:")
    print(at_fit["confusion_row_pct"].round(1).to_string())

    summary = {
        "persons": a.persons, "rows": int(len(frame)), "seed": a.seed,
        "warmup": a.warmup, "sampling": a.sampling, "chains": a.chains,
        "standardisation": {"mean": m, "sd": sd},
        "max_rhat": max_rhat, "divergences": div,
        "max_rho_chain_spread": rho_spread,
        "covered_90": covered, "n_params": int(len(rec)),
        "truth_payload_scale": {"theta": tp["theta"].tolist(),
                                "alpha": tp["coef"][:, 0].tolist(),
                                "rho": tp["rho"].tolist(),
                                "sigma": tp["sigma"],
                                "sigma_meas": tp["sigma_meas"]},
        "estimate": {"theta": prm["theta"].tolist(), "alpha": alpha.tolist(),
                     "rho": prm["rho"].tolist(), "sigma": prm["sigma"],
                     "sigma_meas": prm["sigma_meas"]},
        "certainty_at_truth": {k: v for k, v in at_truth.items()
                               if k != "confusion_row_pct"},
        "certainty_at_estimate": {k: v for k, v in at_fit.items()
                                  if k != "confusion_row_pct"},
        "confusion_row_pct_estimate": at_fit["confusion_row_pct"].round(2)
        .to_dict(orient="index"),
        "confusion_row_pct_truth": at_truth["confusion_row_pct"].round(2)
        .to_dict(orient="index"),
    }
    (a.output / "recovery_k4_summary.json").write_text(
        json.dumps(summary, indent=1, default=float))
    ok = max_rhat < 1.05 and rho_spread < 0.02 and covered >= 0.75 * len(rec)
    print(f"\nRECOVERY {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
