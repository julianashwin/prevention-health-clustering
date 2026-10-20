"""Predicting ahead: the outcomes at t+1 and t+2 from what is known at t, on held-out fits.

The whole pipeline is out of sample in time. Every Bayesian fit used here was
refitted with each person's last two observed rows held out (people with at
least five rows on the health contract): the single-channel K = 3 fits on the
paper's measure under independent residuals (iid) and under AR(1) plus a
one-period "spike" (artifacts/health-next/health-{v}-{base,ssm}-ho), and the
multidimensional fit, physical + mental GRM + Gompertz-Makeham mortality,
under AR(1) plus "spike" (artifacts/multidim-health/{v}-ssm-mort-ho); and partial
K-means on the same fitted windows (22_paper_kmeans.py --last-held, variants
{v}_last2held). Class
posteriors and the models' own one-step forecasts are computed from the
fitted window alone, at the posterior mean, so nothing at t+1 informs them.

One prediction per person and horizon: t is the last fitted row and the
outcomes are at the two held rows, t+1 and t+2. Outcomes: the measure and the
cost proxy in pounds (the flat-cost index, capped at its 99th percentile,
waves 7-15 only). Death cannot occur before t+2, since everyone at t
has two more rows, and t+2 is by construction everyone's last interview; the
held-out death comparison (death after the last interview against leaving
the panel alive) is written to paper_prediction_death_heldout.csv for
41_paper_mortality_prediction.py, which draws it beside the full-sample
version. Predictors at t, each set carrying a quadratic in age:
  age                    age only
  + observables          education (4 groups), income rank, sex
  + observables, region, ethnicity   and the region (12) and ethnic group (5)
  + iid class            posterior class probabilities, independent residuals
  + AR(1)+"spike" class  posterior class probabilities, AR(1) plus "spike"
  + multidim class       posterior class probabilities, multidimensional fit
  + K-means type         the partial K-means type of the fitted window (dummies)
  + v(t)                 the measure at t
  + v(t), v(t-1)         and its lag (people with a lag observed)
  + AR(1)+"spike" forecast   the model's own forecast of v at the outcome's
                         age: the class-weighted path plus the filtered
                         state carried forward from t
  + multidim forecast    the same from the multidimensional fit
  + K-means forecast     the K-means type's mean of the measure at the outcome's
                         age (the type's centroid profile; no person-level state)
  + AR(1)+"spike" class, v(t)
  + multidim class, v(t)
  + all                  AR(1)+"spike" class, v(t), observables
People are split 70/30 on a fixed hash; models fitted on the 70, scored on
the 30; R-squared (AUC for held-out death). Both measures (h
and theta) are run on the same people, the intersection of the held-out
samples. A multidimensional fit that has not finished leaves its rows blank.

The paper's figures are drawn from these csvs by 56_paper_prediction_summary.py.

Outputs: artifacts/descriptives/paper_prediction.csv,
         artifacts/descriptives/paper_prediction_death_heldout.csv,
         artifacts/descriptives/paper_prediction_by_age.csv (R-squared and its gain over age
         alone within age bands at t, for a few sets, on the people with a lag observed)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import CONTRACT, DESC, load_measure, region_ethnicity  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR  # noqa: E402

K = 3
VARIANTS = ["h", "theta"]
VL = {"h": "h", "theta": r"\theta"}
EDUC = ["GCSE", "A-level", "degree or higher"]         # other/none is the reference
HO = ARTIFACTS_DIR / "health-next"
MD = ARTIFACTS_DIR / "multidim-health"
MD_CONTRACT = PROCESSED_DATA_DIR / "contracts" / "multidim_health_20_89_minobs3_v1"
MIN_OBS = 5
COST_CAP = 0.99


def logistic_fit(X, y, iters=60, ridge=1e-4):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.maximum(p * (1 - p), 1e-9)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * b)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def kalman_window(Y, X, coef, sigma, sigma_meas, rho, gap, fs, fe):
    """Per-person, per-class log-likelihood of the fitted window under AR(1) plus
    "spike", and the filtered state at the window's last row (for the forecast)."""
    n_person = len(fs)
    ll = np.zeros((n_person, K)); a_end = np.zeros((n_person, K))
    mu_all = X @ coef.T
    v_stat = sigma ** 2 / (1 - rho ** 2); var_meas = sigma_meas ** 2
    for i in range(n_person):
        a = np.zeros(K); Pv = v_stat.copy(); total = np.zeros(K)
        for j, n in enumerate(range(fs[i] - 1, fe[i])):
            if j > 0:
                rg = rho ** gap[n]
                a = rg * a
                Pv = rg ** 2 * Pv + v_stat * np.maximum(1 - rho ** (2 * gap[n]), 1e-9)
            F = Pv + var_meas; v = Y[n] - mu_all[n] - a
            total += -0.5 * (np.log(2 * np.pi * F) + v ** 2 / F)
            Kg = Pv / F; a = a + Kg * v; Pv = Pv - Kg * Pv
        ll[i] = total; a_end[i] = a
    return ll, a_end


def iid_window(Y, X, coef, sigma, fs, fe):
    mu = X @ coef.T
    lp = -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((Y[:, None] - mu) / sigma) ** 2
    idx = np.concatenate([np.arange(a - 1, b) for a, b in zip(fs, fe)])       # fitted rows only
    person = np.repeat(np.arange(len(fs)), fe - fs + 1)
    ll = np.zeros((len(fs), K)); np.add.at(ll, person, lp[idx])
    return ll, np.zeros((len(fs), K))


def gompertz_window(d, log_b, slope, makeham):
    """Survival likelihood over the FITTED window only. The fit itself used the full
    window (mortality is never held out), but a decedent's event sits on their last
    row, which is a held row here, so the class posterior for prediction must not see
    it: each fitted row contributes survival of its interval, no event."""
    age = np.asarray(d["mort_age"], float); gap = np.asarray(d["mort_gap"], float)
    obs = np.asarray(d["mort_obs"]) == 1
    fs, fe = np.asarray(d["fit_start"]), np.asarray(d["fit_end"])
    idx = np.concatenate([np.arange(a - 1, b) for a, b in zip(fs, fe)])
    person = np.repeat(np.arange(len(fs)), fe - fs + 1)
    out = np.zeros((len(fs), K))
    for k in range(K):
        H = makeham * gap + np.exp(log_b[k] - np.log(slope[k]) + slope[k] * age) * np.expm1(slope[k] * gap)
        out[:, k] = np.bincount(person, weights=(-H * obs)[idx], minlength=len(fs))
    return out


def posterior_and_forecast(d, Ys, coefs, sigmas, smeas, rhos, theta, mort=None):
    """Class posteriors from the fitted window and the filtered state of channel 0 at
    the window's last row (zero under independent residuals). ``rhos`` is K x C."""
    X = np.asarray(d["X"]); fs, fe = np.asarray(d["fit_start"]), np.asarray(d["fit_end"])
    gap = np.asarray(d["age_gap"], float) if d["ar_mode"] else None
    per = np.zeros((len(fs), K)); a0 = None
    for c, Y in enumerate(Ys):
        if d["ar_mode"] == 2:
            ll, a_end = kalman_window(Y, X, coefs[c], sigmas[c], smeas[c], rhos[:, c], gap, fs, fe)
        else:
            ll, a_end = iid_window(Y, X, coefs[c], sigmas[c], fs, fe)
        per += ll
        if c == 0:
            a0 = a_end
    if mort is not None:
        per += gompertz_window(d, *mort)
    un = np.log(theta)[None, :] + per
    w = np.exp(un - un.max(axis=1, keepdims=True)); w /= w.sum(axis=1, keepdims=True)
    return w, a0


def forecast(w, a0, coef0, rho0, age_target, age_fe):
    """The model's forecast of channel 0 at ``age_target`` from the fitted window ending at
    ``age_fe``: the class-weighted path plus the filtered state carried forward."""
    a = (np.asarray(age_target, float) - 55) / 10
    mu = coef0[:, 0][None, :] + coef0[:, 1][None, :] * a[:, None] + coef0[:, 2][None, :] * a[:, None] ** 2
    carry = (rho0[None, :] ** (np.asarray(age_target, float) - np.asarray(age_fe, float))[:, None]) * a0
    return (w * (mu + carry)).sum(axis=1)


def window_ages(d, sub):
    fe, hs, he = (np.asarray(d[k]) for k in ("fit_end", "hold_start", "hold_end"))
    age = sub["age"].to_numpy(float)
    return age[fe - 1], age[hs - 1], age[he - 1]


def single_channel(v: str, tag: str, sub: pd.DataFrame) -> pd.DataFrame:
    fit_dir = HO / f"health-{v}-{tag}-ho"
    d = json.load(open(fit_dir / "stan_data.json")); p = json.load(open(fit_dir / "run_summary.json"))["params"]
    assert len(sub) == d["N_obs"], f"{fit_dir.name}: rows {len(sub):,} vs payload {d['N_obs']:,}"
    m, s = sub[v].mean(), sub[v].std(ddof=1)
    Y = np.asarray(d["y"][0]); assert np.abs((sub[v].to_numpy() - m) / s - Y).max() < 1e-6, "rows do not match the payload"
    theta = np.array([p[f"theta[{k}]"]["mean"] for k in range(1, K + 1)])
    coef = np.array([[p[f"coef[1,{k},{j}]"]["mean"] for j in (1, 2, 3)] for k in range(1, K + 1)])
    sigma = p["sigma[1,1]"]["mean"]
    rho = np.array([[p[f"rho[{k}]"]["mean"]] for k in range(1, K + 1)]) if d["ar_mode"] == 2 else np.zeros((K, 1))
    sm = [p["sigma_meas[1]"]["mean"]] if d["ar_mode"] == 2 else [0.0]
    w, a0 = posterior_and_forecast(d, [Y], [coef], [sigma], sm, rho, theta)
    pids = sub.groupby("pidp", sort=False)["pidp"].first().to_numpy()
    post = pd.DataFrame({"pidp": pids, f"{tag}_c1": w[:, 0], f"{tag}_c2": w[:, 1]})
    return post, {"tag": tag, "w": w, "a0": a0, "coef": coef, "rho": rho[:, 0], "m": m, "s": s, "pidp": pids,
                  "age_fe": window_ages(d, sub)[0]}


def multidim(v: str, fit: str = "ssm-mort-ho", tag: str = "md"):
    """The held-out multidimensional fit {v}-{fit}: class posteriors from the fitted window
    and the forecast spec. With birth-decade shifts in the fit (the -cohort-ho twin) the
    scores are netted of each row's shift before the likelihood, and the person's shift
    on the physical channel is carried as ``offset`` so the forecast adds it back."""
    fit_dir = MD / f"{v}-{fit}"
    if not (fit_dir / "run_summary.json").exists():
        return None
    d = json.load(open(fit_dir / "stan_data.json")); summ = json.load(open(fit_dir / "run_summary.json"))
    p, mom, ch = summ["params"], summ["channel_moments"], summ["channels"]
    long = pd.read_csv(MD_CONTRACT / "long.csv"); n = long.groupby("pidp")["age"].transform("size")
    sub = long[n >= MIN_OBS].sort_values(["pidp", "age"]).reset_index(drop=True)
    assert len(sub) == d["N_obs"]
    Ys = [np.asarray(d["y"][c]) for c in range(2)]
    assert np.abs((sub[ch[0]].to_numpy() - mom[ch[0]]["mean"]) / mom[ch[0]]["sd"] - Ys[0]).max() < 1e-6
    theta = np.array([p[f"theta[{k}]"]["mean"] for k in range(1, K + 1)])
    coefs = [np.array([[p[f"coef[{c},{k},{j}]"]["mean"] for j in (1, 2, 3)] for k in range(1, K + 1)]) for c in (1, 2)]
    sigmas = [p[f"sigma[1,{c}]"]["mean"] for c in (1, 2)]
    rhos = np.array([[p[f"rho[{k},{c}]"]["mean"] for c in (1, 2)] for k in range(1, K + 1)])
    sm = [p[f"sigma_meas[{c}]"]["mean"] for c in (1, 2)]
    mort = (np.array([p[f"log_b[{k}]"]["mean"] for k in range(1, K + 1)]),
            np.array([p[f"gomp_slope[{k}]"]["mean"] for k in range(1, K + 1)]), p["makeham[1]"]["mean"])
    fs = np.asarray(d["fit_start"]); offset = np.zeros(len(fs))
    n_coh = d.get("N_cohort", 1)
    if n_coh > 1:
        ce = np.zeros((2, n_coh))
        for c in range(2):
            ce[c, 1:] = [p[f"cohort_effect[{c + 1},{j}]"]["mean"] for j in range(2, n_coh + 1)]
        cid = np.asarray(d["cohort_id"]) - 1
        Ys = [Ys[c] - ce[c, cid] for c in range(2)]
        offset = ce[0, cid[fs - 1]] * mom[ch[0]]["sd"]           # the person's shift, in the measure's units
    w, a0 = posterior_and_forecast(d, Ys, coefs, sigmas, sm, rhos, theta, mort=mort)
    pids = sub.groupby("pidp", sort=False)["pidp"].first().to_numpy()
    post = pd.DataFrame({"pidp": pids, f"{tag}_c1": w[:, 0], f"{tag}_c2": w[:, 1]})
    return post, {"tag": tag, "w": w, "a0": a0, "coef": coefs[0], "rho": rhos[:, 0], "m": mom[ch[0]]["mean"],
                  "s": mom[ch[0]]["sd"], "pidp": pids, "age_fe": window_ages(d, sub)[0], "offset": offset}


def kmeans_last_held(v: str):
    """Partial K-means types on the fitted windows (each person's last two rows held out) and
    the types' mean profiles by age, from 22_paper_kmeans.py --last-held; None until that has run."""
    var = f"{v}_last2held"
    lab = pd.read_parquet(DESC / "paper_kmeans_labels.parquet")
    lab = lab[lab["variant"] == var]
    if lab.empty:
        return None
    traj = pd.read_csv(DESC / "paper_kmeans_trajectories.csv")
    traj = traj[(traj["variant"] == var) & (traj["lo"] == 20) & (traj["hi"] == 90)]
    prof = traj.pivot(index="age", columns="cluster", values="mean").reindex(range(20, 90)).interpolate(limit_direction="both")
    post = lab[["pidp", "cluster"]].rename(columns={"cluster": "km_type"})
    for c in range(K - 1):
        post[f"km_{c + 1}"] = (post["km_type"] == c).astype(float)
    return post, prof


def kmeans_forecast(at: pd.DataFrame, prof: pd.DataFrame, target_age: np.ndarray) -> np.ndarray:
    """The person's K-means type's mean of the measure at ``target_age`` (integer age, 20-89)."""
    a = np.clip(np.floor(np.asarray(target_age, float)), 20, 89)
    out = np.full(len(at), np.nan); ok = at["km_type"].notna().to_numpy() & ~np.isnan(a)
    vals = prof.to_numpy()
    out[ok] = vals[(a[ok] - 20).astype(int), at["km_type"].to_numpy()[ok].astype(int)]
    return out


def add_forecasts(at: pd.DataFrame, specs: list[dict], target_age: np.ndarray) -> pd.DataFrame:
    """The models' forecasts of the measure at ``target_age`` for the people in ``at``."""
    for sp in specs:
        pos = pd.Series(np.arange(len(sp["pidp"])), index=sp["pidp"]).reindex(at["pidp"]).to_numpy()
        ok = ~np.isnan(pos); idx = pos[ok].astype(int)
        fc = np.full(len(at), np.nan)
        fc[ok] = sp["m"] + sp["s"] * forecast(sp["w"][idx], sp["a0"][idx], sp["coef"], sp["rho"], target_age[ok], sp["age_fe"][idx])
        if "offset" in sp:
            fc[ok] += sp["offset"][idx]
        at[f"{sp['tag']}_fc"] = fc
    return at


def outcomes(v: str, at: pd.DataFrame, k: int) -> pd.DataFrame:
    """Outcomes k waves after the row at t (one row per person), plus the observables."""
    meas = load_measure([v])
    nxt = meas[["pidp", "wave", v]].rename(columns={v: "y_next"}); nxt["wave"] = nxt["wave"] - k
    lag = meas[["pidp", "wave", v]].rename(columns={v: "y_lag"}); lag["wave"] = lag["wave"] + 1
    d = at.merge(nxt, on=["pidp", "wave"], how="left").merge(lag, on=["pidp", "wave"], how="left")
    cost = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "cost_index.parquet", columns=["pidp", "wave", "flat_cost_total"]).dropna()
    cost["cost_next"] = cost["flat_cost_total"].clip(upper=cost["flat_cost_total"].quantile(COST_CAP))
    cost = cost.groupby(["pidp", "wave"], as_index=False)["cost_next"].mean(); cost["wave"] = cost["wave"] - k
    d = d.merge(cost, on=["pidp", "wave"], how="left")
    panel = pd.read_csv(PROCESSED_DATA_DIR / "ukhls_indresp_processed.csv", usecols=["pidp", "wave", "dcsedw_dv", "sex"])
    death_wave = panel.groupby("pidp")["dcsedw_dv"].max() - 18
    d = d.merge(death_wave.rename("death_wave"), left_on="pidp", right_index=True, how="left")
    # death before the wave after the last held row, among those at risk there
    d["died_next"] = np.where(d["wave_he"] >= 15, np.nan, (d["death_wave"] == d["wave_he"] + 1).astype(float))
    d.loc[d["death_wave"] <= d["wave_he"], "died_next"] = np.nan
    per = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "observables_person.parquet").set_index("pidp")
    d = d.join(per[["educ_group", "income_rank_mean"]], on="pidp")
    d["female"] = d["pidp"].map(panel.groupby("pidp")["sex"].first()).eq(2).astype(float)
    for g in EDUC:
        d[f"educ_{g}"] = (d["educ_group"] == g).astype(float)
    d = d.join(region_ethnicity(d["pidp"].unique()), on="pidp")
    d["a"] = (d["age"] - 55) / 10
    return d


def score_sets(s: pd.DataFrame, oc: str, kind: str, SETS: dict) -> list[dict]:
    """Fit every predictor set on the 70% and score it on the 30%."""
    test = ((pd.util.hash_pandas_object(s["pidp"], index=False) % 10) >= 7).to_numpy()
    y = s[oc].to_numpy(float)
    out = []
    for name, cols in SETS.items():
        s2, t2, y2 = s, test, y
        need = [c for c in cols if c in ("y_lag", "md_c1", "md_c2", "md_fc", "mdc_c1", "mdc_c2", "mdc_fc", "km_1", "km_2", "km_fc", *[c for c in cols if c.startswith(("region_", "ethnic_"))])]
        if need:
            keep = s[need].notna().all(axis=1).to_numpy()
            s2, t2, y2 = s[keep], test[keep], y[keep]
        score = np.nan
        if len(s2) > 100 and (kind != "auc" or 0 < y2[t2].sum() < t2.sum()):
            X = np.column_stack([np.ones(len(s2)), s2["a"], s2["a"] ** 2] + [s2[c].to_numpy(float) for c in cols])
            tr, te = ~t2, t2
            if kind == "auc":
                b = logistic_fit(X[tr], y2[tr]); p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
                score = auc(y2[te], p[te])
            else:
                b = np.linalg.lstsq(X[tr], y2[tr], rcond=None)[0]; p = X @ b
                score = float(1 - ((y2[te] - p[te]) ** 2).sum() / ((y2[te] - y2[te].mean()) ** 2).sum())
        out.append({"predictors": name, "n": len(s2), "n_test": int(t2.sum()),
                    "events": float(y2.mean()) if kind == "auc" else np.nan, "score": score})
    return out


AGE_BANDS = [(20, 34), (35, 49), (50, 64), (65, 90)]
BY_AGE_SETS = ["age", "+ observables", "+ {v}(t)", "+ {v}(t), {v}(t-1)", '+ AR(1)+"spike" forecast']


def scores_by_age(s: pd.DataFrame, oc: str, SETS: dict, names: list[str]) -> list[dict]:
    """R-squared within age bands at t, on the test 30%, of each named set fitted on the
    whole 70% (the same fits as score_sets), and its gain over age alone in the band."""
    test = ((pd.util.hash_pandas_object(s["pidp"], index=False) % 10) >= 7).to_numpy()
    y = s[oc].to_numpy(float); out = []
    preds = {}
    for name in names:
        cols = SETS[name]
        X = np.column_stack([np.ones(len(s)), s["a"], s["a"] ** 2] + [s[c].to_numpy(float) for c in cols])
        b = np.linalg.lstsq(X[~test], y[~test], rcond=None)[0]; preds[name] = X @ b
    for lo, hi in AGE_BANDS:
        sel = test & s["age"].between(lo, hi).to_numpy()
        yb = y[sel]; ss = ((yb - yb.mean()) ** 2).sum()
        base = 1 - ((yb - preds["age"][sel]) ** 2).sum() / ss
        for name in names:
            r2 = 1 - ((yb - preds[name][sel]) ** 2).sum() / ss
            out.append({"predictors": name, "band": f"{lo}-{hi}", "n_test": int(sel.sum()), "score": float(r2), "gain": float(r2 - base)})
    return out


def run_variant(v: str, md_people: np.ndarray) -> tuple[pd.DataFrame, dict]:
    long = pd.read_csv(CONTRACT / "long.csv"); n = long.groupby("pidp")["age"].transform("size")
    sub = long[n >= MIN_OBS].sort_values(["pidp", "age"]).reset_index(drop=True)
    parts = [single_channel(v, "base", sub), single_channel(v, "ssm", sub)]
    md = multidim(v)
    have_md = md is not None
    mdc = multidim(v, "ssm-mort-cohort-ho", "mdc")          # birth-decade shifts; theta only so far
    have_mdc = mdc is not None
    specs = [sp for _, sp in parts] + ([md[1]] if have_md else []) + ([mdc[1]] if have_mdc else [])
    d = json.load(open(HO / f"health-{v}-ssm-ho" / "stan_data.json"))
    fe, hs, he = (np.asarray(d[k]) for k in ("fit_end", "hold_start", "hold_end"))
    at = sub.iloc[fe - 1][["pidp", "wave", "age", v]].reset_index(drop=True)          # the row at t
    at["wave_he"] = sub["wave"].to_numpy()[he - 1]; at["age_he"] = sub["age"].to_numpy(float)[he - 1]
    at["age_hs"] = sub["age"].to_numpy(float)[hs - 1]
    at = at[at["pidp"].isin(md_people)]
    for post, _ in parts:
        at = at.merge(post, on="pidp")
    at = at.merge(md[0], on="pidp", how="left") if have_md else at.assign(md_c1=np.nan, md_c2=np.nan)
    at = at.merge(mdc[0], on="pidp", how="left") if have_mdc else at.assign(mdc_c1=np.nan, mdc_c2=np.nan)
    km = kmeans_last_held(v)
    at = at.merge(km[0], on="pidp", how="left") if km is not None else at.assign(km_type=np.nan, km_1=np.nan, km_2=np.nan)
    frames = {}
    for k, target in ((1, at["age_hs"].to_numpy()), (2, at["age_he"].to_numpy()), (3, at["age_he"].to_numpy() + 1.0)):
        f = add_forecasts(at.copy(), specs, target)
        if not have_md:
            f["md_fc"] = np.nan
        if not have_mdc:
            f["mdc_fc"] = np.nan
        f["km_fc"] = kmeans_forecast(f, km[1], target) if km is not None else np.nan
        frames[k] = outcomes(v, f, k)
    obs = ["income_rank_mean", "female"] + [f"educ_{g}" for g in EDUC]
    regeth = [c for c in frames[1].columns if c.startswith(("region_", "ethnic_"))]
    SETS = {
        "age": [],
        "+ observables": obs,
        "+ observables, region, ethnicity": obs + regeth,
        "+ iid class": ["base_c1", "base_c2"],
        '+ AR(1)+"spike" class': ["ssm_c1", "ssm_c2"],
        "+ multidim class": ["md_c1", "md_c2"],
        "+ multidim cohort class": ["mdc_c1", "mdc_c2"],
        "+ K-means type": ["km_1", "km_2"],
        f"+ {v}(t)": [v],
        f"+ {v}(t), {v}(t-1)": [v, "y_lag"],
        '+ AR(1)+"spike" forecast': ["ssm_fc"],
        "+ multidim forecast": ["md_fc"],
        "+ multidim cohort forecast": ["mdc_fc"],
        "+ K-means forecast": ["km_fc"],
        f'+ AR(1)+"spike" class, {v}(t)': ["ssm_c1", "ssm_c2", v],
        f"+ multidim class, {v}(t)": ["md_c1", "md_c2", v],
        "+ all": ["ssm_c1", "ssm_c2", v] + obs,            # these three are drawn in 41's mortality figure
    }
    # (outcome column, label, metric, horizons)
    OUT = [("y_next", f"health, ${VL[v]}$", "r2", (1, 2)), ("cost_next", "cost proxy (£)", "r2", (1, 2))]
    base_cols = ["a", v] + obs + ["base_c1", "ssm_c1", "ssm_fc"]
    rows, info, by_age = [], {}, []
    for oc, olab, kind, horizons in OUT:
        for k in horizons:
            s = frames[k].dropna(subset=[oc] + base_cols)
            y = s[oc].to_numpy(float)
            info[(oc, k)] = {"n": len(s), "rate": float(y.mean()) if kind == "auc" else np.nan}
            for r in score_sets(s, oc, kind, SETS):
                rows.append({"variant": v, "outcome": oc, "outcome_label": olab, "horizon": k, "metric": kind, **r})
            if kind == "r2":
                names = [n.format(v=v) for n in BY_AGE_SETS]
                by_age += [{"variant": v, "outcome": oc, "horizon": k, **r} for r in scores_by_age(s.dropna(subset=["y_lag"]), oc, SETS, names)]
            print(f"{v} {oc} t+{k}: {len(s):,} people" + (f", event rate {y.mean():.3%}" if kind == "auc" else ""))
    # held-out death, for 41_paper_mortality_prediction.py
    s = frames[3].dropna(subset=["died_next"] + base_cols)
    death = pd.DataFrame([{"variant": v, "outcome": "died_next", **r} for r in score_sets(s, "died_next", "auc", SETS)])
    print(f"{v} died_next after the last interview: {len(s):,} people, event rate {s['died_next'].mean():.3%}")
    t = pd.DataFrame(rows)
    t["gain"] = t["score"] - t.groupby(["outcome", "horizon"])["score"].transform("first")
    return t, {"sets": list(SETS), "out": OUT, "have_md": have_md, "info": info, "death": death, "by_age": pd.DataFrame(by_age)}


def main() -> int:
    md_long = pd.read_csv(MD_CONTRACT / "long.csv", usecols=["pidp", "age"])
    md_people = md_long.groupby("pidp").size().pipe(lambda s: s[s >= MIN_OBS]).index.to_numpy()
    # --stage v runs one variant and caches it; --assemble writes the outputs from the cached
    # variants (for machines where one run of both is too long); neither flag runs both in turn
    import pickle
    cache = DESC / ".cache_33"
    if "--stage" in sys.argv:
        v = sys.argv[sys.argv.index("--stage") + 1]
        cache.mkdir(exist_ok=True)
        pickle.dump(run_variant(v, md_people), open(cache / f"{v}.pkl", "wb"))
        print(f"staged {v}"); return 0
    tabs, metas = {}, {}
    for v in VARIANTS:
        if "--assemble" in sys.argv:
            tabs[v], metas[v] = pickle.load(open(cache / f"{v}.pkl", "rb"))
        else:
            tabs[v], metas[v] = run_variant(v, md_people)
    t = pd.concat(tabs.values()); t.to_csv(DESC / "paper_prediction.csv", index=False)
    death = pd.concat([metas[v]["death"] for v in VARIANTS])
    death["gain"] = death["score"] - death.groupby("variant")["score"].transform("first")
    death.to_csv(DESC / "paper_prediction_death_heldout.csv", index=False)
    pd.concat([metas[v]["by_age"] for v in VARIANTS]).to_csv(DESC / "paper_prediction_by_age.csv", index=False)
    print("wrote paper_prediction.csv, paper_prediction_by_age.csv, paper_prediction_death_heldout.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
