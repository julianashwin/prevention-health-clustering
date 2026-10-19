"""What identifies a person's health type: a health reading, or the observables?

The types come from fits that never saw each person's first two rows (people with at
least five rows, so everyone keeps at least three fitted rows): partial K-means on the
remaining rows (22_paper_kmeans.py --first-held, variant theta_first2held), and the K=3
independent-residuals and AR(1) plus "spike" mixtures, the latter also with birth-decade
shifts, with the first two rows held out (artifacts/health-firstheld/*-first2ho), and the
multidimensional AR(1) plus "spike" fit with mortality and the first two rows held out
(artifacts/multidim-health/theta-ssm-mort-first2ho, on the multidim contract; it has no
decade shifts). The Bayesian type is the modal class of the posterior computed from the
fitted window alone (for the multidimensional fit: both channels plus survival through
the fitted window). This mirrors 33_paper_prediction.py, which holds out the last two
rows: here the window starts at t and the held rows are t-1 (the second row, closest to
the window) and t-2 (the first row).

The predictor sets, each on top of a quadratic in age at the first row: education
alone, income alone, education, income and sex together, those plus region and ethnic
group, one reading (theta at t-1, the closest held row, and separately at t-2, the
first row; each with its age interaction), both readings, and the first reading plus
the observables; for the multidimensional types also the mental score at the first row.
Multinomial logit on the modal type, people split 70/30 by a hash of pidp, fitted on the
70, scored on the 30: accuracy (against the majority-type rate), balanced accuracy (the
mean per-type recall, 1/3 by chance) and the macro one-vs-rest AUC. For the mixtures also
the model's own answer: the class posterior implied by the first observation alone
under the fitted parameters, and from both held rows (the Kalman likelihood of the two
rows from the stationary state). The by-age split of the accuracy of each single reading (by the age
it was taken) goes to a csv.

The first-row-only variant (fits with one row held out) is retired; its fits and outputs
are in artifacts/archive/firstho/.

--h runs the same on h (no multidimensional typology). Outputs:
fig_type_from_initial_first2{,_h}.png, tab_type_from_initial_first2{,_h}.tex,
paper_type_from_initial_first2{,_h}.csv and paper_type_from_initial_by_age_first2{,_h}.csv.

    PYTHONPATH=src .venv/bin/python descriptives/43_paper_type_from_initial.py [--h]
"""

from __future__ import annotations

import importlib
import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import CONTRACT, DESC, FIG, MENTAL_COL, MULTIDIM_CONTRACT, TAB, load_labels, region_ethnicity  # noqa: E402
from _style import CLUSTER, GREEN, INK2, ORANGE, PURPLE, apply_style  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR  # noqa: E402

M33 = importlib.import_module("33_paper_prediction")
K = 3
# --h: the exercise on h. The fits hold out the first TWO rows (people with at least five
# rows), the type is defined from the third row on, and the cohort-shift twin
# (health-{v}-ssm-cohort-first2ho) is a fourth typology. (--first2 is accepted and ignored.)
V = "h" if "--h" in sys.argv else "theta"
KHELD = 2
MIN_ROWS = KHELD + 3
HO = "first2ho"
SFX = "_first2" + ("_h" if V == "h" else "")
VL = {"theta": r"$\theta$", "h": "$h$"}[V]
FH = ARTIFACTS_DIR / "health-firstheld"
MD_DIR = ARTIFACTS_DIR / "multidim-health"
MD = MD_DIR / f"theta-ssm-mort-{HO}"
# the birth-decade-shift twin replaces it once it has finished
MD_COHORT = MD_DIR / f"theta-ssm-mort-cohort-{HO}"
if (MD_COHORT / "run_summary.json").exists():
    MD = MD_COHORT
MD_KHELD = KHELD
MD_LABEL = "multidimensional" + (", cohort shifts" if MD == MD_COHORT else "")
EDUC = ["GCSE", "A-level", "degree or higher"]
AGE_BANDS = [(20, 34), (35, 49), (50, 64), (65, 90)]
TYPOLOGIES = [("kmeans", "K-means"), ("base", "independent residuals"), ("ssm", 'AR(1) + "spike"'),
              ("ssm-cohort", 'AR(1) + "spike", cohort shifts'), ("md", MD_LABEL)]


def first_rows(contract=CONTRACT, extra: tuple[str, ...] = (), kheld: int = KHELD) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The retained people's rows (sorted) and their first row with the observables;
    people with at least kheld + 3 rows, the sample of the fit with kheld rows held out."""
    long = pd.read_csv(contract / "long.csv").sort_values(["pidp", "age"]).reset_index(drop=True)
    n = long.groupby("pidp")["age"].transform("size")
    sub = long[n >= kheld + 3].reset_index(drop=True)
    first = sub.groupby("pidp", as_index=False).first()[["pidp", "age", V, *extra]].rename(columns={"age": "age0", V: "y0"})
    if kheld == 2:   # the second held row, for the "+ v(0), v(1)" comparison
        second = sub.groupby("pidp", as_index=False).nth(1)[["pidp", "age", V]].rename(columns={"age": "age1", V: "y1"})
        first = first.merge(second, on="pidp")
    panel = pd.read_csv(PROCESSED_DATA_DIR / "ukhls_indresp_processed.csv", usecols=["pidp", "sex"])
    per = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "observables_person.parquet").set_index("pidp")
    first = first.join(per[["educ_group", "income_rank_mean"]], on="pidp")
    first["female"] = first["pidp"].map(panel.groupby("pidp")["sex"].first()).eq(2).astype(float)
    for g in EDUC:
        first[f"educ_{g}"] = (first["educ_group"] == g).astype(float)
    first = first.join(region_ethnicity(first["pidp"].unique()), on="pidp")
    first["a0"] = (first["age0"] - 55) / 10
    return sub, first


def bayes_types(tag: str, sub: pd.DataFrame, first: pd.DataFrame):
    """Modal class from the fitted window, and the class posterior implied by the first
    row alone, for the first-rows-held-out fit ``tag`` (base, ssm or ssm-cohort). With
    birth-decade shifts in the fit every score is netted of its decade shift first, as
    Stan computed the likelihood."""
    fit_dir = FH / f"health-{V}-{tag}-{HO}"
    if not (fit_dir / "run_summary.json").exists():
        return None
    d = json.load(open(fit_dir / "stan_data.json")); p = json.load(open(fit_dir / "run_summary.json"))["params"]
    assert len(sub) == d["N_obs"], f"{fit_dir.name}: rows {len(sub):,} vs payload {d['N_obs']:,}"
    m, s = sub[V].mean(), sub[V].std(ddof=1)
    Y = np.asarray(d["y"][0]); assert np.abs((sub[V].to_numpy() - m) / s - Y).max() < 1e-6, "rows do not match the payload"
    n_coh = d.get("N_cohort", 1)
    if n_coh > 1:   # the single-channel run_summary carries no cohort terms: read them from the chains
        ce = pd.concat([pd.read_csv(f, comment="#", usecols=[f"cohort_effect.1.{j}" for j in range(1, n_coh + 1)])
                        for f in sorted(fit_dir.glob("chains/*_[0-9].csv"))]).mean().to_numpy()
        Y = Y - ce[np.asarray(d["cohort_id"]) - 1]
    theta = np.array([p[f"theta[{k}]"]["mean"] for k in range(1, K + 1)])
    coef = np.array([[p[f"coef[1,{k},{j}]"]["mean"] for j in (1, 2, 3)] for k in range(1, K + 1)])
    sigma = p["sigma[1,1]"]["mean"]
    ar = d["ar_mode"]
    rho = np.array([[p[f"rho[{k}]"]["mean"]] for k in range(1, K + 1)]) if ar == 2 else np.zeros((K, 1))
    sm = [p["sigma_meas[1]"]["mean"]] if ar == 2 else [0.0]
    w, _ = M33.posterior_and_forecast(d, [Y], [coef], [sigma], sm, rho, theta)        # fitted window only
    fs, hs, he = (np.asarray(d[k]) for k in ("fit_start", "hold_start", "hold_end"))
    assert (he + 1 == fs).all() and (he - hs + 1 == KHELD).all(), "the held rows are not the rows before the fitted window"
    pids = sub.groupby("pidp", sort=False)["pidp"].first().to_numpy()
    # the model's own answer from the single first row: class prior times the marginal
    # density of y0 at age0, stationary variance (plus "spike") under AR(1) + "spike"
    X0 = np.asarray(d["X"])[hs - 1]; y0 = Y[hs - 1]
    mu0 = X0 @ coef.T
    var0 = (sigma ** 2 / (1 - rho[:, 0] ** 2) + sm[0] ** 2) if ar == 2 else np.full(K, sigma ** 2)
    lp = np.log(theta)[None, :] - 0.5 * np.log(2 * np.pi * var0)[None, :] - 0.5 * (y0[:, None] - mu0) ** 2 / var0[None, :]
    w0 = np.exp(lp - lp.max(axis=1, keepdims=True)); w0 /= w0.sum(axis=1, keepdims=True)
    out = pd.DataFrame({"pidp": pids, "type": w.argmax(axis=1)})
    for k in range(K):
        out[f"model_p{k}"] = w0[:, k]
        out[f"post_p{k}"] = w[:, k]          # the fitted-window posterior, the "truth" the targeting exercise scores against
    if KHELD == 2:   # the model's answer from both held rows: the window likelihood of rows hs..he, from the stationary state
        X = np.asarray(d["X"])
        if ar == 2:
            ll2, _ = M33.kalman_window(Y, X, coef, sigma, sm[0], rho[:, 0], np.asarray(d["age_gap"], float), hs, he)
        else:
            ll2, _ = M33.iid_window(Y, X, coef, sigma, hs, he)
        lp2 = np.log(theta)[None, :] + ll2
        w2 = np.exp(lp2 - lp2.max(axis=1, keepdims=True)); w2 /= w2.sum(axis=1, keepdims=True)
        for k in range(K):
            out[f"model2_p{k}"] = w2[:, k]
    return first.merge(out, on="pidp")


def multidim_types(sub: pd.DataFrame, first: pd.DataFrame):
    """Modal class from the fitted window (both channels plus survival through it) and the
    class posterior implied by the first row's two scores, for the first-row-held-out
    multidimensional fit."""
    if not (MD / "run_summary.json").exists():
        return None
    d = json.load(open(MD / "stan_data.json")); summ = json.load(open(MD / "run_summary.json"))
    p, mom, ch = summ["params"], summ["channel_moments"], summ["channels"]
    assert len(sub) == d["N_obs"], f"{MD.name}: rows {len(sub):,} vs payload {d['N_obs']:,}"
    Ys = [np.asarray(d["y"][c]) for c in range(2)]
    assert np.abs((sub[ch[0]].to_numpy() - mom[ch[0]]["mean"]) / mom[ch[0]]["sd"] - Ys[0]).max() < 1e-6
    theta = np.array([p[f"theta[{k}]"]["mean"] for k in range(1, K + 1)])
    coefs = [np.array([[p[f"coef[{c},{k},{j}]"]["mean"] for j in (1, 2, 3)] for k in range(1, K + 1)]) for c in (1, 2)]
    sigmas = [p[f"sigma[1,{c}]"]["mean"] for c in (1, 2)]
    rhos = np.array([[p[f"rho[{k},{c}]"]["mean"] for c in (1, 2)] for k in range(1, K + 1)])
    sm = [p[f"sigma_meas[{c}]"]["mean"] for c in (1, 2)]
    mort = (np.array([p[f"log_b[{k}]"]["mean"] for k in range(1, K + 1)]),
            np.array([p[f"gomp_slope[{k}]"]["mean"] for k in range(1, K + 1)]), p["makeham[1]"]["mean"])
    n_coh = d.get("N_cohort", 1)
    if n_coh > 1:   # net every score of its decade shift, as Stan computed the likelihood
        ce = np.zeros((2, n_coh))
        for c in range(2):
            ce[c, 1:] = [p[f"cohort_effect[{c + 1},{j}]"]["mean"] for j in range(2, n_coh + 1)]
        cid = np.asarray(d["cohort_id"]) - 1
        Ys = [Ys[c] - ce[c, cid] for c in range(2)]
    w, _ = M33.posterior_and_forecast(d, Ys, coefs, sigmas, sm, rhos, theta, mort=mort)
    fs, hs, he = (np.asarray(d[k]) for k in ("fit_start", "hold_start", "hold_end"))
    assert (he + 1 == fs).all() and (he - hs + 1 == MD_KHELD).all(), "the held rows are not the rows before the fitted window"
    pids = sub.groupby("pidp", sort=False)["pidp"].first().to_numpy()
    X0 = np.asarray(d["X"])[hs - 1]
    lp = np.log(theta)[None, :]
    for c in range(2):   # both channels' marginal densities at the first row, no mortality
        var0 = sigmas[c] ** 2 / (1 - rhos[:, c] ** 2) + sm[c] ** 2
        lp = lp - 0.5 * np.log(2 * np.pi * var0)[None, :] - 0.5 * (Ys[c][hs - 1][:, None] - X0 @ coefs[c].T) ** 2 / var0[None, :]
    w0 = np.exp(lp - lp.max(axis=1, keepdims=True)); w0 /= w0.sum(axis=1, keepdims=True)
    out = pd.DataFrame({"pidp": pids, "type": w.argmax(axis=1)})
    for k in range(K):
        out[f"model_p{k}"] = w0[:, k]
    return first.merge(out, on="pidp")


def score(df: pd.DataFrame, cols: list[str], test: np.ndarray, probs: np.ndarray | None = None) -> dict:
    """Scores on the test 30%; a set with missing values (region, ethnicity) is scored on the people
    who have them, and its probabilities are NaN for the rest."""
    keep = df[cols].notna().all(axis=1).to_numpy() if cols else np.ones(len(df), bool)
    if not keep.all():
        r = score(df[keep].reset_index(drop=True), cols, test[keep], None if probs is None else probs[keep])
        full = np.full((len(df), r["probs"].shape[1]), np.nan); full[keep] = r["probs"]; r["probs"] = full
        return r
    y = df["type"].to_numpy()
    if probs is None:
        X = np.column_stack([df["a0"], df["a0"] ** 2] + [df[c].to_numpy(float) for c in cols])
        clf = LogisticRegression(max_iter=2000, C=1e4).fit(X[~test], y[~test])
        probs = clf.predict_proba(X)
    pred = probs.argmax(axis=1)
    recall = [float((pred[test & (y == k)] == k).mean()) for k in np.unique(y[test])]
    return {"accuracy": float((pred[test] == y[test]).mean()), "balanced_accuracy": float(np.mean(recall)),
            "auc": float(roc_auc_score(y[test], probs[test], multi_class="ovr", average="macro")),
            "probs": probs}


def main() -> int:
    apply_style()
    sub, first = first_rows()
    frames = {}
    L = load_labels(f"{V}_first{'2' if KHELD == 2 else ''}held")
    frames["kmeans"] = first.merge(pd.DataFrame({"pidp": L.index, "type": L.to_numpy()}), on="pidp")
    for tag in ("base", "ssm") + (("ssm-cohort",) if KHELD == 2 else ()):
        f = bayes_types(tag, sub, first)
        if f is None:
            print(f"{tag}: first-two-rows-held-out fit not finished, skipped")
        else:
            frames[tag] = f
    if V == "theta":
        sub_md, first_md = first_rows(MULTIDIM_CONTRACT, extra=(MENTAL_COL,), kheld=MD_KHELD)
        f = multidim_types(sub_md, first_md.rename(columns={MENTAL_COL: "m0"}))
        if f is None:
            print("md: first-rows-held-out multidimensional fit not finished, skipped")
        else:
            frames["md"] = f
    obs = ["income_rank_mean", "female"] + [f"educ_{g}" for g in EDUC]
    educ = [f"educ_{g}" for g in EDUC]
    HEALTH = f"+ {V}(0)"
    regeth = [c for c in first.columns if c.startswith(("region_", "ethnic_"))]
    SETS = {"age(0)": [], "+ education": educ, "+ income rank": ["income_rank_mean"], "+ education, income, sex": obs,
            "+ education, income, sex, region, ethnicity": obs + regeth,
            f"+ {V}(1)": ["y1", "y1a"],         # the closest held reading, t-1
            HEALTH: ["y0", "y0a"]}               # the first row, t-2
    SETS[f"+ {V}(0), {V}(1)"] = ["y0", "y0a", "y1", "y1a"]
    SETS[f"+ {V}(0), education, income, sex"] = ["y0", "y0a"] + obs
    MD_SETS = {f"+ {V}(0), mental(0)": ["y0", "y0a", "m0", "m0a"]}      # multidimensional types only
    MODEL = "model posterior from the first row"
    MODEL2 = f"model posterior from {V}(0), {V}(1)"       # both held rows, the counterpart of the two-reading set
    rows, by_age = [], []
    for tag, lab in TYPOLOGIES:
        if tag not in frames:
            continue
        df = frames[tag].dropna(subset=["y0", "y1", "income_rank_mean"]).reset_index(drop=True)
        df["y0a"] = df["y0"] * df["a0"]
        if "y1" in df:
            df["y1a"] = df["y1"] * df["a0"]
        if "m0" in df:
            df["m0a"] = df["m0"] * df["a0"]
        test = ((pd.util.hash_pandas_object(df["pidp"], index=False) % 10) >= 7).to_numpy()
        shares = df["type"].value_counts(normalize=True).sort_index()
        majority = float(df.loc[test, "type"].value_counts(normalize=True).max())
        print(f"{lab}: {len(df):,} people, type shares {np.round(shares.to_numpy(), 3)}, majority rate on the test 30% {majority:.3f}")
        probs_by_set = {}
        for name, cols in {**SETS, **(MD_SETS if tag == "md" else {})}.items():
            if any(c not in df for c in cols):      # the placeholder multidimensional fit has one held reading only
                continue
            r = score(df, cols, test); probs_by_set[name] = r.pop("probs")
            rows.append({"typology": tag, "label": lab, "predictors": name, "n": len(df), "n_test": int(test.sum()), "majority": majority, **r})
            print(f"  {name:36s} accuracy {r['accuracy']:.3f}  macro AUC {r['auc']:.3f}")
        if tag != "kmeans":
            r = score(df, [], test, probs=df[[f"model_p{k}" for k in range(K)]].to_numpy()); probs_by_set[MODEL] = r.pop("probs")
            rows.append({"typology": tag, "label": lab, "predictors": MODEL, "n": len(df), "n_test": int(test.sum()), "majority": majority, **r})
            print(f"  {MODEL:36s} accuracy {r['accuracy']:.3f}  macro AUC {r['auc']:.3f}")
        if "model2_p0" in df:
            r = score(df, [], test, probs=df[[f"model2_p{k}" for k in range(K)]].to_numpy()); r.pop("probs")
            rows.append({"typology": tag, "label": lab, "predictors": MODEL2, "n": len(df), "n_test": int(test.sum()), "majority": majority, **r})
            print(f"  {MODEL2:36s} accuracy {r['accuracy']:.3f}  macro AUC {r['auc']:.3f}")
        for set_name, age_col in ((HEALTH, "age0"), (f"+ {V}(1)", "age1")):     # each reading by the age it was taken
            pred = probs_by_set[set_name].argmax(axis=1)
            for lo, hi in AGE_BANDS:
                sel = test & df[age_col].between(lo, hi).to_numpy()
                yb = df.loc[sel, "type"].to_numpy(); pb = pred[sel]
                by_age.append({"typology": tag, "label": lab, "predictors": set_name, "band": f"{lo}-{hi}", "n_test": int(sel.sum()),
                               "accuracy": float((pb == yb).mean()),
                               "balanced_accuracy": float(np.mean([(pb[yb == k] == k).mean() for k in np.unique(yb)])),
                               "majority": float(pd.Series(yb).value_counts(normalize=True).max())})
    t = pd.DataFrame(rows); t.to_csv(DESC / f"paper_type_from_initial{SFX}.csv", index=False)
    ba = pd.DataFrame(by_age); ba.to_csv(DESC / f"paper_type_from_initial_by_age{SFX}.csv", index=False)
    have = [(tag, lab) for tag, lab in TYPOLOGIES if tag in frames]

    # ---- table
    sets = list(SETS) + (list(MD_SETS) if "md" in frames else []) + [MODEL] + ([MODEL2] if KHELD == 2 else [])
    with open(TAB / f"tab_type_from_initial{SFX}.tex", "w") as f:
        f.write("\\begin{tabular}{l" + "rr" * len(have) + "}\n\\toprule\n")
        SHORT = {"kmeans": "K-means", "base": "indep. residuals", "ssm": 'AR(1) + "spike"', "ssm-cohort": "+ cohort shifts",
                 "md": "multidim"}       # short headers so five typologies fit
        f.write("predictors at the first rows & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{SHORT[tag]}}}" for tag, _ in have) + " \\\\\n")
        f.write(" ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(have))) + "\n & " + " & ".join(["accuracy & AUC"] * len(have)) + " \\\\\n\\midrule\n")
        f.write("majority type & " + " & ".join(f"{t[t['typology'] == tag]['majority'].iloc[0]:.3f} & " for tag, _ in have) + " \\\\\n")
        for name in sets:
            cells = []
            for tag, _ in have:
                q = t[(t["typology"] == tag) & (t["predictors"] == name)]
                cells.append(f"{q['accuracy'].iloc[0]:.3f} & {q['auc'].iloc[0]:.3f}" if len(q) else " & ")
            f.write(name.replace("mental", "mental $\\theta$").replace(V + "(", VL + "(").replace(" x ", " $\\times$ ") + " & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---- figure: accuracy and macro AUC by predictor set, the observables against the health reading
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"wspace": 0.08}, sharey=True)
    cols = {"kmeans": INK2, "base": CLUSTER[2], "ssm": CLUSTER[0], "ssm-cohort": "#6a51a3", "md": PURPLE}
    width = 0.8 / len(have)
    for ax, metric, title in ((axes[0], "accuracy", "(a) accuracy on the test 30%\n(dashed: the majority-type rate)"),
                              (axes[1], "auc", "(b) macro one-vs-rest AUC on the test 30%")):
        for i, (tag, lab) in enumerate(have):
            q = t[t["typology"] == tag].set_index("predictors").reindex(sets)
            ypos = np.arange(len(sets)) + (i - (len(have) - 1) / 2) * width
            ax.barh(ypos, q[metric].fillna(0), height=width * 0.92, color=cols[tag], label=lab)
            if metric == "accuracy":
                ax.axvline(q["majority"].dropna().iloc[0], color=cols[tag], lw=1.0, ls="--")
        ax.set_xlim(0.3 if metric == "accuracy" else 0.45, max(0.75, t[metric].max() + 0.05)); ax.grid(True, axis="x")
        ax.set_title(title, fontsize=9, loc="left")
    axes[0].set_yticks(np.arange(len(sets)))
    axes[0].set_yticklabels([n.replace("mental", r"mental $\theta$").replace(V + "(", VL + "(") for n in sets], fontsize=8)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, loc="lower center", ncol=len(have), bbox_to_anchor=(0.5, -0.02), frameon=False)
    n = len(frames[have[0][0]]); n_md = len(frames["md"]) if "md" in frames else 0
    held = "first two rows"
    footer = (f"Health contract people with at least {MIN_ROWS} rows ({n:,}), on {V}; every typology is fitted without each person's {held} (K-means on the remaining rows; the mixtures with the {held} held out, "
             "type = modal class of the posterior from the fitted window)."
             + (f" The multidimensional fit (theta + mental GRM + mortality) is on the multidim contract's people with at least {MD_KHELD + 3} rows ({n_md:,}), "
                + "its posterior from both channels and survival through the fitted window." if n_md else "")
             + f"\nEvery predictor set is on top of a quadratic in age at the first row: education (three groups), income rank, the three observables together, and with region and ethnic group, {V} at the second held row (t-1) or the first (t-2) "
             "(with its age interaction), both readings, and the first reading plus the observables; multinomial logit, people split 70/30 by a hash of pidp, fitted on the 70, scored on the 30.\n\"Model posterior\": the class posterior the fitted mixture itself implies "
             "from the single first observation (class prior times the marginal density at that age; stationary variance plus \"spike\" under AR(1) + \"spike\"), no regression. "
             "AUC is the macro one-vs-rest AUC over the three types." + (" The multidimensional model's posterior uses both first-row scores." if n_md else "")
             + (" The posterior from both held rows is the Kalman likelihood of the two rows from the stationary state." if KHELD == 2 else ""))
    fig.text(0.01, -0.09, "\n".join(textwrap.wrap(footer.replace("\n", " "), 200)), fontsize=7.2, color=INK2, va="top")
    fig.savefig(FIG / f"fig_type_from_initial{SFX}.png", bbox_inches="tight")
    print(f"wrote fig_type_from_initial{SFX}.png and tab_type_from_initial{SFX}.tex ({', '.join(lab for _, lab in have)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
