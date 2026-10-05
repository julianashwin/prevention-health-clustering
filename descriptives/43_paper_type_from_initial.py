"""How well a person's health type can be predicted from their initial health value.

The types come from fits that never saw the initial row: partial K-means on theta
with each person's first observed row dropped (22_paper_kmeans.py --first-held,
variant theta_firstheld), and the K=3 independent-residuals and AR(1) plus "spike"
mixtures with the first row held out (clustering/runs/health_firstheld_queue.py,
artifacts/health-firstheld/), and the multidimensional AR(1) plus "spike" fit with
mortality and the first row held out (clustering/runs/multidim_firstheld_launch.sh,
artifacts/multidim-health/theta-ssm-mort-firstho, on the multidim contract). People
with at least four rows, so everyone keeps at least three fitted rows. The
Bayesian type is the modal class of the posterior computed from the fitted window
alone (for the multidimensional fit: both channels plus survival through the
fitted window).

Predictors known at the first row: age (quadratic), the observables (income
rank, sex, education), theta at the first row, and its interaction with age; for
the multidimensional types also the mental score at the first row.
Multinomial logit on the modal type, people split 70/30 by a hash of pidp,
fitted on the 70, scored on the 30: accuracy (against the majority-type rate)
and the macro one-vs-rest AUC. For the mixtures there is also the model's own
answer, the class posterior implied by the single first observation under the
fitted parameters (stationary variance plus "spike" under AR(1) + "spike"), which
needs no regression. A second panel splits the "+ theta(0) x age" accuracy by the
age at the first observation.

Outputs: paper/figures/fig_type_from_initial.png, paper/tables/tab_type_from_initial.tex,
artifacts/descriptives/paper_type_from_initial{,_by_age}.csv. A fit that has not
finished is skipped with a note.

    PYTHONPATH=src .venv/bin/python descriptives/43_paper_type_from_initial.py
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
from _paper_common import CONTRACT, DESC, FIG, MENTAL_COL, MULTIDIM_CONTRACT, TAB, load_labels  # noqa: E402
from _style import CLUSTER, GREEN, INK2, ORANGE, PURPLE, apply_style  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR  # noqa: E402

M33 = importlib.import_module("33_paper_prediction")
K = 3
V = "theta"
MIN_ROWS = 4
FH = ARTIFACTS_DIR / "health-firstheld"
MD = ARTIFACTS_DIR / "multidim-health" / "theta-ssm-mort-firstho"
EDUC = ["GCSE", "A-level", "degree or higher"]
AGE_BANDS = [(20, 34), (35, 49), (50, 64), (65, 90)]
TYPOLOGIES = [("kmeans", "K-means"), ("base", "independent residuals"), ("ssm", 'AR(1) + "spike"'), ("md", "multidimensional")]


def first_rows(contract=CONTRACT, extra: tuple[str, ...] = ()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The retained people's rows (sorted) and their first row with the observables."""
    long = pd.read_csv(contract / "long.csv").sort_values(["pidp", "age"]).reset_index(drop=True)
    n = long.groupby("pidp")["age"].transform("size")
    sub = long[n >= MIN_ROWS].reset_index(drop=True)
    first = sub.groupby("pidp", as_index=False).first()[["pidp", "age", V, *extra]].rename(columns={"age": "age0", V: "y0"})
    panel = pd.read_csv(PROCESSED_DATA_DIR / "ukhls_indresp_processed.csv", usecols=["pidp", "sex"])
    per = pd.read_parquet(PROCESSED_DATA_DIR / "measures" / "observables_person.parquet").set_index("pidp")
    first = first.join(per[["educ_group", "income_rank_mean"]], on="pidp")
    first["female"] = first["pidp"].map(panel.groupby("pidp")["sex"].first()).eq(2).astype(float)
    for g in EDUC:
        first[f"educ_{g}"] = (first["educ_group"] == g).astype(float)
    first["a0"] = (first["age0"] - 55) / 10
    return sub, first


def bayes_types(tag: str, sub: pd.DataFrame, first: pd.DataFrame):
    """Modal class from the fitted window, and the class posterior implied by the first
    row alone, for the first-row-held-out fit ``tag`` (base or ssm)."""
    fit_dir = FH / f"health-{V}-{tag}-firstho"
    if not (fit_dir / "run_summary.json").exists():
        return None
    d = json.load(open(fit_dir / "stan_data.json")); p = json.load(open(fit_dir / "run_summary.json"))["params"]
    assert len(sub) == d["N_obs"], f"{fit_dir.name}: rows {len(sub):,} vs payload {d['N_obs']:,}"
    m, s = sub[V].mean(), sub[V].std(ddof=1)
    Y = np.asarray(d["y"][0]); assert np.abs((sub[V].to_numpy() - m) / s - Y).max() < 1e-6, "rows do not match the payload"
    theta = np.array([p[f"theta[{k}]"]["mean"] for k in range(1, K + 1)])
    coef = np.array([[p[f"coef[1,{k},{j}]"]["mean"] for j in (1, 2, 3)] for k in range(1, K + 1)])
    sigma = p["sigma[1,1]"]["mean"]
    ar = d["ar_mode"]
    rho = np.array([[p[f"rho[{k}]"]["mean"]] for k in range(1, K + 1)]) if ar == 2 else np.zeros((K, 1))
    sm = [p["sigma_meas[1]"]["mean"]] if ar == 2 else [0.0]
    w, _ = M33.posterior_and_forecast(d, [Y], [coef], [sigma], sm, rho, theta)        # fitted window only
    fs, hs = np.asarray(d["fit_start"]), np.asarray(d["hold_start"])
    assert (hs + 1 == fs).all(), "the held row is not the row before the fitted window"
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
    w, _ = M33.posterior_and_forecast(d, Ys, coefs, sigmas, sm, rhos, theta, mort=mort)
    fs, hs = np.asarray(d["fit_start"]), np.asarray(d["hold_start"])
    assert (hs + 1 == fs).all(), "the held row is not the row before the fitted window"
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
    y = df["type"].to_numpy()
    if probs is None:
        X = np.column_stack([df["a0"], df["a0"] ** 2] + [df[c].to_numpy(float) for c in cols])
        clf = LogisticRegression(max_iter=2000, C=1e4).fit(X[~test], y[~test])
        probs = clf.predict_proba(X)
    pred = probs.argmax(axis=1)
    return {"accuracy": float((pred[test] == y[test]).mean()),
            "auc": float(roc_auc_score(y[test], probs[test], multi_class="ovr", average="macro")),
            "probs": probs}


def main() -> int:
    apply_style()
    sub, first = first_rows()
    frames = {}
    L = load_labels("theta_firstheld")
    frames["kmeans"] = first.merge(pd.DataFrame({"pidp": L.index, "type": L.to_numpy()}), on="pidp")
    for tag in ("base", "ssm"):
        f = bayes_types(tag, sub, first)
        if f is None:
            print(f"{tag}: first-row-held-out fit not finished, skipped")
        else:
            frames[tag] = f
    sub_md, first_md = first_rows(MULTIDIM_CONTRACT, extra=(MENTAL_COL,))
    f = multidim_types(sub_md, first_md.rename(columns={MENTAL_COL: "m0"}))
    if f is None:
        print("md: first-row-held-out multidimensional fit not finished, skipped")
    else:
        frames["md"] = f
    obs = ["income_rank_mean", "female"] + [f"educ_{g}" for g in EDUC]
    SETS = {"age(0)": [], "+ observables": obs, "+ theta(0)": ["y0"], "+ theta(0) x age(0)": ["y0", "y0a"],
            "+ theta(0) x age(0), observables": ["y0", "y0a"] + obs}
    MD_SETS = {"+ theta(0), mental(0) x age(0)": ["y0", "y0a", "m0", "m0a"]}      # multidimensional types only
    MODEL = "model posterior from the first row"
    rows, by_age = [], []
    for tag, lab in TYPOLOGIES:
        if tag not in frames:
            continue
        df = frames[tag].dropna(subset=["y0", "income_rank_mean"]).reset_index(drop=True)
        df["y0a"] = df["y0"] * df["a0"]
        if "m0" in df:
            df["m0a"] = df["m0"] * df["a0"]
        test = ((pd.util.hash_pandas_object(df["pidp"], index=False) % 10) >= 7).to_numpy()
        shares = df["type"].value_counts(normalize=True).sort_index()
        majority = float(df.loc[test, "type"].value_counts(normalize=True).max())
        print(f"{lab}: {len(df):,} people, type shares {np.round(shares.to_numpy(), 3)}, majority rate on the test 30% {majority:.3f}")
        probs_by_set = {}
        for name, cols in {**SETS, **(MD_SETS if tag == "md" else {})}.items():
            r = score(df, cols, test); probs_by_set[name] = r.pop("probs")
            rows.append({"typology": tag, "label": lab, "predictors": name, "n": len(df), "n_test": int(test.sum()), "majority": majority, **r})
            print(f"  {name:36s} accuracy {r['accuracy']:.3f}  macro AUC {r['auc']:.3f}")
        if tag != "kmeans":
            r = score(df, [], test, probs=df[[f"model_p{k}" for k in range(K)]].to_numpy()); probs_by_set[MODEL] = r.pop("probs")
            rows.append({"typology": tag, "label": lab, "predictors": MODEL, "n": len(df), "n_test": int(test.sum()), "majority": majority, **r})
            print(f"  {MODEL:36s} accuracy {r['accuracy']:.3f}  macro AUC {r['auc']:.3f}")
        pred = probs_by_set["+ theta(0) x age(0)"].argmax(axis=1)
        for lo, hi in AGE_BANDS:
            sel = test & df["age0"].between(lo, hi).to_numpy()
            yb = df.loc[sel, "type"].to_numpy()
            by_age.append({"typology": tag, "label": lab, "band": f"{lo}-{hi}", "n_test": int(sel.sum()),
                           "accuracy": float((pred[sel] == yb).mean()), "majority": float(pd.Series(yb).value_counts(normalize=True).max())})
    t = pd.DataFrame(rows); t.to_csv(DESC / "paper_type_from_initial.csv", index=False)
    ba = pd.DataFrame(by_age); ba.to_csv(DESC / "paper_type_from_initial_by_age.csv", index=False)
    have = [(tag, lab) for tag, lab in TYPOLOGIES if tag in frames]

    # ---- table
    sets = list(SETS) + list(MD_SETS) + [MODEL]
    with open(TAB / "tab_type_from_initial.tex", "w") as f:
        f.write("\\begin{tabular}{l" + "rr" * len(have) + "}\n\\toprule\n")
        f.write("predictors at the first row & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for _, lab in have) + " \\\\\n")
        f.write(" ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(have))) + "\n & " + " & ".join(["accuracy & AUC"] * len(have)) + " \\\\\n\\midrule\n")
        f.write("majority type & " + " & ".join(f"{t[t['typology'] == tag]['majority'].iloc[0]:.3f} & " for tag, _ in have) + " \\\\\n")
        for name in sets:
            cells = []
            for tag, _ in have:
                q = t[(t["typology"] == tag) & (t["predictors"] == name)]
                cells.append(f"{q['accuracy'].iloc[0]:.3f} & {q['auc'].iloc[0]:.3f}" if len(q) else " & ")
            f.write(name.replace("theta", "$\\theta$").replace("mental", "mental $\\theta$").replace(" x ", " $\\times$ ") + " & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---- figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"wspace": 0.3, "width_ratios": [1.25, 1]})
    ax = axes[0]
    cols = {"kmeans": INK2, "base": CLUSTER[2], "ssm": CLUSTER[0], "md": PURPLE}
    width = 0.8 / len(have)
    for i, (tag, lab) in enumerate(have):
        q = t[t["typology"] == tag].set_index("predictors").reindex(sets)
        ypos = np.arange(len(sets)) + (i - (len(have) - 1) / 2) * width
        ax.barh(ypos, q["accuracy"].fillna(0), height=width * 0.92, color=cols[tag], label=lab)
        ax.axvline(q["majority"].dropna().iloc[0], color=cols[tag], lw=1.0, ls="--")
    ax.set_yticks(np.arange(len(sets))); ax.set_yticklabels([n.replace("theta", r"$\theta$").replace("mental", r"mental $\theta$").replace(" x ", r" $\times$ ") for n in sets], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0.3, max(0.75, t["accuracy"].max() + 0.05)); ax.grid(True, axis="x")
    ax.set_title("(a) accuracy on the test 30%, by predictor set\n(dashed: the majority-type rate)", fontsize=9, loc="left")
    ax.legend(fontsize=7.5, loc="upper right")
    ax = axes[1]
    x = np.arange(len(AGE_BANDS))
    for tag, lab in have:
        q = ba[ba["typology"] == tag].set_index("band").reindex([f"{lo}-{hi}" for lo, hi in AGE_BANDS])
        ax.plot(x, q["accuracy"], marker="o", color=cols[tag], lw=1.8, label=lab)
        ax.plot(x, q["majority"], color=cols[tag], lw=1.0, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([f"{lo}-{hi}" for lo, hi in AGE_BANDS]); ax.set_xlabel("age at the first observation")
    ax.set_ylim(0.3, 1.0); ax.grid(True, axis="y")
    ax.set_title(r"(b) accuracy of age(0) + $\theta$(0) $\times$ age(0), by age at the first row" + "\n(dashed: the majority-type rate in the band)", fontsize=9, loc="left")
    ax.legend(fontsize=7.5, loc="upper left")
    n = len(frames[have[0][0]]); n_md = len(frames["md"]) if "md" in frames else 0
    footer = (f"Health contract people with at least four rows ({n:,}); every typology is fitted without each person's first row (K-means on the remaining rows; the mixtures with the first row held out, "
             "type = modal class of the posterior from the fitted window)."
             + (f" The multidimensional fit (theta + mental GRM + mortality) is on the multidim contract's people with at least four rows ({n_md:,}), its posterior from both channels and survival through the fitted window." if n_md else "")
             + "\nPredictors are known at the first row: age (quadratic), the observables (income rank, sex, education), theta at the first row and its "
             "interaction with age; multinomial logit, people split 70/30 by a hash of pidp, fitted on the 70, scored on the 30.\n\"Model posterior\": the class posterior the fitted mixture itself implies "
             "from the single first observation (class prior times the marginal density at that age; stationary variance plus \"spike\" under AR(1) + \"spike\"), no regression. "
             "AUC is the macro one-vs-rest AUC over the three types." + (" The multidimensional model's posterior uses both first-row scores." if n_md else ""))
    fig.text(0.01, -0.03, "\n".join(textwrap.wrap(footer.replace("\n", " "), 200)), fontsize=7.2, color=INK2, va="top")
    fig.savefig(FIG / "fig_type_from_initial.png", bbox_inches="tight")
    print(f"wrote fig_type_from_initial.png and tab_type_from_initial.tex ({', '.join(lab for _, lab in have)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
