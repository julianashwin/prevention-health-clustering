"""What should prevention target on, when the type is not observed?

The return to prevention by class, P_j(r) (42_paper_returns_bayes.py, the K = 3 AR(1) +
"spike" classes), is only collectable if the policy can tell who is in which class. This
exercise prices targeting rules that use what is known at a person's FIRST observation,
with the types taken from the first-row-held-out fit so that the first row never shaped
them (43_paper_type_from_initial.py):

  information set g      age(0) alone; + education, income, sex; + theta(0); + theta(0) and the
                         observables; the model's own posterior from the first row; and the
                         oracle, the class posterior from the person's fitted window
  predicted return       E[P | g, i] = sum_j p_ij^g P_j(r_i), r_i the age at the first row, with
                         p^g the multinomial-logit class probabilities fitted on the 70% split
  targeting rule         treat the share q of the test 30% with the highest predicted return
  realised return        the average over the treated of sum_j w_ij P_j(r_i), w the fitted-window
                         class posterior

The age-only rule is the benchmark: returns fall with age, so ranking on age alone already
collects something; what the health reading and the observables add on top is the value of
type information. --first2 uses the first-two-rows fits and adds the set with both readings;
--h the same on h.

Outputs: paper/figures/fig_targeting{,_h,_first2}.png, paper/tables/tab_targeting*.tex,
artifacts/descriptives/paper_targeting*.csv.

    PYTHONPATH=src .venv/bin/python descriptives/46_paper_targeting.py [--h] [--first2]
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import DESC, FIG, TAB  # noqa: E402
from _style import CLUSTER, GREEN, INK2, ORANGE, PURPLE, VERM, apply_style  # noqa: E402

M43 = importlib.import_module("43_paper_type_from_initial")
K = 3
V, KHELD, SFX, VL = M43.V, M43.KHELD, M43.SFX, M43.VL
EDUC = M43.EDUC
Q_GRID = np.round(np.arange(0.05, 1.0001, 0.05), 2)
Q_TABLE = (0.10, 0.25, 0.50)


def predicted_probs(df: pd.DataFrame, cols: list[str], test: np.ndarray) -> np.ndarray:
    """Multinomial logit of the modal type on a quadratic in age plus ``cols``, fitted on the 70%."""
    X = np.column_stack([df["a0"], df["a0"] ** 2] + [df[c].to_numpy(float) for c in cols])
    clf = LogisticRegression(max_iter=2000, C=1e4).fit(X[~test], df["type"].to_numpy()[~test])
    return clf.predict_proba(X)


def main() -> int:
    apply_style()
    sub, first = M43.first_rows()
    df = M43.bayes_types("ssm", sub, first)
    if df is None:
        print("the first-rows-held-out AR(1) + spike fit is not finished"); return 1
    df = df.dropna(subset=["y0", "income_rank_mean"]).reset_index(drop=True)
    df["y0a"] = df["y0"] * df["a0"]
    if "y1" in df:
        df["y1a"] = df["y1"] * df["a0"]
    test = ((pd.util.hash_pandas_object(df["pidp"], index=False) % 10) >= 7).to_numpy()
    # the return by class at the person's first-row age
    R = pd.read_csv(DESC / f"paper_returns_bayes{'' if V == 'theta' else '_h'}_ssm.csv").set_index("r")
    r_i = df["age0"].clip(R.index.min(), R.index.max()).astype(int).to_numpy()
    Pj = np.column_stack([R.loc[r_i, f"P_{k + 1}"].to_numpy() for k in range(K)])          # people x K
    W = df[[f"post_p{k}" for k in range(K)]].to_numpy()
    realised = (W * Pj).sum(axis=1)                                                            # what treating person i is worth
    obs = ["income_rank_mean", "female"] + [f"educ_{g}" for g in EDUC]
    SETS = {"age(0)": [], "+ education, income, sex": obs, f"+ {V}(0)": ["y0", "y0a"], f"+ {V}(0), education, income, sex": ["y0", "y0a"] + obs}
    if KHELD == 2:
        SETS[f"+ {V}(0), {V}(1)"] = ["y0", "y0a", "y1", "y1a"]
    MODEL, ORACLE = "model posterior from the first row", "oracle: the fitted-window class"
    probs = {name: predicted_probs(df, cols, test) for name, cols in SETS.items()}
    probs[MODEL] = df[[f"model_p{k}" for k in range(K)]].to_numpy()
    probs[ORACLE] = W
    expected = {name: (p * Pj).sum(axis=1) for name, p in probs.items()}
    # treat the top share q of the test people by predicted return; score with the realised return
    rows = []
    uniform = float(realised[test].mean())
    for name in list(SETS) + [MODEL, ORACLE]:
        e = expected[name][test]; val = realised[test]
        order = np.argsort(-e)
        for q in Q_GRID:
            n = max(1, int(round(q * len(order))))
            rows.append({"rule": name, "q": q, "n_treated": n, "return_per_treated": float(val[order[:n]].mean()),
                         "share_of_total": float(val[order[:n]].sum() / val.sum())})
    t = pd.DataFrame(rows); t["uniform"] = uniform
    t.to_csv(DESC / f"paper_targeting{SFX}.csv", index=False)
    print(f"{len(df):,} people ({test.sum():,} scored); uniform return per person £{uniform:,.0f}")
    piv = t.pivot(index="q", columns="rule", values="return_per_treated")
    print(piv.loc[list(Q_TABLE)].round(0).to_string())

    # ---- table: return per treated person at three coverages, and the share of the oracle's
    rules = list(SETS) + [MODEL, ORACLE]
    tabrows = []
    for name in rules:
        cells = []
        for q in Q_TABLE:
            v = piv.loc[q, name]; o = piv.loc[q, ORACLE]
            cells += [f"{v:,.0f}", f"{v / uniform:.2f}", f"{v / o:.2f}"]
        tabrows.append([name.replace(V + "(", VL + "("), *cells])
    with open(TAB / f"tab_targeting{SFX}.tex", "w") as fh:        # grouped header: one block per coverage
        fh.write("\\begin{tabular}{l" + "rrr" * len(Q_TABLE) + "}\n\\toprule\n")
        fh.write("targeting on & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{top {int(100 * q)}\\% treated}}" for q in Q_TABLE) + " \\\\\n")
        fh.write(" ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(Q_TABLE))) + "\n & " + " & ".join(["\\pounds & /uniform & /oracle"] * len(Q_TABLE)) + " \\\\\n\\midrule\n")
        for row in tabrows:
            fh.write(" & ".join(row) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    # ---- figure
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"wspace": 0.25})
    cols = {"age(0)": INK2, "+ education, income, sex": ORANGE, f"+ {V}(0)": CLUSTER[1], f"+ {V}(0), education, income, sex": CLUSTER[0],
            f"+ {V}(0), {V}(1)": PURPLE, MODEL: VERM, ORACLE: GREEN}
    ax = axes[0]
    for name in rules:
        q = piv.index.to_numpy(); ax.plot(q, piv[name] / 1000, color=cols[name], lw=2.2 if name in (ORACLE, MODEL) else 1.6,
                                         ls="--" if name == ORACLE else "-", label=name.replace(V + "(", VL + "("))
    ax.axhline(uniform / 1000, color=INK2, lw=1.0, ls=":", label="uniform: everyone treated")
    ax.set_xlabel("share of people treated $q$ (the top $q$ by predicted return)"); ax.set_ylabel("realised return per treated person, £000")
    ax.set_title("(a) return per treated person against coverage", loc="left", fontsize=9.5); ax.grid(True, axis="y"); ax.legend(fontsize=7, loc="upper right")
    ax = axes[1]
    for name in rules:
        q = piv.index.to_numpy(); sh = t[t["rule"] == name].set_index("q")["share_of_total"].reindex(q)
        ax.plot(q, sh, color=cols[name], lw=2.2 if name in (ORACLE, MODEL) else 1.6, ls="--" if name == ORACLE else "-")
    ax.plot([0, 1], [0, 1], color=INK2, lw=1.0, ls=":")
    ax.set_xlabel("share of people treated $q$"); ax.set_ylabel("share of the total return collected")
    ax.set_title("(b) share of the total return collected against coverage", loc="left", fontsize=9.5); ax.grid(True, axis="y")
    footer = (f"The K = 3 AR(1) + \"spike\" classes on {V} with the {'first row' if KHELD == 1 else 'first two rows'} held out ({len(df):,} people); the return to prevention by class P_j(r) of the returns figure, "
              f"at each person's age at the first row. Each rule predicts a person's return from what is known at that row (a multinomial logit of the fitted-window class on a quadratic in age plus the set shown, "
              f"fitted on a 70% split; the model posterior and the oracle need no regression), treats the top q of the other 30% by predicted return, and is scored by the realised return, "
              "the fitted-window class posterior times P_j(r). The age-only rule is the benchmark: returns fall with age, so what the reading and the observables add is the value of type information. "
              "Dotted: everyone treated. Pounds of treatment cost avoided to 90 per unit of mu, no survival weighting, no discounting.")
    fig.text(0.01, -0.03, "\n".join(textwrap.wrap(footer, 200)), fontsize=7.2, color=INK2, va="top")
    fig.savefig(FIG / f"fig_targeting{SFX}.png", bbox_inches="tight")
    print(f"wrote fig_targeting{SFX}.png and tab_targeting{SFX}.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
