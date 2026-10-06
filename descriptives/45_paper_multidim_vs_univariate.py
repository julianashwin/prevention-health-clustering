"""Do the multidimensional classes differ from the univariate physical and mental ones?

Three K = 3 AR(1) + "spike" fits with birth-decade shifts, all read at the 1950s:
  physical   health-theta-ssm-cohort        theta alone, the health contract
  mental     mental-theta-ssm-cohort        the mental GRM alone, the multidim contract
  multidim   theta-ssm-mort-cohort          theta + mental GRM + Gompertz-Makeham mortality
Panels (a) and (b) draw each channel's class paths from the univariate fit (solid) and
from the multidimensional fit (dashed); (c) and (d) cross-tabulate the multidimensional
modal class against the univariate physical and mental modal classes on the people in
both fits, rows normalised, with the ARI and the share in the same class. Classes are
ordered worst to best on each fit's anchor (the physical channel for the multidim fit).

Outputs: paper/figures/fig_multidim_vs_univariate.png,
artifacts/descriptives/paper_multidim_vs_univariate.csv (paths and agreement).

    PYTHONPATH=src .venv/bin/python descriptives/45_paper_multidim_vs_univariate.py
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import AGES, DESC, FIG, MENTAL_COL, MULTIDIM_CONTRACT, bayes_fit  # noqa: E402
from _style import CLUSTER, INK2, apply_style, share_ylim  # noqa: E402

from prevention_health_clustering.config import ARTIFACTS_DIR  # noqa: E402

M39 = importlib.import_module("39_paper_multidim")
K = 3
A = (AGES - 55) / 10
MD_TAG = "theta-ssm-mort-cohort"


def uni_paths(r) -> np.ndarray:
    """Class paths in the measure's units from a bayes_fit result (already at the 1950s when cohort)."""
    return np.vstack([r["mom"]["mean"] + r["mom"]["sd"] * (c[0] + c[1] * A + c[2] * A ** 2) for c in r["coef"]])


def modal(post: pd.DataFrame) -> pd.Series:
    cols = [f"class{k + 1}" for k in range(K)]
    return pd.Series(post[cols].to_numpy().argmax(axis=1), index=post["pidp"])


def main() -> int:
    apply_style()
    phys = bayes_fit(ARTIFACTS_DIR / "health-ssm-cohort" / "health-theta-ssm-cohort", "theta", K)
    ment = bayes_fit(ARTIFACTS_DIR / "mental-health" / "mental-theta-ssm-cohort", "mental", K, contract=MULTIDIM_CONTRACT, col=MENTAL_COL)
    md = M39.load_fit(MD_TAG, "theta", None)
    P = {"physical": uni_paths(phys), "mental": uni_paths(ment)}
    rows = []
    for name, paths, src in (("physical", P["physical"], "univariate"), ("mental", P["mental"], "univariate"),
                             ("physical", md["paths"][0], "multidim"), ("mental", md["paths"][1], "multidim")):
        for k in range(K):
            rows.append({"channel": name, "fit": src, "class": k + 1, "share": (phys if name == "physical" else ment)["theta"][k] if src == "univariate" else md["share"][k],
                         "at_20": paths[k][0], "at_55": paths[k][35], "at_90": paths[k][-1]})
    # agreement on the shared people
    m_md = modal(md["post"]); agree = {}
    for name, r in (("physical", phys), ("mental", ment)):
        m_u = modal(r["post"]); both = m_md.index.intersection(m_u.index)
        a, b = m_md.loc[both].to_numpy(), m_u.loc[both].to_numpy()
        ct = pd.crosstab(a, b, normalize="index")
        agree[name] = {"n": len(both), "ari": adjusted_rand_score(a, b), "same": float((a == b).mean()), "ct": ct}
        print(f"multidim vs univariate {name}: {len(both):,} people, ARI {agree[name]['ari']:.2f}, same class {agree[name]['same']:.0%}\n{ct.round(2).to_string()}")
    pd.DataFrame(rows).to_csv(DESC / "paper_multidim_vs_univariate.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), gridspec_kw={"wspace": 0.32, "width_ratios": [1.3, 1.3, 0.7, 0.7]})
    for j, (name, r, lab) in enumerate((("physical", phys, r"$\theta$"), ("mental", ment, r"mental $\theta$"))):
        ax = axes[j]
        for k in range(K):
            ax.plot(AGES, P[name][k], color=CLUSTER[k], lw=1.0 + 3.5 * r["theta"][k], label=f"univariate class {k + 1} ({r['theta'][k]:.0%}, $\\rho$ {r['rho'][k]:.2f})")
            ax.plot(AGES, md["paths"][j][k], color=CLUSTER[k], lw=1.0 + 3.5 * md["share"][k], ls="--",
                    label=f"multidim class {k + 1} ({md['share'][k]:.0%}, $\\rho$ {md['rho'][k, j]:.2f})")
        ax.set_title(f"({'ab'[j]}) {lab}: univariate (solid), multidimensional (dashed)", loc="left", fontsize=9)
        ax.legend(fontsize=6.5, loc="lower left", ncols=2); ax.grid(True, axis="y"); ax.set_xlim(AGES[0], AGES[-1]); ax.set_xlabel("age")
    for j, name in enumerate(("physical", "mental")):
        ax = axes[2 + j]; ct = agree[name]["ct"].reindex(index=range(K), columns=range(K)).fillna(0).to_numpy()
        ax.imshow(ct, cmap="Blues", vmin=0, vmax=1)
        for i in range(K):
            for c in range(K):
                ax.text(c, i, f"{ct[i, c]:.2f}", ha="center", va="center", fontsize=8, color="white" if ct[i, c] > 0.55 else INK2)
        ax.set_xticks(range(K)); ax.set_xticklabels([f"{k + 1}" for k in range(K)]); ax.set_yticks(range(K)); ax.set_yticklabels([f"{k + 1}" for k in range(K)])
        ax.set_xlabel(f"univariate {name} class"); ax.set_ylabel("multidimensional class")
        ax.set_title(f"({'cd'[j]}) vs {name}: ARI {agree[name]['ari']:.2f},\nsame class {agree[name]['same']:.0%}", loc="left", fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
    footer = ("Three K = 3 AR(1) + \"spike\" fits with one level shift per birth decade common to the classes, all drawn at the 1950s: theta alone on the health contract (38,963 people), "
              "the mental GRM alone on the multidim contract (38,181 people), and the multidimensional fit of theta, the mental GRM and Gompertz-Makeham mortality on the multidim contract. "
              f"Line width is proportional to the class share; the legend gives the share and the class persistence on that channel. (c), (d): rows are the multidimensional modal class, columns the univariate modal class, "
              f"row-normalised, on the {agree['physical']['n']:,} and {agree['mental']['n']:,} people in both fits; classes ordered worst to best on each fit's anchor (the physical channel for the multidimensional fit).")
    fig.text(0.01, -0.03, "\n".join(textwrap.wrap(footer, 230)), fontsize=7.2, color=INK2, va="top")
    fig.savefig(FIG / "fig_multidim_vs_univariate.png", bbox_inches="tight")
    print(f"wrote fig_multidim_vs_univariate.png to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
