"""The model behind the combined metric: hurdles on the health axis.

The empirics note's Figure 31 drawn for the most all-encompassing bank -- the
seventeen-item COMBINED GRM. The graded response model says a person has one
number, theta, and that given theta every item answer is an independent draw.
Each item is a set of HURDLES on the theta axis: to answer at least at level
k you must clear hurdle b_jk, with probability logistic in how far above it
you sit and steepness a_j. So b says WHERE on the health axis an answer
flips, and a says HOW SHARPLY -- which is the item's weight, because a
sharper flip pins theta down more tightly.

Four panels:
  (a) every hurdle of every item on one axis, against the population
      distribution of theta -- where each item actually bites
  (b) the hurdle curves themselves for items spanning the weight range
  (c) how one person's answers combine into a posterior
  (d) two people with the SAME summed score but different answers

Outputs: docs/figures/fig_combined_grm_model.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from _style import INK2, apply_style  # noqa: E402

from prevention_health_clustering.config import PROCESSED_DATA_DIR, ROOT_DIR
from prevention_health_clustering.measures.grm import cat_probs
from prevention_health_clustering.measures.grm2 import COMBINED_NCAT

TH = np.linspace(-4.0, 3.0, 701)
FAMILY = {
    "GH": "physical", "PF": "physical", "RP": "physical", "BP": "physical",
    "FUNC": "physical", "CVD": "diagnosis", "METAB": "diagnosis",
    "RESP": "diagnosis", "MSK": "diagnosis", "CANCER": "diagnosis",
    "OTHER": "diagnosis", "MH": "mental", "RE": "mental", "SF": "mental",
    "GHQPOS": "mental", "GHQNEG": "mental", "DEPR": "diagnosis",
}
FCOL = {"physical": "#12395B", "mental": "#CC79A7", "diagnosis": "#CC5500"}
SHOWCASE = ["GHQNEG", "PF", "FUNC", "GH", "CVD", "RESP"]


def load_items():
    df = pd.read_csv(PROCESSED_DATA_DIR / "measures" / "grm2_items.csv")
    df = df[df["spec"] == "COMBINED"]
    out = {}
    for _, r in df.iterrows():
        b = r[[c for c in df.columns if c.startswith("b")]].dropna().to_numpy(float)
        out[r["item"]] = (float(r["a"]), b)
    return {k: out[k] for k in COMBINED_NCAT if k in out}


def loglik(items, pattern):
    return {v: np.log(cat_probs(a, b, TH)[:, pattern[v] - 1])
            for v, (a, b) in items.items()}


def posterior(items, pattern):
    lw = sum(loglik(items, pattern).values()) + norm.logpdf(TH)
    w = np.exp(lw - lw.max())
    return w / w.sum()


def main() -> int:
    apply_style()
    items = load_items()
    scores = pd.read_parquet(
        PROCESSED_DATA_DIR / "measures" / "grm2_scores.parquet",
        columns=["theta_combined"])["theta_combined"].dropna()

    fig = plt.figure(figsize=(13.0, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], hspace=0.42,
                          wspace=0.22)

    # ---- (a) every hurdle, against the population -------------------------
    ax = fig.add_subplot(gs[:, 0])
    order = sorted(items, key=lambda v: items[v][0])
    LEFT = -4.2
    for i, v in enumerate(order):
        a, b = items[v]
        vis, off = b[b >= LEFT], b[b < LEFT]
        ax.plot(vis, np.full(len(vis), i), "o", ms=3 + 2.2 * a / 3.5,
                color=FCOL[FAMILY[v]], alpha=0.85, zorder=3)
        if len(off):  # hurdles beyond the axis, marked at the edge
            ax.plot([LEFT + 0.06], [i], "<", ms=4, color=FCOL[FAMILY[v]],
                    alpha=0.9, zorder=3)
            ax.text(LEFT + 0.16, i, f"{off.min():.1f}", fontsize=5.4,
                    va="center", color=FCOL[FAMILY[v]])
        lo = max(b.min(), LEFT)
        ax.plot([lo, b.max()], [i, i], lw=0.7, color="#cccccc", zorder=1)
        ax.text(3.05, i, f"a={a:.2f}", fontsize=6, va="center", color=INK2)
    ax.set_yticks(range(len(order)), order, fontsize=7)
    ax.set_xlim(-4.2, 3.5)
    ax.set_xlabel(r"latent health $\theta$")
    dens = np.histogram(scores, bins=80, range=(-4, 3), density=True)[0]
    centres = np.linspace(-4, 3, 80)
    ax.fill_between(centres, -1.4, -1.4 + 5.5 * dens, color="#8FD1C0",
                    alpha=0.55, zorder=0)
    ax.text(-4.05, -1.1, "population", fontsize=6.5, color="#2b7a68")
    ax.set_ylim(-1.6, len(order) - 0.3)
    ax.set_title("(a) Where each item bites: every hurdle of all 17 items",
                 fontsize=10)
    for fam, col in FCOL.items():
        ax.plot([], [], "o", color=col, label=fam, ms=5)
    ax.legend(fontsize=7, loc="lower right", ncols=3)

    # ---- (b) the hurdle curves for a spread of weights --------------------
    ax = fig.add_subplot(gs[0, 1])
    for v in SHOWCASE:
        a, b = items[v]
        for k, bk in enumerate(b):
            ax.plot(TH, 1 / (1 + np.exp(-a * (TH - bk))),
                    lw=1.5, color=FCOL[FAMILY[v]],
                    alpha=0.85 if k == 0 else 0.35,
                    label=f"{v} (a={a:.2f})" if k == 0 else None)
    ax.axhline(0.5, ls=":", color="#bbbbbb", lw=0.8)
    ax.set_xlim(-4, 3)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("P(answer at least this good)")
    ax.set_title("(b) Steepness is the weight", fontsize=10)
    ax.legend(fontsize=6.4, loc="upper left")

    # ---- (c) and (d) two people with the same summed score ----------------
    # Two people whose category numbers sum to the SAME total, one carrying
    # the deficit on the physical items, the other on the mental ones. Built
    # by subtracting the same number of steps (21) from the top pattern.
    best = {v: COMBINED_NCAT[v] for v in items}
    pA = dict(best)
    pA.update({"GH": 2, "PF": 2, "RP": 3, "BP": 2, "FUNC": 1,
               "CVD": 1, "MSK": 1})               # -3-3-6-3-3-2-1 = -21
    pB = dict(best)
    pB.update({"MH": 2, "RE": 3, "SF": 2, "GHQPOS": 18, "GHQNEG": 16,
               "DEPR": 1})                        # -7-6-3-1-3-1  = -21
    sumA, sumB = sum(pA.values()), sum(pB.values())
    assert sumA == sumB, (sumA, sumB)
    for pat in (pA, pB):
        for v, k in pat.items():
            assert 1 <= k <= COMBINED_NCAT[v], (v, k)

    ax = fig.add_subplot(gs[1, 1])
    for pat, col, name in ((pA, "#12395B", "A: physical burden"),
                           (pB, "#CC79A7", "B: mental burden")):
        w = posterior(items, pat)
        eap = float((w * TH).sum())
        sd = float(np.sqrt((w * (TH - eap) ** 2).sum()))
        ax.plot(TH, w / w.max(), lw=2.6, color=col,
                label=f"{name}: $\\theta$={eap:+.2f} (sd {sd:.2f})")
        ax.axvline(eap, ls=":", lw=1.0, color=col)
    ax.set_xlim(-4, 3)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("posterior, scaled")
    ax.set_title(f"(c) Same summed score ({sumA}), different answers",
                 fontsize=10)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9,
              frameon=True, edgecolor="none")

    fig.suptitle("The model behind the combined metric: hurdles on the health axis",
                 fontweight="bold", y=0.99)
    fig.text(0.005, -0.005,
             "Each item is a set of hurdles on the health axis; clearing one is logistic in how far above it you sit,\n"
             "with steepness a. Marginal answer frequencies fix where the hurdles are; the associations between\n"
             "items fix how steep they are. Point size in (a) is proportional to a, and arrows mark hurdles beyond\n"
             "the axis, which are weakly identified because so few people sit that low.",
             fontsize=7.4, color=INK2, va="top")
    fig.tight_layout()
    out = ROOT_DIR / "docs" / "figures" / "fig_combined_grm_model.png"
    fig.savefig(out)

    print("COMBINED bank, items ordered by weight:")
    for v in order[::-1]:
        a, b = items[v]
        print(f"  {v:7s} ({FAMILY[v]:9s}) a={a:5.2f}  "
              f"{len(b):2d} hurdles from {b.min():6.2f} to {b.max():6.2f}")
    for pat, name in ((pA, "A"), (pB, "B")):
        w = posterior(items, pat)
        eap = float((w * TH).sum())
        print(f"  person {name}: summed score {sum(pat.values())}, "
              f"theta {eap:+.3f}, posterior sd "
              f"{np.sqrt((w*(TH-eap)**2).sum()):.3f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
