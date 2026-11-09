"""Section 5's main figure: returns to prevention, the bill, when the saving arrives, and what to target on.

A 2 x 2 on the K = 3 AR(1) + "spike" classes on theta, the savings exact for the cubic cost curve:
  (a) the saving from prevention at mu = 0.1 by class, S_j(r; 0.1), and uniform, in £000, by start age
      (42_paper_returns_bayes.py; P_j(r), the first-order return, is in the appendix figure);
  (b) the treatment bill still to come C(r), stacked into the cost at the mean path, what the class
      differences add and what realised deviations add, with the exact saving at mu = 1 (42);
  (c) when the saving arrives: the saving at mu = 0.1 in each year of age from prevention started at 25 and
      at 60, by class, the per-age terms whose sum is (a) (42's savings profiles);
  (d) targeting at mu = 0.1: the exact saving per treated person against the share treated, by what the
      prediction uses (age; + education, income, sex; + theta(t-1); + theta(t-1), theta(t-2), the readings of
      the type-prediction figure) and the estimated
      class, on the same classes refitted with the first two rows held out (46_paper_targeting.py --first2,
      the version used throughout the draft; the other rules are in its appendix figure).
The cumulative-share and present-value view of (c), and the share-of-return view of (d), are in the
appendix figures 42 --timing and 46 write.

Output: paper/figures/fig_prevention.png.

    PYTHONPATH=src .venv/bin/python descriptives/47_paper_prevention_main.py
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

sys.path.insert(0, str(Path(__file__).resolve().parent))
if "--first2" not in sys.argv:      # 43/46 read their mode from argv at import: the draft uses the first-two-rows fits
    sys.argv.append("--first2")
from _paper_common import FIG  # noqa: E402
from _style import INK2, apply_style  # noqa: E402

M42 = importlib.import_module("42_paper_returns_bayes")
M46 = importlib.import_module("46_paper_targeting")


def main() -> int:
    apply_style()
    R, cs = M42.compute("theta", "ssm", True)
    MU_A = 0.1                                                     # the intensity of panels (a), (c) and (d)
    res = M46.compute(mu=MU_A)
    if res is None:
        return 1
    D_RULES = [M46.AGE_RULE, M46.OBS_RULE, M46.R1, M46.R12, M46.ORACLE]
    res = {**res, "rules": [r for r in D_RULES if r in res["rules"]]}
    cfun, cprime, _, _, _ = M42.cost_curve("theta", True)
    sv = M42.savings(cs["rows"], "theta", cs["W"], cs["paths"], cs["shares"], cprime, mu=MU_A)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2), gridspec_kw={"hspace": 0.36, "wspace": 0.24})
    M42.draw_saving_mu(axes[0, 0], R, cs, MU_A, "a", scale=1000, ylabel="total saving per person (£000)")
    M42.draw_bill(axes[0, 1], R, cs, "b", exact=True, mean_channel=False, scale=1000, ylabel="£000 per person")
    axes[0, 0].set_xlabel("start age $r$"); axes[0, 1].set_xlabel("start age $r$")
    M42.draw_timing_profiles(axes[1, 0], sv, cs, r_list=(25, 60), letter="c", uniform=False, mu=MU_A)
    M46.draw_targeting(axes[1, 1], None, res, letters="d")
    footer = (f"The K = 3 AR(1) + transitory classes on theta with the cost curve estimated in pounds on theta (a degree-{cs['deg']} polynomial on the cost index, waves 7-15). "
              "Savings are exact: the cost curve is a cubic, so the expansion in mu stops at the cube. "
              "(a) The treatment cost avoided by prevention at mu = 0.1 (a tenth of each class's future decline held back) from start age r, by class, line width proportional to the class share, dashed the uniform policy. "
              "(b) C(r), the expected treatment cost from r to 90 on the observed cross-section, split into the cost at the mean path, what the class differences add and what realised deviations add "
              "(the part prevention cannot reach), with the saving from uniform prevention at mu = 1 (the whole decline held back). "
              "(c) The saving at mu = 0.1 in each year of age a from prevention started at 25 (solid) and 60 (dashed), by class, the per-age terms "
              "whose sum is (a). Pounds to 90, no survival weighting, no discounting. "
              f"(d) Targeting at mu = 0.1, on the same classes refitted with each person's first two rows held out ({res['n']:,} people): each rule predicts a person's saving from what is known at t-1, the later of the two held rows (t is the first fitted row) "
              "(a multinomial logit of the class on a quadratic in age plus the set shown, fitted on a 70% split), treats the top q of the other 30% by predicted saving "
              "and is scored by the realised saving, the fitted-window class posterior times S_j(r; 0.1) at the person's age at t-1. Dashed, targeting on the estimated class, the posterior from the "
              "whole fitted window; dotted, everyone treated. The other rules and the share-of-return view are in the appendix targeting figure.")
    # the footer starts at the panels' left edge (their axis labels included) and just below them, so the tight
    # crop leaves no margin beside the panels
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    box = fig.transFigure.inverted().transform(
        Bbox.union([ax.get_tightbbox(rend) for ax in axes.flat]))
    fig.text(box[0, 0], box[0, 1] - 0.012, "\n".join(textwrap.wrap(footer, 190)), fontsize=7.2, color=INK2, va="top")
    fig.savefig(FIG / "fig_prevention.png", bbox_inches="tight")
    print(f"wrote fig_prevention.png to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
