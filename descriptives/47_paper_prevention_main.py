"""Section 5's main figure: returns to prevention and what to target on, in a 2 x 2.

Left column, the K = 3 AR(1) + "spike" classes on theta (42_paper_returns_bayes.py):
  (a) the return to prevention by class P_j(r) and the uniform return, by start age;
  (c) the treatment bill still to come C(r), stacked into the cost at the mean path, what
      the class differences add and what realised deviations add, with P(r) over it.
Right column, the targeting exercise on the same classes refitted with the first row held
out (46_paper_targeting.py):
  (b) the realised return per treated person against the share treated, by rule;
  (d) the share of the total return collected.

With --with-timing a third row is added, when the savings arrive (42's savings profiles):
  (e) the saving in each year of age from prevention started at 30, 50 and 70, uniform and by class;
  (f) the share of the return collected by age, with the half-way ages and the present value at 3.5%;
and the figure is written as fig_prevention_timing_combined.png instead.

Output: paper/figures/fig_prevention.png (or fig_prevention_timing_combined.png).

    PYTHONPATH=src .venv/bin/python descriptives/47_paper_prevention_main.py [--with-timing]
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import FIG  # noqa: E402
from _style import INK2, apply_style  # noqa: E402

M42 = importlib.import_module("42_paper_returns_bayes")
M46 = importlib.import_module("46_paper_targeting")


def main() -> int:
    apply_style()
    R, cs = M42.compute("theta", "ssm", True)
    res = M46.compute()
    if res is None:
        return 1
    timing = "--with-timing" in sys.argv
    nrows = 3 if timing else 2
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 4.6 * nrows), gridspec_kw={"hspace": 0.36, "wspace": 0.24})
    M42.draw_pj(axes[0, 0], R, cs, "theta", "a")
    M42.draw_bill(axes[1, 0], R, cs, "c")
    axes[1, 0].set_xlabel("start age $r$")
    M46.draw_targeting(axes[0, 1], axes[1, 1], res, letters="bd")
    if timing:
        cfun, cprime, _, _, _ = M42.cost_curve("theta", True)
        sv = M42.savings(cs["rows"], "theta", cs["W"], cs["paths"], cs["shares"], cprime)
        M42.draw_timing_profiles(axes[2, 0], sv, cs, letter="e")
        M42.draw_timing_cumulative(axes[2, 1], sv, cs, letter="f")
    footer = (f"Left: the K = 3 AR(1) + \"spike\" classes on theta with the cost curve estimated in pounds on theta (a degree-{cs['deg']} polynomial on the uncapped cost index, waves 7-15). "
              "(a) The return to prevention by class, the treatment cost avoided per unit of mu, line width proportional to the class share, dashed the uniform return. "
              "(c) C(r), the expected treatment cost from r to 90 on the observed cross-section, split into the cost at the mean path, what the class differences add and what realised deviations add "
              "(the part prevention cannot reach), with P(r) and its mean channel. Pounds to 90, no survival weighting, no discounting. "
              f"Right: the same classes refitted with each person's first row held out ({res['n']:,} people); each rule predicts a person's return from what is known at that row "
              "(a multinomial logit of the class on a quadratic in age plus the set shown, fitted on a 70% split; the model's posterior and the oracle need no regression), treats the top q of the other 30% "
              "by predicted return and is scored by the realised return, the fitted-window class posterior times P_j(r) at the person's first-row age. Dotted in (b): everyone treated."
              + (" Bottom: when the saving arrives. (e) The saving in each year of age a from prevention started at r (solid 30, dashed 50, dotted 70), the per-age terms whose sum is P(r), "
                 "uniform in grey and by class; (f) the share of the return collected by age, with the age by which half has arrived and the present value at 3.5% a year as a share of the undiscounted return." if timing else ""))
    fig.text(0.01, -0.02, "\n".join(textwrap.wrap(footer, 190)), fontsize=7.2, color=INK2, va="top")
    out = "fig_prevention_timing_combined.png" if timing else "fig_prevention.png"
    fig.savefig(FIG / out, bbox_inches="tight")
    print(f"wrote {out} to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
