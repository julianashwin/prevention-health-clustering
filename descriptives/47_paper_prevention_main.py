"""Section 5's main figure: returns to prevention, the bill, when the saving arrives, and what to target on.

A 2 x 2 on the K = 3 AR(1) + "spike" classes on theta:
  (a) the return to prevention by class P_j(r) and the uniform return, by start age (42_paper_returns_bayes.py);
  (b) the treatment bill still to come C(r), stacked into the cost at the mean path, what the class
      differences add and what realised deviations add, with P(r) over it (42);
  (c) when the saving arrives: the saving in each year of age from prevention started at 30, 50 and
      70, uniform and by class, the per-age terms whose sum is P(r) (42's savings profiles);
  (d) targeting: the realised return per treated person against the share treated, by what the
      prediction uses, on the same classes refitted with the first two rows held out
      (46_paper_targeting.py --first2, the version used throughout the draft).
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
    res = M46.compute()
    if res is None:
        return 1
    cfun, cprime, _, _, _ = M42.cost_curve("theta", True)
    sv = M42.savings(cs["rows"], "theta", cs["W"], cs["paths"], cs["shares"], cprime)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2), gridspec_kw={"hspace": 0.36, "wspace": 0.24})
    M42.draw_pj(axes[0, 0], R, cs, "theta", "a")
    M42.draw_bill(axes[0, 1], R, cs, "b")
    axes[0, 0].set_xlabel("start age $r$"); axes[0, 1].set_xlabel("start age $r$")
    M42.draw_timing_profiles(axes[1, 0], sv, cs, letter="c")
    M46.draw_targeting(axes[1, 1], None, res, letters="d")
    footer = (f"The K = 3 AR(1) + \"spike\" classes on theta with the cost curve estimated in pounds on theta (a degree-{cs['deg']} polynomial on the uncapped cost index, waves 7-15). "
              "(a) The return to prevention by class, the treatment cost avoided per unit of mu, line width proportional to the class share, dashed the uniform return. "
              "(b) C(r), the expected treatment cost from r to 90 on the observed cross-section, split into the cost at the mean path, what the class differences add and what realised deviations add "
              "(the part prevention cannot reach), with P(r) and its mean channel. (c) The saving in each year of age a from prevention started at r (solid 30, dashed 50, dotted 70), the per-age terms "
              "whose sum is P(r), uniform in grey and by class. Pounds to 90, no survival weighting, no discounting. "
              f"(d) The same classes refitted with each person's first two rows held out ({res['n']:,} people); each rule predicts a person's return from what is known at the first row "
              "(a multinomial logit of the class on a quadratic in age plus the set shown, fitted on a 70% split; the model's posterior and the oracle need no regression), treats the top q of the other 30% "
              "by predicted return and is scored by the realised return, the fitted-window class posterior times P_j(r) at the person's first-row age; dotted, everyone treated. "
              "The share-of-return view of (d) is in the appendix targeting figure.")
    fig.text(0.01, -0.02, "\n".join(textwrap.wrap(footer, 190)), fontsize=7.2, color=INK2, va="top")
    fig.savefig(FIG / "fig_prevention.png", bbox_inches="tight")
    print(f"wrote fig_prevention.png to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
