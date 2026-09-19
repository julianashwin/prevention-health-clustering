# Conceptual framework

The conceptual-framework section of the prevention/treatment paper, packaged
as a self-contained folder: the note, its figures, and the code that generates
the simulated figures and verifies the results.

## Contents

- `conceptual_framework.tex` - the note (compile with `pdflatex`, twice for
  cross-references). Line numbers in the margin are a review device; comment
  out the marked `lineno` block in the preamble to remove them.
- `make_framework_figures.R` - generates every SIMULATED figure into
  `figures/` (the two-worlds chart, the Case 1-4 anatomy charts, the K(r)
  blocks, and the appendix anatomy/surface charts). Deterministic; run from
  this folder.
- `check_framework.R` - numerical verification of every result the note flags
  as "verified numerically": the case reads (Cases 1-4 and the five-wave
  general member, including exact recovery on population moments), the
  targeting results (Result 6 in linear and general form, the life-course
  channels, the local-reading identities), and the growth-regression
  representation (congruence under three noise processes; exact level-route /
  coefficient-route equivalence). Prints true-vs-estimated per block.
- `figures/` - all figures the note includes.

## Empirical figures

The eight `emp_fig_*.pdf` files are produced by the UKHLS data pipeline,
which is transferred separately with the data work; they are included here as
static files so the note compiles. Their producers (the moments pipeline and
`emp_27_v6_illustrations.R`, plus the estimator suite referenced in the
appendix on the local covariance) live with the empirical companion repo,
as does the `testing_metrics.tex` companion the note cites.

## Requirements

R with the MASS package (checks and one figure); pdflatex (TeX Live) with
standard packages (amsmath, booktabs, enumitem, xcolor, graphicx, float,
comment, hyperref, array, lineno).
