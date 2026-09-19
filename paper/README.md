# paper

The paper itself. `paper_structure.txt` is the outline; `paper_skeleton.tex`
is the bullet-point first draft that fleshes it out, with every figure and
table produced by a script in this repository. Red `TODO` marks in the PDF are
open decisions or analyses not yet run; grey notes are drafting notes. The
source keeps each paragraph and bullet on one line, for editors that wrap.

```bash
cd paper && pdflatex paper_skeleton && bibtex paper_skeleton && pdflatex paper_skeleton && pdflatex paper_skeleton
```

- `figures/` — figures made for the paper by `descriptives/22–33_paper_*.py`
  (aggregate statistics, safe to commit). `figures/external/` holds copies of
  the figures taken from `conceptual_framework/figures` and
  `measuring_health/figures`, so this folder compiles on its own; refresh the
  copies if those figures are regenerated.
- `tables/` — `.tex` fragments written by the same scripts and `\input` by the
  skeleton.
- `references.bib` — natbib bibliography.

Placeholders, deliberately: the Bayesian mixture results are on the archived
P-FULL measure until refitted on the paper's measure; Exhibit 2 is paused;
the introduction is not started.
