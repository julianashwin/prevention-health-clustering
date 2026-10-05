# paper

The paper itself. `paper_structure.txt` is the outline; `paper_skeleton.tex`
is the bullet-point draft that fleshes it out, with every figure and table
produced by a script in this repository (`descriptives/22`–`43_paper_*.py`;
the script-by-script map is `descriptives/README.md`). Red `\todo` marks in
the PDF are open decisions or analyses not yet run; grey `\note` marks are
drafting notes. The source keeps each paragraph and bullet on one line, for
editors that wrap. Open items, running fits and the feedback list are tracked
in [`../STATUS.md`](../STATUS.md); `feedback_29sept.txt` is the feedback of
29 September verbatim.

Build (natbib, latexmk):

```bash
cd paper && latexmk -pdf paper_skeleton.tex
```

- `figures/` — figures made for the paper (aggregate statistics, safe to
  commit). The main-text figures are on theta; the `_h`, `_measures`,
  `_frailty`, `_mental`, `_cohort`, `_by_K`, `_first2` suffixes are the
  appendix and robustness versions the same scripts write with their flags.
  `figures/external/` holds copies of the figures taken from
  `conceptual_framework/figures` and `measuring_health/figures`, so this folder
  compiles on its own; refresh the copies if those figures are regenerated.
- `tables/` — `.tex` fragments written by the same scripts and `\input` by the
  skeleton.
- `references.bib` — natbib bibliography.
- `paper_overleaf/`, `paper_overleaf.zip` — an 18 September copy for Overleaf;
  stale, regenerate before sharing.

The introduction is not started.
