# baseline_measures

The baseline physical-health measure comparison: the criteria workbook
(`AnalysisForEIT/exploratory_w_johannes/redo_concepts/baseline_measure_comparison.xlsx`)
and the measures note's section on the SF-12 + limitation banks
(`measuring_health/health_measures_note.tex`, "Physical SF-12 plus limitations").

Scripts 01–06 and 11 were recovered on 15 September 2026 from the session
scratch scripts that built the first workbook; rerun here they reproduce every
earlier cell exactly. Intermediate files go to `data/processed/baseline_measures/`
(gitignored: person-level, derived from licensed UKHLS data). The limitation
banks themselves come from `data_cleaning/archive/07_build_limitation_banks.py`, which
must run first; it also fits the condition banks (each limitation bank plus the chronic-condition count, +CC, or P-FULL's six groups, +CG), and scripts 05–11 carry them through.

Run from the repo root, in order (`PYTHONPATH=src` for the Python scripts):

| Script | Produces |
|---|---|
| `01_baseline_candidates.py` | PCS, phys, the PHYS indices and frailty indices (`baseline_candidates.parquet`) |
| `02_baseline_criteria.py` | the original GRM and P-FULL scores on both scales (`criteria_measures.parquet`) |
| `03_fs_pc_items.R` | item coding shared by 04; sourced, not run |
| `04_fs_pc_build.R` | principal-component and factor-score indices and weights (`fs_pc_measures.parquet`, `fs_pc_weights.csv`) |
| `05_master_criteria.py` | every measure's sample, floor, cost, in-patient and age-profile criteria (`master_measures.parquet`, `master_py.csv`) |
| `06_master_ident.R` | identification under Cases 2, 3 and 4 on balanced and on pairwise moments, and mortality (`master_r.csv`); the scorecard reads Case 3 balanced and Case 2 pairwise |
| `07_limitation_weights.R` | Pearson and polychoric one-factor weights and first components for the limitation banks |
| `08_limitation_moments.R` | Exhibit 1 profiles, and Corr(H, d) by band and the barchart decompositions on h for all six estimators, at the best-fitting and best admissible persistence |
| `09_limitation_clusters.py` | partial K-means, K = 3, for the twelve limitation-bank scores (about 10 minutes, six processes) |
| `10_limitation_note_outputs.py` | the note's figures (`measuring_health/figures/fig_lim_*.png` and, for the condition banks, `fig_con_*.png`), tables (`measuring_health/tables/lim_*.tex` and `con_*.tex`) and `limitation_weights.csv` |
| `11_build_workbook.py` | the workbook, including the Identification sheet, with every scorecard formula checked by a small evaluator and its result cached in the file; needs openpyxl, so run with a Python that has it |
