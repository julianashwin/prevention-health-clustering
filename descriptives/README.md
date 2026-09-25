# descriptives

Descriptive figures and tables. Scripts import the package and write to
`artifacts/descriptives/`. Shared figure style belongs in
`src/prevention_health_clustering/plotting`, not here.

The measure the project uses is built by `data_cleaning/04_build_health.py`;
`21_health_measure.py` is the descriptive that documents it. Scripts numbered
01-20 were written against the earlier measure generations and read
`measure_panel.parquet`, `grm_scores.parquet` or `grm2_scores.parquet`, which
`data_cleaning/archive/` still builds.

Paper scripts (`2x_paper_*.py`, all on the health measure; figures to
`paper/figures/`, table fragments to `paper/tables/`, data to
`artifacts/descriptives/paper_*`), in dependency order:

| Script | Produces |
|---|---|
| `22_paper_kmeans.py` | partial K-means, K = 3, on theta / h / fi10 over 20-90, and on sliding and backward-expanding age windows (h, theta); labels, trajectories, composition; ~15 min on 4 processes |
| `23_paper_exhibit1.py` | Exhibit 1: mean, variance and covariance rows, h and theta (`fig_exhibit1.png`) |
| `24_paper_measure_tables.py` | correlations with PCS/MCS/chronic count, mortality by age band, cost gradient by age band (`tab_measure_*.tex`) |
| `25_paper_clusters.py` | Figures 3 and 4: types with within-type bands and autocorrelation; between/within, Var(d) and Cov(H, d) by age |
| `26_paper_windows.py` | Figure 7: types over limited and backward-expanding windows, agreement with the lifecycle typology |
| `27_paper_observables.py` | Figure 8: trajectories by education and income against the types (needs `data_cleaning/06_build_observables.py`) |
| `28_paper_returns.py` | Section 5 first pass: T(r), P_j(r), P(r) and type targeting from the K-means types, or from the mixture classes with `ssm` / `base` on the command line |
| `29_paper_moments.py` | Exhibit 2 and the identification moments behind it, self-contained: four ages two years apart, Cases 2-4, balanced and pairwise, theta / h / fi10 |
| `30_paper_framework_empirics.py` | the framework note's empirical figures on the paper's measure: rows, surface, local triple, Case 2 windows, convexity, the bundle-fit C_Hd(a) with a person bootstrap (`BOOT`, default 200; ~5 min) |
| `31_paper_appendix_extras.py` | Appendix B: Exhibit 1 for the cost index, and the returns under the estimated cost curve |
| `36_paper_trajectories.py` | structure's Figure 5: class paths and shares for h and theta under both specifications, with composition strips; deficit-index twin for the appendix |
| `34_paper_cohort.py` | Appendix B.5: the birth-decade cohort fits (`artifacts/health-ssm-cohort/`): class paths with and without cohort shifts, decade profiles raw and net, the shifts with intervals |
| `33_paper_prediction.py` | Figure 9: next-wave h, death, in-patient stay and GP band from what is known at t; observables, types, class posteriors, the measure and its lag |
| `32_paper_bayes_fits.py` | the K = 3 mixtures on the health contract against the K-means types: paths, composition, person-by-person cross-tab; `base` (`artifacts/health-base/`) or `ssm` (`artifacts/health-ssm/`, Kalman posteriors) on the command line |

`_paper_common.py` holds the loaders, the path smoother and the Exhibit 1 helpers they share; `_paper_moments.py` the identification panel (one row per person-age, at least four observed ages) and the pooled-moment, case-fit and row-fit machinery ported from the companion pipeline.

Earlier scripts, in dependency order:

| Script | Produces |
|---|---|
| `21_health_measure.py` | the health measure construction note's three figures (`measuring_health/figures/fig_health_*.png`) + `artifacts/descriptives/health_measure_properties.csv` |
| `baseline_measures/` | the archived measure search: 85 candidates against every criterion, its own README |
| `01_measure_landscape.py` | `data/processed/measures/measure_panel.parquet` + correlation/age-mean CSVs under `artifacts/descriptives/` |
| `02_cluster_figure.py` | partial K-means trajectory panels (`docs/figures/fig_cluster_trajectories.png`) + agreement table |
| `03_moments_grid.py` | mean/variance/covariance-rows grid (`docs/figures/fig_moments_grid.png`) |
| `04_ever_sensitivity.py` | the ever-diagnosis accumulation checks (`docs/figures/fig_ever_sensitivity.png`) |
| `05_grm_versions_figure.py` | the measure-choice panel across GRM versions (`docs/figures/fig_grm_versions.png`) + criteria CSV |
| `06_information_ceiling.py` | item-information decomposition per GRM (`docs/figures/fig_grm_information.png`) + ceiling/floor and floor-depth tables |
| `14_expected_cost.py` | the two margins of in-patient cost and whether convexity survives logs (`docs/figures/fig_expected_cost.png`) |
| `13_convexity_detail.py` | how much of the healthcare convexity survives the axis and link choices (`docs/figures/fig_convexity_detail.png`) |
| `12_outcome_prediction.py` | mortality and utilisation prediction from each model's classes, in and out of sample (`docs/figures/fig_outcome_prediction.png`) |
| `11_class_composition.py` | model class composition by age for all twelve fits (`artifacts/descriptives/class_composition_by_age.csv`) |
| `10_multidim_trajectories.py` | class trajectories and shares for the four multidimensional fits, all three channels (`docs/figures/fig_multidim_trajectories.png`) |
| `09_combined_grm_model.py` | the combined GRM drawn as hurdles on the health axis (`docs/figures/fig_combined_grm_model.png`) |
| `08_trajectory_comparison.py` | fitted class trajectories and shares across the eight fits (`docs/figures/fig_trajectory_comparison.png`) + spread table |
| `07_convexity.py` | convexity of in-/out-patient use and GP contact in each metric (`docs/figures/fig_convexity.png`) + tests CSV |

Figures live under `measuring_health/figures/` (the table above still spells
some of them `docs/figures/`, the folder's former name). They are aggregate
statistics and safe to commit; everything person-level stays under `data/`
(gitignored).
