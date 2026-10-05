# descriptives

Descriptive figures and tables. Scripts import the package and write to
`artifacts/descriptives/`. Shared figure style belongs in
`src/prevention_health_clustering/plotting`, not here.

The measure the project uses is built by `data_cleaning/04_build_health.py`;
`21_health_measure.py` is the descriptive that documents it, and
`15_cost_validation.py`, `16_cost_convexity.py` and `17_cost_age_interaction.py`
document the cost index (`measuring_health/`). The project reports four health
measures only: theta and h, and the 31-deficit frailty index and its log as
robustness measures; the mental GRM is the second channel of the
multidimensional model. The earlier measure generations and the scripts that
read them have been removed.

Paper scripts (`2x`-`4x`, all on the health measure; figures to
`paper/figures/`, table fragments to `paper/tables/`, data to
`artifacts/descriptives/paper_*`), in dependency order. The Bayesian fits they read are produced by the queues in `clustering/runs/` (see `clustering/README.md`); `30_paper_framework_empirics.py` is archived under `archive/`:

| Script | Produces |
|---|---|
| `22_paper_kmeans.py` | partial K-means, K = 3, on theta / h over 20-90, and on sliding and backward-expanding age windows (h, theta); labels, trajectories, composition; ~15 min on 4 processes |
| `23_paper_exhibit1.py` | Exhibit 1: mean, variance and covariance rows, h and theta (`fig_exhibit1.png`) |
| `24_paper_measure_tables.py` | correlations with PCS/MCS/chronic count, mortality by age band, cost gradient by age band (`tab_measure_*.tex`) |
| `25_paper_clusters.py` | Figures 3 and 4: types with within-type bands and autocorrelation; between/within, Var(d) and Cov(H, d) by age |
| `26_paper_windows.py` | Figure 7: types over limited and backward-expanding windows, agreement with the lifecycle typology |
| `27_paper_observables.py` | Figure 8: trajectories by education and income against the types (needs `data_cleaning/06_build_observables.py`) |
| `28_paper_returns.py` | Section 5 first pass: T(r), P_j(r), P(r) and type targeting from the K-means types, or from the mixture classes with `ssm` / `base` on the command line |
| `29_paper_moments.py` | Exhibit 2 and the identification moments behind it, self-contained: four ages two years apart, Cases 2-4, balanced and pairwise, theta / h / fi10 |
| `31_paper_appendix_extras.py` | Appendix B: Exhibit 1 for the cost index, and the returns under the estimated cost curve |
| `36_paper_trajectories.py` | structure's Figure 5: class paths and shares for h and theta under both specifications, with composition strips; deficit-index twin for the appendix |
| `34_paper_cohort.py` | Appendix B.5: the birth-decade cohort fits (`artifacts/health-ssm-cohort/`): class paths with and without cohort shifts, decade profiles raw and net, the shifts with intervals |
| `32_paper_bayes_fits.py` | the K = 3 mixtures on the health contract against the K-means types: paths, composition, person-by-person cross-tab; `base` (`artifacts/health-base/`) or `ssm` (`artifacts/health-ssm/`, Kalman posteriors) on the command line; `combined` draws the paper's two-column figure `fig_bayes_mixtures.png` (theta), with `--h`, `--frailty`, `--logfrailty` (the fits in `artifacts/health-frailty/`, classes relabelled so class 1 is frailest) and `--mental` (three columns, the cohort fit at the 1950s) for the appendix versions |
| `35_kmeans_cohort.py` | Appendix on K-means: cohort adjustment within partial K-means (age and birth-decade dummies, shift then cluster), agreement with the unadjusted types (`fig_kmeans_cohort.png`) |
| `37_paper_cost_mortality_age.py` | Section 3.1: cost and mortality against health by age band, curvature and age-invariance tests (`fig_cost_mortality_age.png`) |
| `38_paper_employment_checks.py` | the data section's employment checks: the jump at 65 per item, and the cost-health relationship for the employed (csv only) |
| `39_paper_multidim.py` | the multidimensional fits (`artifacts/multidim-health/`, from `clustering/runs/multidim_health_queue.py`): physical and mental class paths, the Gompertz-Makeham hazard and composition (`fig_multidim_trajectories.png`); reads the h state-space fit over its three agreeing chains; `--h` the h version, `--cohort` the theta AR(1) + "spike" fit with birth-decade shifts (`fig_multidim_trajectories_cohort.png`, paths and observed means at the 1950s); `--k` the theta AR(1) + "spike" fit at K = 3, 4 and 5, one row each (`fig_multidim_trajectories_by_K.png`, from `clustering/runs/multidim_health_k45_queue.py`) |
| `40_paper_k_comparison.py` | K = 3, 4, 5 compared by plug-in BIC and classification certainty (csv only; not in the paper) |
| `33_paper_prediction.py` | Figure 9 and its table: the measure, the cost proxy and employment one and two waves ahead, on the held-out fits (`artifacts/health-next/health-{v}-{base,ssm}-ho`, `artifacts/multidim-health/{v}-ssm-mort-ho`); also writes the held-out death comparison for 41; reads the fits directly |
| `41_paper_mortality_prediction.py` | the mortality prediction figure: every person-wave at risk on the full-sample fits, and the held-out death-after-last-interview comparison (`fig_prediction_mortality.png`); needs 32 (full-sample posteriors) and 33 (held-out death), and imports the multidim readers from 39 |
| `42_paper_returns_bayes.py` | Section 5: returns to prevention and treatment on the Bayesian classes on theta, in pounds under the uncapped estimated cost curve (`fig_returns_bayes.png`, `tab_returns_bayes.tex`); `--h`, `--kmeans`, `--capped` and `--curve` write the appendix versions and the cost-curve figure; imports `fit_cost_curve` from 31 |

**Main text on theta, other measures in the appendix.** The paper's figures are on theta alone. `23`, `25`, `29` and `37` take `--measures` for the appendix versions on h, the frailty index and log frailty (`*_measures.png`); `26`, `27`, `32 combined`, `36`, `39` and `42` take `--h` for the h version, and `33` and `41` write both in one run.

**Frailty comparison.** `22_paper_kmeans.py`, `23_paper_exhibit1.py`, `25_paper_clusters.py`, `29_paper_moments.py` and `37_paper_cost_mortality_age.py` take `--frailty`, which runs Figures 1-5 on the 31-deficit frailty index and its shifted log, log(frailty + 1/31), (`data_cleaning/04b_build_frailty.py`) instead of h and theta, writing `*_frailty.png` to `paper/figures/` and `*_frailty.csv` to `artifacts/descriptives/` beside the draft's outputs without touching them (`--measures` on 23, 25, 29 and 37 is the appendix version that puts h and the two frailty indices side by side). Order: `22 --frailty` before `25 --frailty`.

**Type from an initial value.** `22_paper_kmeans.py --first-held [1|2] [--variants theta,h]` runs the K-means on theta and h with each person's first one or two rows (by age) dropped, people with at least four or five rows, stored as variants `theta_firstheld`, `h_firstheld`, `theta_first2held`, `h_first2held`; `43_paper_type_from_initial.py` then predicts each typology's type (that K-means, and the modal class of the first-row-held-out mixtures in `artifacts/health-firstheld/`, from `clustering/runs/health_firstheld_queue.py`) and the modal class of the first-row-held-out multidimensional fit in `artifacts/multidim-health/theta-ssm-mort-firstho/`, from `clustering/runs/multidim_firstheld_launch.sh`) from age, the observables and theta at the first row (plus the mental score for the multidimensional types), multinomial logit on a 70/30 split, plus the mixture's own posterior from the single first observation (`fig_type_from_initial.png`, `tab_type_from_initial.tex`, `paper_type_from_initial{,_by_age}.csv`). `43 --h` is the same on h; `43 --first2` reads the first-two-rows fits (`*-first2ho`, including the cohort-shift twin from `clustering/runs/cohort_first2_queue.py`) and adds a predictor set with both held readings; outputs carry `_h`, `_first2`, `_first2_h`. Order: `22 --first-held`, the fits, then `43`.

**Mental health.** The same five scripts take `--mental`, which runs Figures 1-5 on the mental GRM (`theta_ment_nodepr` in `data/processed/measures/mental_grm.parquet`, variant `mental`, higher = better) for the paper's mental-health appendix, writing `*_mental.png` and `*_mental.csv`. The K-means (`22 --mental`, then `25 --mental`) runs on the multidim health contract's rows, the rows the univariate mental Bayesian fits (`clustering/runs/mental_queue.py`) and the multidimensional fits use.

`_paper_common.py` holds the loaders, the path smoother and the Exhibit 1 helpers they share; `_paper_moments.py` the identification panel (one row per person-age, at least four observed ages) and the pooled-moment, case-fit and row-fit machinery ported from the companion pipeline.

The measure and cost-index notes (`measuring_health/`):

| Script | Produces |
|---|---|
| `21_health_measure.py` | the health measure construction note's figures (`measuring_health/figures/fig_health_*.png`) and its properties table |
| `15_cost_validation.py` | `fig_cost_validation.png`: the cost index against the utilisation it is built from |
| `16_cost_convexity.py` | `fig_cost_convexity.png`: convexity of cost in pounds on the four measures and under every costing assumption (`cost_convexity.csv`) |
| `17_cost_age_interaction.py` | `fig_cost_age_interaction.png`: the cost gradient within age bands |

Figures live under `measuring_health/figures/`. They are aggregate
statistics and safe to commit; everything person-level stays under `data/`
(gitignored).
