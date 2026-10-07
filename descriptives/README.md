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
| `26_paper_windows.py` | Figure: partial K-means types refitted on sliding twenty-year windows and on windows expanding backwards from 90, against the lifecycle typology, with the agreement (ARI) and, in the lower row, the worst type's share of the person-waves observed at each age per window and of each window's people (`fig_windows.png`; `--h` the appendix version; `paper_windows_agreement.csv`) |
| `27_paper_observables.py` | Figure 8: trajectories by education and income against the types (needs `data_cleaning/06_build_observables.py`) |
| `29_paper_moments.py` | Exhibit 2 and the identification moments behind it, self-contained: four ages two years apart, Cases 2-4, balanced and pairwise, theta / h / fi10 |
| `31_paper_appendix_extras.py` | Appendix B: Exhibit 1 for the cost index (`fig_exhibit1_cost.png`); also `fit_cost_curve`, the polynomial cost curve on the measure that `42` prices the returns with |
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
| `42_paper_returns_bayes.py` | the treatment bill still to come C(r) (cost at the mean path + type differences + realised deviations) and the return to prevention P_j(r) as treatment cost avoided per unit of mu, for any typology (`class_set`: kmeans, base, ssm, ssm-cohort, multidim, multidim-cohort) on any measure, with the cost curve estimated in pounds. Default: theta, ssm and base side by side (`fig_returns_bayes.png`, appendix; `tab_returns_bayes.tex`, main text); `--h`, `--kmeans`, `--capped`; `--all` panel (a) for six typologies (`fig_returns_by_model.png`); `--measures` panel (a) under AR(1) + "spike" on theta, h, frailty and log frailty (`fig_returns_by_measure.png`); `--curve` the cost-curve figure. Its `draw_*` panels are reused by `47` |

**Main text on theta, other measures in the appendix.** The paper's figures are on theta alone. `23`, `25`, `29` and `37` take `--measures` for the appendix versions on h, the frailty index and log frailty (`*_measures.png`); `26`, `27`, `32 combined`, `36`, `39` and `42` take `--h` for the h version, and `33` and `41` write both in one run.

**Frailty comparison.** `22_paper_kmeans.py`, `23_paper_exhibit1.py`, `25_paper_clusters.py`, `29_paper_moments.py` and `37_paper_cost_mortality_age.py` take `--frailty`, which runs Figures 1-5 on the 31-deficit frailty index and its shifted log, log(frailty + 1/31), (`data_cleaning/04b_build_frailty.py`) instead of h and theta, writing `*_frailty.png` to `paper/figures/` and `*_frailty.csv` to `artifacts/descriptives/` beside the draft's outputs without touching them (`--measures` on 23, 25, 29 and 37 is the appendix version that puts h and the two frailty indices side by side). Order: `22 --frailty` before `25 --frailty`.

**Type from an initial value.** `22_paper_kmeans.py --first-held [1|2] [--variants theta,h]` runs the K-means on theta and h with each person's first one or two rows (by age) dropped, people with at least four or five rows, stored as variants `theta_firstheld`, `h_firstheld`, `theta_first2held`, `h_first2held`; `43_paper_type_from_initial.py` then predicts each typology's type (that K-means, and the modal class of the first-row-held-out mixtures in `artifacts/health-firstheld/`, from `clustering/runs/health_firstheld_queue.py`) and the modal class of the first-row-held-out multidimensional fit in `artifacts/multidim-health/theta-ssm-mort-firstho/`, from `clustering/runs/multidim_firstheld_launch.sh`) from age, the observables and theta at the first row (plus the mental score for the multidimensional types), multinomial logit on a 70/30 split, plus the mixture's own posterior from the single first observation (`fig_type_from_initial.png`, `tab_type_from_initial.tex`, `paper_type_from_initial{,_by_age}.csv`). `43 --h` is the same on h; `43 --first2` reads the first-two-rows fits (`*-first2ho`, including the cohort-shift twin from `clustering/runs/cohort_first2_queue.py`), adds a predictor set with both held readings and the mixture's posterior from both held rows, and uses the first-row multidimensional fit as a labelled placeholder until `theta-ssm-mort-first2ho` lands (`clustering/runs/multidim_first2_launch.sh`); outputs carry `_h`, `_first2`, `_first2_h`. **The draft uses the `--first2` versions throughout** (7 Oct): `fig:typeinit` is `fig_type_from_initial_first2.png`, the h version is in the measures appendix. Order: `22 --first-held`, the fits, then `43`.

**Prevention in years: life expectancy and inequality.** `48_paper_prevention_le.py` prices the same instrument in years on the multidimensional K = 3 AR(1) + "spike" classes: `44`'s calibrated class hazards and healthy shares (exposed by `44.baseline()`), the class paths shifted by mu times the mean change after r - 1, mortality moved through the within-age logit gradient of death in theta (as in `37`) and the healthy share through the same design on the healthy indicator. Scenarios: targeted (worst class at mu = 1), uniform with the same budget (mu = pi_1(r)), uniform at mu = 1. Outputs LE and HLE by class, the class gaps, the sd of age at death, and the education / income-third / region gradients through class composition (`fig_prevention_le.png`, `tab_prevention_le.tex`, `paper_prevention_le{,_groups}.csv`). Run `44` first (it is imported, not re-run).

**The prevention section's main figure.** `47_paper_prevention_main.py` composes `fig_prevention.png`, the 2 x 2 of Section 5 on the AR(1) + "spike" theta classes: (a) the return by class and (b) the bill and what prevention avoids (`42`), (c) when the saving arrives, the per-age terms behind P(r) by start age (`42 --timing` draws the detailed by-class version with the cumulative shares and present values, `fig_prevention_timing.png`, for the appendix), and (d) the targeting rules (`46 --first2`; the script sets the flag itself). Run `42` and `46 --first2` first.

**Targeting.** `46_paper_targeting.py` prices targeting rules for prevention on what is known at a person's first observation (age; education, income and sex; one reading of the measure; both; the mixture's own posterior; the oracle that knows the fitted-window class), using the class returns of `42` at the person's first-row age and the first-rows-held-out classes of `43`: treat the top share q by predicted return, score by the realised return (`fig_targeting{,_h,_first2,_first2_h}.png`, `tab_targeting*.tex`, `paper_targeting*.csv`). With `--first2` the rules also include two readings and the mixture's posterior from both held rows; the draft uses `--first2` throughout (`fig:targeting`, `tab:targeting`, panel (d) of `fig:prevention`, and `--first2 --h` in the measures appendix). Order: `42`, the first-held fits, then `46`.

**Life expectancy by class.** `44_paper_life_expectancy.py` turns the class-specific Gompertz-Makeham hazards of the multidimensional K = 3 AR(1) + "spike" fit (`artifacts/multidim-health/theta-ssm-mort/`) into remaining life expectancy and healthy life expectancy by class. Every class hazard is scaled by one common age-varying factor, calibrated so that the composition-weighted population LE at 65 and at 90 matches the ONS UK 2020-22 national life tables (the correction for deaths the panel does not record; the ONS values are hard-coded in the script); healthy = self-rated general health good or better, posterior-weighted by class and age from the item cache (`fig_life_expectancy.png`, `tab_life_expectancy.tex`, `paper_life_expectancy.csv`).

**Multidimensional against univariate.** `45_paper_multidim_vs_univariate.py` sets the three K = 3 AR(1) + "spike" cohort fits side by side (theta alone, the mental GRM alone, the multidimensional fit of both plus mortality), all at the 1950s: each channel's class paths from the univariate and the multidimensional fit, and the cross-tabs of the multidimensional modal class against each univariate one with ARI and same-class share (`fig_multidim_vs_univariate.png`, `paper_multidim_vs_univariate.csv`).

**Mental health.** The same five scripts take `--mental`, which runs Figures 1-5 on the mental GRM (`theta_ment_nodepr` in `data/processed/measures/mental_grm.parquet`, variant `mental`, higher = better) for the paper's mental-health appendix, writing `*_mental.png` and `*_mental.csv`. The K-means (`22 --mental`, then `25 --mental`) runs on the multidim health contract's rows, the rows the univariate mental Bayesian fits (`clustering/runs/mental_queue.py`) and the multidimensional fits use.

`_style.py` holds the shared figure style, `share_ylim` (one vertical scale across panels that show the same quantity) and a footer guard installed by `apply_style()`: before any figure is saved, a footer wider than the figure is re-flowed to fit, so notes never stick out past the panels. `_paper_common.py` holds the loaders, the path smoother and the Exhibit 1 helpers they share; `_paper_moments.py` the identification panel (one row per person-age, at least four observed ages) and the pooled-moment, case-fit and row-fit machinery ported from the companion pipeline.

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
