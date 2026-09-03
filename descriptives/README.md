# descriptives

Descriptive figures and tables. Scripts import the package and write to
`artifacts/descriptives/`. Shared figure style belongs in
`src/prevention_health_clustering/plotting`, not here.

Current scripts, in dependency order:

| Script | Produces |
|---|---|
| `01_measure_landscape.py` | `data/processed/measures/measure_panel.parquet` + correlation/age-mean CSVs under `artifacts/descriptives/` |
| `02_cluster_figure.py` | partial K-means trajectory panels (`docs/figures/fig_cluster_trajectories.png`) + agreement table |
| `03_moments_grid.py` | mean/variance/covariance-rows grid (`docs/figures/fig_moments_grid.png`) |
| `04_ever_sensitivity.py` | the ever-diagnosis accumulation checks (`docs/figures/fig_ever_sensitivity.png`) |
| `05_grm_versions_figure.py` | the measure-choice panel across GRM versions (`docs/figures/fig_grm_versions.png`) + criteria CSV |
| `06_information_ceiling.py` | item-information decomposition per GRM (`docs/figures/fig_grm_information.png`) + ceiling/floor and floor-depth tables |
| `13_convexity_detail.py` | how much of the healthcare convexity survives the axis and link choices (`docs/figures/fig_convexity_detail.png`) |
| `12_outcome_prediction.py` | mortality and utilisation prediction from each model's classes, in and out of sample (`docs/figures/fig_outcome_prediction.png`) |
| `11_class_composition.py` | model class composition by age for all twelve fits (`artifacts/descriptives/class_composition_by_age.csv`) |
| `10_multidim_trajectories.py` | class trajectories and shares for the four multidimensional fits, all three channels (`docs/figures/fig_multidim_trajectories.png`) |
| `09_combined_grm_model.py` | the combined GRM drawn as hurdles on the health axis (`docs/figures/fig_combined_grm_model.png`) |
| `08_trajectory_comparison.py` | fitted class trajectories and shares across the eight fits (`docs/figures/fig_trajectory_comparison.png`) + spread table |
| `07_convexity.py` | convexity of in-/out-patient use and GP contact in each metric (`docs/figures/fig_convexity.png`) + tests CSV |

Figures under `docs/figures/` are aggregate statistics and safe to commit;
everything person-level stays under `data/` (gitignored).
