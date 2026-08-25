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
| `06_information_ceiling.py` | item-information decomposition per GRM (`docs/figures/fig_grm_information.png`) + ceiling/floor table |

Figures under `docs/figures/` are aggregate statistics and safe to commit;
everything person-level stays under `data/` (gitignored).
