# prevention-health-clustering

Health trajectory analysis for Understanding Society (UKHLS): data
construction, health measures, descriptives, Bayesian latent-class models,
and (planned) policy analysis.

A focused rebuild of the analysis core from `ukhls-health-clusters`, carrying
only what is verified, used, and understood. The predecessor repository remains
the archive of record for published evidence and its audit trail.

## Layout

Workstream folders hold thin scripts; everything reusable lives in one
installable package they all import.

```
├── data_cleaning/   raw extract -> panel -> frozen contracts -> measures
├── descriptives/    figures and tables            -> artifacts/descriptives/
├── clustering/      model fits, validation, benchmarks -> artifacts/
├── policy/          counterfactual analysis (planned)
├── src/prevention_health_clustering/
│   ├── data/        UKHLS ingest and cleaning
│   ├── measures/    SF-12 rebuild, UK variants, phys-only, SF-6D, composites
│   ├── contracts/   frozen sample contracts
│   ├── models/      Stan programs + the model registry
│   ├── runner/      payload building, fitting, initialisation
│   ├── scoring/     folds, held-out scoring, leakage guards
│   └── plotting/    shared figure style [next]
├── tests/           fast unit tests of the package
├── data/            gitignored: licensed inputs and derived data
└── artifacts/       gitignored: fits, figures, benchmark outputs
```

Rule of thumb: code lives in its workstream folder until a second stream needs
it; then it moves into the package.

## Validation scoreboard

| Component | Check | Result |
|---|---|---|
| Panel + contracts | vs frozen roster | 50,194 persons / 446,966 rows exact; PCS mean bit-identical |
| Fold manifest | vs frozen ID | `1797b5c5a937d7f5671d` reproduced exactly |
| PCS model | full roster vs published | max 0.05 posterior SDs |
| MCS model | full roster vs published | max 0.04 posterior SDs |
| Joint PCS+MCS | full roster, dispersed chains | R-hat 1.0036; shares 13.4/27.9/58.7 vs report 13.4/27.5/59.2 |
| Mixed five-channel | synthetic recovery, off-bounds truth | all groups pass; R-hat 1.008 |
| Held-out scoring | leakage probe | exactly 0 at a 1e-12 gate |
| SF-12 measures | vs the PCS construction note | `sf12pcs_dv` rebuilt to max err 0.0054 (MCS same, given the zero floor); UK norms, variant-D row and the 6.6-pt artifact exact; SF-6D tariff worked example/ceiling/floor pass |
| GRM (physical) | vs the EIT note's grm.rds fit | own-code Python refit: same 498,424 person-years, discriminations to 3e-5, person-wave theta to 1e-5, age latent distributions to 5dp |

## Setup

```bash
uv sync --extra dev
ln -s /path/to/UKDA-6614-tab data/raw/UKDA-6614-tab
export CMDSTAN=~/.cmdstan/cmdstan-2.38.0
```

Then, in order: `data_cleaning/01_build_panel.py`,
`data_cleaning/02_build_contracts.py`, `data_cleaning/03_build_measures.py`,
and the scripts under `clustering/`.
