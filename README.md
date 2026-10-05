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
├── data_cleaning/   raw extract -> panel -> health measure + cost index
├── descriptives/    figures and tables            -> artifacts/descriptives/
├── clustering/      model fits, validation, benchmarks -> artifacts/
├── policy/          counterfactual analysis (planned)
├── measuring_health/  the empirical construction note (measure + cost index), the archived measure search
├── conceptual_framework/  the framework note, its figures and checks
├── paper/           the paper skeleton
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
| Health measure | gates per run | the bank the project uses: max positive Q3 +0.05, EAP identity 1.000, the weighted-sum twin agreeing with h in rank at 0.99 — see measuring_health/health_measure_construction.pdf |
| Mental GRM | 3 gates per run | the diagnosis-free mental bank, GHQ wording testlets triggered by Q3 as pre-specified; EAP identity 1.000 (`data_cleaning/04c_build_mental_grm.py`) |
| Held-out windows | `tests/test_gq_heldout.py`, `tests/test_payload_holdout.py` | Stan's held-out generated quantities match a numpy reimplementation on simulated data; last-k, first-k and age holdouts partition the rows as specified |

## Setup

```bash
uv sync --extra dev
ln -s /path/to/UKDA-6614-tab data/raw/UKDA-6614-tab
export CMDSTAN=~/.cmdstan/cmdstan-2.38.0
```

Then `data_cleaning/01_build_panel.py` through `05_build_cost_proxy.py` in
order (see `data_cleaning/README.md`), and the scripts under `clustering/`.
The clustering contracts are built by the archived scripts under
`data_cleaning/archive/`, which still hold the measures those fits were
estimated on.
