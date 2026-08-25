# data_cleaning

Thin drivers that turn the licensed raw extract into analysis inputs, in order:

| Script | Produces |
|---|---|
| `01_build_panel.py` | `data/processed/ukhls_indresp_processed.csv` |
| `02_build_contracts.py` | frozen sample contracts under `data/processed/contracts/` |
| (next) `03_build_measures.py` | alternative health measures: phys composite, SF-6D, GRM |

All reusable logic lives in `src/prevention_health_clustering/{data,contracts,measures}`;
scripts here only orchestrate and print. Outputs under `data/` are gitignored —
they derive from licensed UKHLS microdata and must never be committed.
