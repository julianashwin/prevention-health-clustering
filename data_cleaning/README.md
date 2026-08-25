# data_cleaning

Thin drivers that turn the licensed raw extract into analysis inputs, in order:

| Script | Produces |
|---|---|
| `01_build_panel.py` | `data/processed/ukhls_indresp_processed.csv` |
| `02_build_contracts.py` | frozen sample contracts under `data/processed/contracts/` |
| `03_build_measures.py` | `data/processed/measures/sf12_measures.parquet`: subscales, US/UK PCS-MCS variants, `PCS_phys_only`, SF-6D utility, Farivar composites |
| (next) `04_build_grm.py` | GRM (IRT) theta and TCC-based scores from the EIT note |

All reusable logic lives in `src/prevention_health_clustering/{data,contracts,measures}`;
scripts here only orchestrate and print. Outputs under `data/` are gitignored —
they derive from licensed UKHLS microdata and must never be committed.

`03_build_measures.py` validates itself against the PCS construction note on
every run: variant A must reproduce `sf12pcs_dv` to max error 0.0054 over
528,485 person-years, the UK norms table must match, the 6.6-point
mental-health artifact must reproduce, and the SF-6D tariff must pass its
worked example, ceiling and floor. SF-6D utilities are research-grade —
obtain the IQVIA/Sheffield licence before publishing them.
