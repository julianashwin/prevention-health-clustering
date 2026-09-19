# data_cleaning

Thin drivers that turn the licensed raw extract into the paper's two analysis
inputs — the health measure and the cost index — in order:

| Script | Produces |
|---|---|
| `01_build_panel.py` | `data/processed/ukhls_indresp_processed.csv`: the processed UKHLS panel |
| `02_build_measures.py` | `data/interim/sf12_items_long.parquet` (the item extract every later step reads) and `data/processed/measures/sf12_measures.parquet`: subscales, US/UK PCS-MCS variants, `PCS_phys_only`, SF-6D utility, Farivar composites |
| `03_build_chronic.py` | `data/processed/measures/chronic_conditions.parquet`: per-condition ever indicators, diagnosis ages, clinical group counts, 10-year-recency counts |
| `04_build_health.py` | `data/processed/measures/health_measure.parquet`: the health measure as `theta`, `h`, `fi10` and the weighted-sum twin `ws`, plus item parameters, Q3, the latent age profile and the appendix weights |
| `05_build_cost_proxy.py` | `data/processed/measures/cost_index.parquet`: the flat-rate and condition-weighted cost index, one row per person-wave that answered the utilisation block |
| `07_build_health_contract.py` | `data/processed/contracts/health_lifecycle_20_89_minobs3_v1`: the clustering contract on the health measure, carrying `theta`, `h` and `fi10` as metrics (38,963 people, the P-FULL roster) |
| `06_build_observables.py` | `data/processed/measures/observables.parquet` and `observables_person.parquet`: equivalised household net income as within-wave percentile ranks, and highest qualification, read from the raw tab files; the paper's observational comparison |

```bash
cd ~/Documents/GitHub/prevention-health-clustering && for s in data_cleaning/0*.py; do PYTHONPATH=src .venv/bin/python "$s" || break; done
```

All reusable logic lives in `src/prevention_health_clustering/{data,contracts,measures}`;
scripts here only orchestrate and print. Outputs under `data/` are gitignored —
they derive from licensed UKHLS microdata and must never be committed.

## The health measure

`04_build_health.py` fits the bank defined in `measures/health.py`: the four
validated SF-12 physical testlets (GH, PF, RP, BP), three counts over the UKHLS
impairment areas (functional, self-care, sensory) and one count of
ever-diagnosed physical conditions. 387,599 person-waves, 69,958 people, ages
20–90. It is reported three ways, all monotone in the same answers: `theta`
from the graded-response model, `h` from its expected-score curve on [0, 1],
and `fi10`, a ten-deficit index anyone can compute by hand. What it is and why
it is this: `measuring_health/health_measure_construction.tex`. The search
behind the choice is archived (see below).

Every run gates on: no positive local dependence between testlets (max Yen's
Q3 +0.05), the EAP identity (Var(theta) + mean posterior variance = 1), the
three variants agreeing in rank (Spearman 0.99), and — while the search
outputs are still on disk — exact reproduction of the search's P-LIM3+CC
scores.

## Validation carried by the other steps

`02_build_measures.py` validates itself against the PCS construction note on
every run: variant A must reproduce `sf12pcs_dv` to max error 0.0054 over
528,485 person-years, the UK norms table must match, the 6.6-point
mental-health artifact must reproduce, and the SF-6D tariff must pass its
worked example, ceiling and floor. SF-6D utilities are research-grade —
obtain the IQVIA/Sheffield licence before publishing them.

`03_build_chronic.py` validates the total condition count against the EIT
note's `build_chronic_count.rds` and reports the diagnosis-timing coverage the
recency sensitivity rests on.

`05_build_cost_proxy.py` prices every admission at the average non-elective
spell plus excess bed days, and out-patient and GP contacts at their national
average unit costs; the condition-weighted variant imputes an unattributed
admission's weight within decile of the health measure, never globally. The
index is a cost variable, so it covers every person-wave that answered the
utilisation block (301,390 of them), health score or not.

## archive/

The measure search and the earlier measure generations, moved intact and still
runnable. Each script's docstring says what superseded it and who still reads
its output.

| Script | Status |
|---|---|
| `02_build_contracts.py` | PCS-based clustering contracts; the fits in `artifacts/` were estimated on them |
| `04_build_grm.py` | the first graded-response measure (`grm_scores.parquet`); also the EIT-note replication |
| `06_build_grm2.py` | the physical/mental bank generation, P-FUNC / P-FULL / P-REC (`grm2_scores.parquet`) |
| `07_build_grm_contracts.py` | frozen lifecycle contracts for the GRM clustering channels (`physgrm_*`, `combgrm_*`) |
| `07_build_limitation_banks.py` | the measure search: six limitation banks under both models, ten condition banks |
| `08_build_multidim_contract.py` | the four-channel multidimensional contract |

Archived does not mean dead: several `descriptives/` scripts and the existing
clustering fits still read `grm_scores.parquet`, `grm2_scores.parquet` and the
contracts. What is settled is that new work uses `health_measure.parquet`, and
that the clustering contract on it is the next build step to write.
