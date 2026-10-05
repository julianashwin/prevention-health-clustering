# data_cleaning

Thin drivers that turn the licensed raw extract into the paper's two analysis
inputs — the health measure and the cost index — in order:

| Script | Produces |
|---|---|
| `01_build_panel.py` | `data/processed/ukhls_indresp_processed.csv`: the processed UKHLS panel |
| `02_build_measures.py` | `data/interim/sf12_items_long.parquet` (the item extract every later step reads): the SF-12 and GHQ-12 items, age and identifiers, one row per person-wave |
| `03_build_chronic.py` | `data/processed/measures/chronic_conditions.parquet`: per-condition ever indicators, diagnosis ages, clinical group counts, 10-year-recency counts |
| `04_build_health.py` | `data/processed/measures/health_measure.parquet`: the health measure as `theta`, `h` and the weighted-sum twin `ws`, plus item parameters, Q3, the latent age profile and the appendix weights |
| `04b_build_frailty.py` | `data/processed/measures/frailty_index.parquet`: a Rockwood-style frailty index, the share of 31 equal-weight deficits (six SF-12 physical items graded 0-1, eight impairment areas, the ever-diagnosed conditions; higher = frailer), its shifted log, log(frailty + 1/31), which keeps the 7% of zeros at the floor, and for reference the log on positive frailty only as in Hosseini, Kopecky and Zhao (2022). A comparison ruler, not the paper's measure; runs after 04 and before 07, which carries both as auxiliary contract columns (`frailty`, `logfrailty`) without changing the contract's rows or metrics |
| `04c_build_mental_grm.py` | `data/processed/measures/mental_grm.parquet`: the mental GRM score `theta_ment_nodepr` (three SF-12 mental testlets and two GHQ-12 wording testlets, no depression diagnosis), the multidimensional model's second channel; also `mental_grm_items.csv` |
| `05_build_cost_proxy.py` | `data/processed/measures/cost_index.parquet`: the flat-rate and condition-weighted cost index, one row per person-wave that answered the utilisation block |
| `07_build_health_contract.py` | `data/processed/contracts/health_lifecycle_20_89_minobs3_v1`: the clustering contract on the health measure, carrying `theta` and `h` as metrics (38,963 people), with `frailty` and `logfrailty` as auxiliary columns |
| `08_build_multidim_health_contract.py` | `data/processed/contracts/multidim_health_20_89_minobs3_v1`: the multidimensional contract, the health contract's people with the mental GRM (`theta_ment_nodepr`, from the archived GRM-2 banks) on every row, carrying `theta`, `h`, `theta_ment_nodepr` and the death event on each decedent's last row (38,181 people, 2,045 deaths) |
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
20–90. It is reported two ways, both monotone in the same answers: `theta`
from the graded-response model and `h` from its expected-score curve on
[0, 1]. What it is and why it is this:
`measuring_health/health_measure_construction.tex`.

Every run gates on: no positive local dependence between testlets (max Yen's
Q3 +0.05), the EAP identity (Var(theta) + mean posterior variance = 1), and the
weighted-sum twin agreeing with h in rank (Spearman 0.99).

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

## Removed

The measure search and the earlier measure generations (SF-12 PCS/MCS and
their variants, SF-6D, the physical GRM banks P-FUNC / P-FULL / P-REC, the
combined GRM, the ten-deficit index and the four-channel multidimensional
contract) have been removed from the repository together with the fits on
them. The project reports four health measures: `theta` and `h`, and the
31-deficit frailty index and its log as robustness measures; the mental GRM
(`04c_build_mental_grm.py`) is the second channel of the multidimensional
model.
