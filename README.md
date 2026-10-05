# prevention-health-clustering

Health trajectories in Understanding Society (UKHLS): a health measure and a
cost index built from the raw panel, descriptive moments, partial K-means and
Bayesian latent-class models of the lifecycle, a multidimensional model with
mortality, and the prevention-versus-treatment returns the paper prices. Every
figure and table in `paper/paper_skeleton.tex` is produced by a script in this
repository.

**Where things stand** (running fits, hooks waiting on them, the feedback
items and the open decisions): [`STATUS.md`](STATUS.md).

## The four health measures

The project reports four measures and nothing else:

| Measure | What it is | Built by |
|---|---|---|
| `theta` | EAP score of the health graded-response model (six SF-12 items as four testlets, three impairment-area counts, an ever-diagnosed condition count); the main-text measure | `data_cleaning/04_build_health.py` |
| `h` | the same model's expected-score share, 0 to 1; a monotone transform of `theta` | same |
| `frailty` | the 31-deficit Rockwood-style index (higher = frailer); robustness | `data_cleaning/04b_build_frailty.py` |
| `logfrailty` | `log(frailty + 1/31)`, Hosseini-style, zeros kept at the floor; robustness | same |

Plus the mental GRM `theta_ment_nodepr` (`data_cleaning/04c_build_mental_grm.py`),
the second channel of the multidimensional model and the measure of the
mental-health appendix. The earlier measure generations (SF-12 PCS/MCS and
their variants, SF-6D, the physical GRM banks, the ten-item deficit index) and
every fit on them were removed in October 2026.

## Pipeline

```
raw UKHLS tabs
  └─ data_cleaning/01 panel ─ 02 item cache ─ 03 chronic ─ 04 health GRM ─ 04b frailty ─ 04c mental GRM
                              └─ 05 cost index ─ 06 observables
       └─ 07 health contract (theta, h; frailty, logfrailty auxiliary)     38,963 people, 334,194 person-ages
       └─ 08 multidim contract (theta, h, mental, mortality event)         38,181 people, 325,822 person-ages
            └─ clustering/run_fit.py        single-channel K-class mixtures   -> artifacts/health-*/, mental-health/
            └─ clustering/run_multidim.py   physical + mental + Gompertz-Makeham mortality -> artifacts/multidim-health/
                 └─ descriptives/22..43_paper_*.py  figures and tables -> paper/figures/, paper/tables/
                      └─ paper/paper_skeleton.tex
```

Each stage's README says what every script reads and writes:
[`data_cleaning/`](data_cleaning/README.md), [`clustering/`](clustering/README.md)
(the fit sets and the queues that produced them), [`descriptives/`](descriptives/README.md)
(script-by-script, with the `--h`, `--measures`, `--frailty`, `--mental`,
`--cohort`, `--first2` and `--k` modes), [`paper/`](paper/README.md).

## Layout

```
├── data_cleaning/     raw extract -> panel -> measures -> contracts
├── clustering/        run_fit.py, run_multidim.py, runs/ (the queues), one salvage tool
├── descriptives/      15-17, 21: the measure and cost-index notes; 22-43: the paper's figures and tables
├── paper/             paper_skeleton.tex, figures/, tables/, the feedback notes
├── measuring_health/  the empirical construction note (measure + cost index)
├── conceptual_framework/  the framework note, its figures and checks
├── policy/            counterfactual analysis (planned)
├── src/prevention_health_clustering/
│   ├── data/          UKHLS ingest and cleaning
│   ├── measures/      items, the GRM estimator, the health bank + frailty, the mental bank, chronic, cost
│   ├── contracts/     frozen sample contracts
│   ├── models/        the two Stan programs + the model registry
│   ├── runner/        payload building, fitting, initialisation (single-channel and multidim)
│   └── trajectories/  partial K-means
├── tests/             fast unit tests of the package (plus one slow Stan test)
├── data/              gitignored: licensed inputs and derived data
└── artifacts/         gitignored: fits (artifacts/<set>/<tag>/run_summary.json + chains/), descriptives data
```

## Fit sets on disk

All K = 3 unless stated; `ssm` = an AR(1) latent state plus a one-period
"spike", `base` = independent residuals, `cohort` = plus one level shift per
birth decade common to the classes, `ho` = last two rows held out, `firstho`
/ `first2ho` = first one / two rows held out.

| Directory | Fits |
|---|---|
| `artifacts/health-base`, `health-ssm`, `health-ssm-cohort` | theta, h |
| `artifacts/health-next` | theta, h: plain AR(1) (no "spike") and the `ssm-ho` twins |
| `artifacts/health-k45` | theta, h: base and ssm at K = 4, 5 |
| `artifacts/health-frailty` | frailty, logfrailty: base and ssm |
| `artifacts/health-firstheld` | theta, h: base, ssm (and ssm-cohort) with the first one or two rows held out |
| `artifacts/mental-health` | the mental GRM: base, ssm, ssm-cohort |
| `artifacts/multidim-health` | theta or h + mental + mortality: base, ssm, ssm-cohort, ssm-ho, ssm-cohort-ho, ssm-firstho, and theta ssm at K = 4, 5 |

A fit is read by the paper scripts only through its `run_summary.json`,
`stan_data.json` and, where a chain subset is needed, its `chains/*.csv`.

## Setup

```bash
uv sync --extra dev
ln -s /path/to/UKDA-6614-tab data/raw/UKDA-6614-tab
export CMDSTAN=~/.cmdstan/cmdstan-2.38.0
```

Then `data_cleaning/01` to `08` in order, the queues under `clustering/runs/`
(each launcher is a detached `nice` job that skips finished fits), and
`descriptives/22` to `43` in the order their README gives; `paper/README.md`
has the LaTeX build.

## Validation

| Component | Check | Result |
|---|---|---|
| Health measure | gates per run | max positive Q3 +0.05, EAP identity 1.000, the weighted-sum twin agreeing with h in rank at 0.99 (`measuring_health/health_measure_construction.pdf`) |
| Mental GRM | 3 gates per run | the diagnosis-free mental bank, GHQ wording testlets triggered by Q3 as pre-specified; EAP identity 1.000 |
| Contracts | rebuild | row and column identity against the previous build is checked whenever a contract is rebuilt (the October 2026 rebuild left every kept column byte-identical) |
| Held-out windows | `tests/test_gq_heldout.py`, `tests/test_payload_holdout.py` | Stan's held-out generated quantities match a numpy reimplementation on simulated data; last-k, first-k and age holdouts partition the rows as specified |
| Mode splits | `clustering/runs/multidim_mode_check.py`, per-chain `lp__` in every `run_summary.json` | fits read over a chain subset are listed in `descriptives/39_paper_multidim.py` and `_paper_common.py` |
