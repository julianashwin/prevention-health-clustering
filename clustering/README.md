# clustering

The Bayesian latent-class modelling stream.

Fast unit tests live in `tests/` (the held-out generated quantities against a
numpy reimplementation, the holdout window partitions), not here. Fits write
to `artifacts/`.

## Batch fitting

`run_fit.py` fits any registry model on any frozen contract with the honest
partition-init recipe, computing structural diagnostics without touching
person-level columns; `--holdout-last-k 2 --holdout-min-obs 5` holds out each
person's last two observations (sample limited to people with at least five)
and writes per-person held-out predictive densities.
`runs/health_base_queue.py` (launched by `runs/health_base_launch.sh`)
runs the two baseline K=3 fits on the paper's measure, theta / h, one
at a time into `artifacts/health-base/`; `runs/health_ssm_queue.py` (launched by
`runs/health_ssm_launch.sh`) the same two under the AR(1) plus measurement error specification
into `artifacts/health-ssm/`; `runs/health_cohort_queue.py` (launched by
`runs/health_cohort_launch.sh`, which waits for the AR(1) plus measurement error queue to finish)
the same two with a shared birth-decade level shift, into
`artifacts/health-ssm-cohort/`; `runs/health_next_queue.py` (launched by
`runs/health_next_launch.sh`) the AR(1)-without-measurement-error comparison
and the held-out twins, into `artifacts/health-next/`; `runs/mental_queue.py`
(launched by `runs/mental_launch.sh`, which waits for the multidimensional
cohort queue to finish) the K=3 fits on the mental GRM alone, independent
residuals, AR(1) plus "spike" and AR(1) plus "spike" with birth-decade shifts
(`mental-theta-base`, `mental-theta-ssm`, `mental-theta-ssm-cohort`), on the
multidim health contract's rows, into `artifacts/mental-health/`;
`runs/health_firstheld_queue.py` (launched by `runs/health_firstheld_launch.sh`,
which waits for the mental queue) the theta independent-residuals and AR(1) plus
"spike" fits with each person's FIRST row held out (`run_fit.py --holdout-first-k 1
--holdout-min-obs 4`), into `artifacts/health-firstheld/`, for the
type-from-an-initial-value exercise (`descriptives/43_paper_type_from_initial.py`);
`runs/multidim_firstheld_launch.sh` the multidimensional twin, `run_multidim.py
--variant ssm-firstho` (theta + mental GRM + mortality, first row held out, people
with at least four rows) into `artifacts/multidim-health/theta-ssm-mort-firstho/`.
`runs/frailty_firstheld_queue.py` (launcher `runs/frailty_firstheld_launch.sh`)
runs the K=3 fits on the 31-deficit frailty index and log(frailty + 1/31),
independent residuals and AR(1) plus "spike", into `artifacts/health-frailty/`,
then h with the first row held out and theta / h with the first two rows held
out (`--holdout-first-k 2 --holdout-min-obs 5`, tags `*-first2ho`) into
`artifacts/health-firstheld/`; `runs/cohort_first2_queue.py` (launcher
`runs/cohort_first2_launch.sh`, run alongside) the AR(1) plus "spike" plus
birth-decade-shift twins of the first-two-rows fits; and
`runs/multidim_health_k45_queue.py` (launcher `runs/multidim_health_k45_launch.sh`,
which waits for the frailty/first-held queue) the multidimensional theta
AR(1) plus "spike" + mortality fit at K = 4 and K = 5
(`artifacts/multidim-health/theta-ssm-mort-k{4,5}/`).
`run_multidim.py` fits the multidimensional mixture: a physical channel
(`--physical theta` or `h`) and the mental GRM, Gaussian with optional AR(1)
or AR(1) plus "spike" and persistence by class and channel, and a Gompertz-Makeham mortality hazard (`--with-mortality`, class-specific
level and slope, common Makeham constant, integrated over each row's years at
risk, never held out). Variants: baseline, holdout, ar1, ar1-holdout, ssm,
ssm-holdout, on `multidim_health_20_89_minobs3_v1`
(`data_cleaning/08_build_multidim_health_contract.py`).
`runs/multidim_health_queue.py` (launcher `multidim_health_launch.sh`) runs
theta-ssm-mort, h-ssm-mort, theta-base-mort, h-base-mort one at a time into
`artifacts/multidim-health/`; `runs/multidim_health_ho_queue.py` (launcher
`multidim_health_ho_launch.sh`) refits the two state-space versions with each
person's last two rows held out, for the prediction exercise;
`runs/multidim_health_cohort_queue.py` (launcher
`multidim_health_cohort_launch.sh`) fits theta-ssm-mort-cohort and its
held-out twin with `--cohort decade` (birth-decade level shifts on both
Gaussian channels, common to the classes, oldest decade = 0);
`runs/multidim_mode_check.py` reads a running fit's chains for a mode split.
