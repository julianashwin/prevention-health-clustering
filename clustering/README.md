# clustering

The Bayesian latent-class modelling stream.

- `validation/` — comparisons against the predecessor's published fits and
  synthetic parameter-recovery runs. Key results: PCS and MCS full-roster fits
  reproduce the published posteriors to ≤0.05 posterior SDs; the joint fit
  matches the reported modal shares with honestly dispersed chains.
- `benchmarks/` — timing probes and negative results (sufficient-statistic and
  vectorised variants are SLOWER than the row loop; kept as recorded findings).

Fast unit tests live in `tests/`, not here. Fits write to `artifacts/`.

## Batch fitting

`run_fit.py` fits any registry model on any frozen contract with the honest
partition-init recipe, computing structural diagnostics without touching
person-level columns; `--holdout-last-k 2 --holdout-min-obs 5` holds out each
person's last two observations (sample limited to people with at least five)
and writes per-person held-out predictive densities.
`runs/health_base_queue.py` (launched by `runs/health_base_launch.sh`)
runs the three baseline K=3 fits on the paper's measure, theta / h / fi10, one
at a time into `artifacts/health-base/`; `runs/health_ssm_queue.py` (launched by
`runs/health_ssm_launch.sh`) the same three under the AR(1) plus measurement error specification
into `artifacts/health-ssm/`; `runs/health_cohort_queue.py` (launched by
`runs/health_cohort_launch.sh`, which waits for the AR(1) plus measurement error queue to finish)
the same three with a shared birth-decade level shift, into
`artifacts/health-ssm-cohort/`; `runs/health_next_queue.py` (launched by
`runs/health_next_launch.sh`) the AR(1)-without-measurement-error comparison
and the held-out twins, into `artifacts/health-next/`.
`runs/overnight_queue.py` runs the eight-fit AR(1)/GRM batch, two at a time,
longest first, and writes `artifacts/overnight/digest.json`.
`oos_assessment.py` is the full out-of-sample assessment of the holdout
fits: it reproduces the model's own held-out density offline (validated to
corr 0.9999998), adds the AR-conditional forecast the generated quantities
do not compute, and scores both against age-quadratic and
last-observation-carried-forward benchmarks on density, point accuracy and
interval calibration (note section 9). It needs structural draws extracted
from the chain CSVs (columns 1-33) into `<dir>/<tag>.csv`.

`run_multidim.py` fits the multidimensional mixture: a physical channel
(`--physical theta` or `h`) and the mental GRM, Gaussian with optional AR(1)
or AR(1) plus "spike" and persistence by class and channel; an optional
negative-binomial chronic count (`--with-chronic`, off in the paper's runs);
and a Gompertz-Makeham mortality hazard (`--with-mortality`, class-specific
level and slope, common Makeham constant, integrated over each row's years at
risk, never held out). Variants: baseline, holdout, ar1, ar1-holdout, ssm,
ssm-holdout, on `multidim_health_20_89_minobs3_v1`
(`data_cleaning/08_build_multidim_health_contract.py`).
`runs/multidim_health_queue.py` (launcher `multidim_health_launch.sh`) runs
theta-ssm-mort, h-ssm-mort, theta-base-mort, h-base-mort one at a time into
`artifacts/multidim-health/`; `runs/multidim_mode_check.py` reads a running
fit's chains for a mode split. The archived four-channel fits in
`artifacts/multidim*` were run by the previous version of the script
(logit-quadratic hazard, one persistence per class) and no longer match the
Stan source.
