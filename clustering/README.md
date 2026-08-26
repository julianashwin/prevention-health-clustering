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
`runs/overnight_queue.py` runs the eight-fit AR(1)/GRM batch, two at a time,
longest first, and writes `artifacts/overnight/digest.json`.
`oos_assessment.py` is the full out-of-sample assessment of the holdout
fits: it reproduces the model's own held-out density offline (validated to
corr 0.9999998), adds the AR-conditional forecast the generated quantities
do not compute, and scores both against age-quadratic and
last-observation-carried-forward benchmarks on density, point accuracy and
interval calibration (note section 9). It needs structural draws extracted
from the chain CSVs (columns 1-33) into `<dir>/<tag>.csv`.

`run_multidim.py` fits the multidimensional mixture (two GRM channels
Gaussian with optional AR(1), chronic count negative-binomial, mortality
hazard toggleable and never held out) in four variants: baseline, holdout,
ar1, ar1-holdout. `runs/multidim_queue.py` runs all four, two at a time.
The sample filter applies to every variant so the four differ by
specification alone.
