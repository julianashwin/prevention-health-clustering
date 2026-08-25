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
