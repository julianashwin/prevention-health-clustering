# clustering

The Bayesian latent-class modelling stream.

- `validation/` — comparisons against the predecessor's published fits and
  synthetic parameter-recovery runs. Key results: PCS and MCS full-roster fits
  reproduce the published posteriors to ≤0.05 posterior SDs; the joint fit
  matches the reported modal shares with honestly dispersed chains.
- `benchmarks/` — timing probes and negative results (sufficient-statistic and
  vectorised variants are SLOWER than the row loop; kept as recorded findings).

Fast unit tests live in `tests/`, not here. Fits write to `artifacts/`.
