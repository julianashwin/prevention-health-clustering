# prevention-health-clustering

Bayesian latent-class health trajectory models for Understanding Society (UKHLS).

A focused rebuild of the analysis core from `ukhls-health-clusters`, carrying
only the parts that are verified, used, and understood. The predecessor
repository remains the archive of record for published evidence and its audit
trail.

## Status

| Step | Component | State |
|---|---|---|
| 1 | Data layer and frozen sample contracts | reproduces the published roster exactly |
| 2 | Gaussian mixture model + registry + runner | in progress |
| 3 | Held-out scoring | pending |
| 4 | Multidimensional model | pending |

## Validation

The step-1 contract is checked against the published `six-fit-v1` parameter
bundle, which was fitted on the same roster:

| Quantity | This repo | Published | Delta |
|---|---|---|---|
| Persons | 50,194 | 50,194 | exact |
| Person-age rows | 446,966 | 446,966 | exact |
| PCS mean | 49.1684050688419 | 49.1684050688419 | 0 |
| PCS sd | 11.172456701314845 | 11.1724567013148 | 4.4e-14 |

## Setup

```bash
uv sync --extra dev
ln -s /path/to/UKDA-6614-tab data/raw/UKDA-6614-tab
```
