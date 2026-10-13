# Status

Last updated 13 October 2026. The draft is `paper/paper_skeleton.tex` (95 pages,
builds clean); red `\todo` marks in the PDF are open decisions, grey `\note`
marks are drafting notes.

## Fits running or queued (13 October)

Three detached queues, two at a time (about 80% of the machine); each skips
finished fits, so a launcher can be rerun after an interruption. Logs are in
the output directory named.

| Queue | Fits, in order | Log | Hook that reads them |
|---|---|---|---|
| `clustering/runs/frailty_firstheld_launch.sh` | frailty base/ssm, logfrailty base/ssm, h base/ssm first row, theta and h base/ssm with the first two rows held out: all done by 6 Oct 22:41, clean (ROUND2 QUEUE COMPLETE) | `artifacts/health-frailty/queue.log` | `32_paper_bayes_fits.py combined --frailty` / `--logfrailty` (in the draft); `43_paper_type_from_initial.py --first2`, `--first2 --h` and `46_paper_targeting.py --first2`, `--first2 --h` (in the draft, 7 Oct) |
| `clustering/runs/cohort_first2_launch.sh` | theta and h ssm-cohort with the first two rows held out: both done 6 Oct, clean | `artifacts/health-firstheld/queue_cohort_first2.log` | `43 --first2` (the cohort typology is a row of `fig:typeinit`) |
| `clustering/runs/multidim_first2_launch.sh` | theta multidimensional ssm + mortality with the first two rows held out: done 10 Oct 06:54 (13.4 h, converged) | `artifacts/multidim-health/queue_first2.log` | done 11 Oct: `43 --first2` rerun, the multidimensional row of `fig:typeinit` is the two-row fit and the bullet rewritten (theta(0) AUC 0.77, two readings or the mental score 0.81) |
| `clustering/runs/health_k45_frailty_launch.sh` (launched 12 Oct 16:02; **queue stopped by hand 13 Oct 14:35** after the start of log-frailty K = 4, so the K = 5 pair is deferred) | frailty ssm K = 4: done 13 Oct 04:10 (12.2 h; chains 1-3 agree, chain 4 sits 950 log-points lower, so `42` reads [1, 2, 3] via its `CHAINS` map); **running: log-frailty ssm K = 4** (since 13 Oct 04:10, finishing on its own, about 18:00); frailty K = 5 and log-frailty K = 5 deferred: relaunch the launcher after the K = 4 windows queue, it skips the finished fits | `artifacts/health-k45/queue_frailty.log` | `42 --measures`: fills the "not fitted" cells of `fig:returnsmeasures` automatically (check each fit's chains first); rerun it when the K = 4 pair is in, then refresh the bullet |
| `clustering/runs/windows_k4_launch.sh` (launched 13 Oct 14:35; sleeps until log-frailty K = 4 writes its summary, then runs) | theta AR(1)+"spike" at **K = 4** refitted on the ten age windows (`windows_queue.py --k 4`, model `health-theta-ssm-k4`, tags `theta-ssm-k4-{lo}-{hi}`, warmup 1500); the K = 3 windows took 73 h, expect 4-5 days | `artifacts/health-windows/queue_k4.log` | `50_paper_windows_bayes.py` needs a `--k 4` mode (to write) for a K = 4 twin of `fig:windows` |
| `clustering/runs/multidim_health_k45_cohort_launch.sh` | cohort K = 4: done 13 Oct 07:40 (36.1 h; two modes of identical log posterior, -646,561: chains 2 and 3 reproduce the no-shift K = 4 classes (shares 11/26/41/23), chains 1 and 4 a 12/36/39/14 split; read [2, 3] for comparability); **running: cohort K = 5** (since 13 Oct 07:40; expect 60+ h and several modes, as at K = 5 without shifts) | `artifacts/multidim-health/queue_k45_cohort.log` | `39 --k`: a cohort block of rows (paths at the 1950s, same vertical scale) is **not yet coded**; write it when cohort K = 5 lands, choosing chains per fit |
| `clustering/runs/multidim_health_k45_launch.sh` | theta multidimensional ssm at K = 4 (done 8 Oct, 29.4 h: chains 1, 3, 4 agree, chain 2 sits 300 log-points lower, so `39` reads [1, 3, 4]) and K = 5 (done 10 Oct 19:31, 63.3 h: four chains, four different modes, lp -647394 / -647723 / -647600 / -647541; `39` draws chain 1, the highest, alone, R-hat 1.007 within it) | `artifacts/multidim-health/queue_k45.log` | `39_paper_multidim.py --k` drawn with K = 3, 4, 5 (11 Oct) and `49_paper_multidim_k5_chains.py` draws the four K = 5 chains as four rows (`fig_multidim_k5_chains.png`, 12 Oct); not yet in the draft: decide whether a single-chain K = 5 is shown (with the chains figure in the appendix), or K = 5 is rerun with more chains |
| `clustering/runs/windows_launch.sh` | theta ssm refitted on the ten age windows: all done 9 Oct 17:28 (73 h in all, clean) | `artifacts/health-windows/queue.log` | done 13 Oct: `50_paper_windows_bayes.py` draws `fig_windows_bayes.png`, now Figure 10 (`fig:windows`) with three bullets; the K-means version moved to `app:kmeans` as `fig:windows_kmeans`. Agreement is the co-assignment correlation (ARI in the csv). The bad type appears only when the window contains the 40-60 divergence; K=3 unsupported at 70-90 |

K-means twins for the first-rows exercises (`22_paper_kmeans.py --first-held 1 --variants h`,
`--first-held 2`) were run on 5 October; the labels sit in
`artifacts/descriptives/paper_kmeans_labels.parquet` as `h_firstheld`,
`theta_first2held`, `h_first2held`.

**Hooks still pending (13 Oct):**

- `42 --measures` once the frailty / log-frailty K = 4 pair is in (frailty K = 4 landed 13 Oct, log-frailty K = 4 due this evening), and again after the deferred K = 5 pair; refresh the bullet under `fig:returnsmeasures`.
- Cohort K = 4 landed 13 Oct (chains 2, 3): a side-by-side with the no-shift K = 4 was drawn for the chat only (same classes, 11/26/41/23; decade shifts +0.49 for the 1920s falling to -0.45 for the 2000s on the physical channel); goes into the K comparison figure's cohort block when K = 5 lands.
- `50 --k 4` once the K = 4 windows land.
- The K comparison figure (`39 --k`, drawn with K = 3, 4, 5 on 11 Oct; `fig_multidim_trajectories_by_K.png`, not in the draft) waits on a decision about K = 5: show chain 1 alone with the chain-by-chain figure (`49`, `fig_multidim_k5_chains.png`) in the appendix as the caveat, rerun K = 5 with more chains, or drop the K = 5 row. Then place it in Section 4 next to `fig:multidim` with bullets (the K = 4 reading: the worst class shrinks to an 11% chronically ill core, the middle splits along the mental dimension). Also fix the legend overlapping the lines in the K = 4 mental panel.
- A cohort block for `39 --k` once cohort K = 5 lands (code to write).
- Result 2 reformulation in Section 2: proposal made 8 Oct (by-type result first, uniform decomposition as a corollary, additive-prevention remark, moments table to the identification subsection), awaiting the go.

**The held-out convention (7 Oct):** the draft uses the first-TWO-rows fits
throughout the type-from-initial and targeting exercises (`43 --first2`,
`46 --first2`, panel (d) of `fig:prevention`), so that one reading and two
readings (t-1 and t-2) can be compared. The first-row fits remain on disk
(`*-firstho`); their figures are no longer in the draft and have been removed
from `paper/figures` (regenerable with `43` / `46` without `--first2`).

## Not yet set off (7 October)

Fits that have been discussed but not queued, so they are not forgotten:

| Fit | Why | Cost | Decision |
|---|---|---|---|
| cohort-shift versions of the first-*one*-row held-out fits (theta, h) | would complete the type-from-initial typologies | 2 fits, 7-13 h | on ice (5 Oct) |
| h in the multidimensional model at K = 4 and 5 | the h rows of a multidimensional K comparison | 2 fits, 40-60 h, mode risk | on ice (5 Oct) |
| the single-channel returns on the first-row-held-out fit's own classes (`42 --firstho`) | would make the targeting figure self-contained on one fit | no fit needed, a script flag | open, and now moot since the draft uses the first-two-rows fits |
| Bayesian windows on h | the appendix twin of `fig:windows` (the K-means h version is there now) | 10 fits, about 70 h | open (12 Oct) |
| frailty and log-frailty AR(1)+"spike" at K = 5 | the last two "not fitted" cells of `fig:returnsmeasures` | 2 fits, 20-25 h each | deferred 13 Oct behind the K = 4 windows; relaunch `health_k45_frailty_launch.sh` |
| multidimensional K = 5 with more chains (8) | to see whether chain 1's mode is found repeatedly | 1 fit, about 60 h | open (12 Oct); the alternative is to show chain 1 with the caveat |
| mental GRM at K = 4 and 5 | would complete the mental appendix | 2 fits, 10-20 h | open (12 Oct) |
| a two-row model posterior for the multidimensional typology in `43` | the single-channel mixtures have it; the multidimensional row's model posterior uses the first row only | no fit, code | open (11 Oct) |

## Feedback of 29 September (`paper/feedback_29sept.txt`)

| Item | Status |
|---|---|
| Predict the type from an initial, held-out health value, baseline and full clustering | Done: `fig:typeinit`, reframed (5 Oct) as observables versus a health reading and (7 Oct) moved to the first-two-rows fits: education, income rank and sex add 0.08-0.10 of AUC over age, one reading of theta takes it to 0.80-0.81, two readings to 0.85 (four to five points of accuracy), and on top of a reading the observables add 0.01; the mixture's own posterior from one or two rows matches the regression. K-means / independent / AR(1)+"spike" / + cohort shifts / multidimensional (first-row placeholder until the two-row fit lands); the h version is `fig:typeinit_h` in the measures appendix |
| Relabel "spike" as "episodic" | Open (a global rename in the draft and the figure footers; `descriptives/` prints "spike" in about 30 places) |
| Is the C_Hd reversal driven by the small very-ill group? | Open; the bottom-decile and worst-class composition figures exist (`fig:decomp`, `fig:multidim`) but the statement has not been made |
| Result 2: focus on the by-type form and why the evolution of the mean matters | Open (Section 2 rewrite; see the framework todos below) |
| Same vertical axis across K panels | Done (5 Oct; and 6 Oct a footer guard in `_style.py` re-flows any figure note wider than its panels, every paper figure regenerated): `_style.share_ylim` gives one vertical scale to the six K panels (`fig_class_trajectories_by_K`, theta and h), to each channel across rows in the multidimensional figures (specifications, and K = 3/4/5 when those land), to the sliding and expanding windows, to the raw and net decade profiles, and to the two specifications in the returns figure. Panels on different measures or quantities keep their own scales |
| More discrimination in the unhealthy type as K rises and as the stochastic structure is added? | Partly: the K = 4/5 single-channel rows are in; the multidimensional K = 4/5 fits are queued |
| Is there always a bad group, and does K-means understate it because the process is noisy? | Open; the ingredients are `fig:bayes` (dashed K-means lines) and the composition strips |
| Express type differences as LE / HLE | Done (6 Oct): `fig:le` / `tab:le` at the end of subsection (B), from `44_paper_life_expectancy.py`: class hazards of the multidimensional K = 3 AR(1)+"spike" fit, one common age-varying factor calibrated so population LE at 65 and 90 matches ONS 2020-22 (attrition correction); HLE with healthy = self-rated good or better. Open todo: whether to calibrate the healthy share to ONS health state life expectancies too |
| Univariate mental-health clusters versus the multidimensional classes | Done: Appendix `app:mental`, and (6 Oct) `fig:multidim_vs_uni` in Section 4 after the multidimensional figure: the multidimensional classes are the physical classes (73% same class) with a mental path attached; the univariate mental classes partition by persistence, ARI 0.20 with the multidimensional ones. The main-text multidimensional figure is now the cohort fit (the no-cohort fit moved to Appendix `app:cohort`) |
| Windows (`fig:windows`): class composition by age as the window moves | Done (6 Oct on K-means, 13 Oct on the AR(1)+"spike" fits, now the main-text figure): on the mixture the worst class is 33% in the 40-60 window and 7-14% of the person-waves at 60-85 in windows starting at 50 or later (lifecycle 18-22%); backward windows find the 25-30% bad type only once they reach back to 40. On K-means (appendix) the rise in the worst share with the window's age is composition. Agreement statistic: the co-assignment correlation |
| Returns with targeting on observables or on theta(t-1) (6 Oct: Section 5's main figure is now `fig:prevention`, a 2 x 2 of the class returns and the bill beside the targeting panels, AR(1)+"spike" theta only; the detailed figures moved to Appendix `app:returns_detail`, and `app:returns_models` shows the return by class under six typologies and on the four measures) | Done (6 Oct): `fig:targeting` / `tab:targeting` after the returns table, from `46_paper_targeting.py`: rules on what is known at the first observation (age; + observables; + one theta reading; both; the model's posterior; the oracle), treating the top q by predicted return and scored by the realised return. On the first-two-rows fits (7 Oct): one theta reading closes 40% of the age-to-oracle gap at 10% coverage, observables a fifth, two readings 53%; the model's posterior from one or two rows matches the regressions; the h version is `fig:targeting_h` in the measures appendix. Open todo: net of a screening cost; targeting on the predicted class rather than the predicted return |
| Treatment as the cost prevented; when the savings come (done 7 Oct: panel (c) of `fig:prevention` and `fig:timing` in the appendix, with the half-way ages and the present values at 3.5%); LE and inequality effects of prevention | Treatment as the cost prevented: done (6 Oct). The tau instrument and T(r) are gone; the framework and Section 5 now price the treatment bill still to come C(r), split into the cost at the mean path, type differences and realised deviations (the unpreventable part, 16% from 30), and prevention as the share of the bill it avoids (P/C 0.91, 0.67, 0.39 from 30, 50, 70). The note's T(r) is 2C(r) under the quadratic. Timing of the savings: done 7 Oct (see the item name). LE and inequality effects: done 7 Oct, `fig:prevle` / `tab:prevle` at the end of Section 5 from `48_paper_prevention_le.py` (LE and HLE by class under targeted / same-budget uniform / full uniform prevention; class gaps; sd of age at death; education, income and regional gradients through class composition). The univariate AR(1)+"spike" versions on theta, h, the frailty index and log frailty (class hazards estimated by weighted MLE on the same death records, the logit links re-estimated on the measure) are `fig:prevle_uni`, `fig:prevle_h`, `fig:prevle_frailty`, `fig:prevle_logfrailty` and `tab:prevle_measures` in `app:returns_models` (8 Oct); the frailty K = 4/5 fits are not needed for these (K = 3 throughout). Open todo: within-class gradients, and the cost of a unit of mu by class |

## Open decisions in the draft (the red todos)

- Section 2: Result 2 around the by-type return (line 126; proposal of 8 Oct awaiting the go); which noise class to lead with, whether a permanent shock is a type level, selection (lines 223-225).
- Section 3: the PSID replication (line 246); which Exhibit 2 cases to present, pairwise versus balanced moments (line 376).
- Section 4: a bounded observation model for h and frailty (line 529); whether to calibrate the healthy share to the ONS health state life expectancies (line 577).
- Section 5: the cost of a unit of mu and survival weighting in the timing arithmetic (line 694); pricing the targeting rules net of a screening cost and targeting on the predicted class (line 698); within-class gradients in the prevention-in-years exercise, and the cost of a unit of mu by class (line 717).
- The introduction (line 31).

## Housekeeping

- Work is committed as checkpoints after each change (last: 13 Oct). `paper/feedback_29sept.txt` carries an uncommitted edit by the user (marking the timing item done); left for the user to commit.
- 13 Oct: four outputs no script produces any more were deleted from `paper/` (`fig_bayes_ssm.png`, `fig_returns_bayes_uncapped.png`, `tab_bayes_ssm.tex`, `tab_returns_bayes_uncapped.tex`). Outputs that scripts still produce but the draft does not place (the single-measure frailty figures of `23`, `25`, `29`, `37`; `tab_observables`, `tab_k_comparison`, `tab_measure_*`, `fig_bayes_base`; the per-measure `tab_prevention_le_*` and the `_first2_h` tables) are kept as regenerable by-products.
- Editor build files (`paper/*.synctex*`) are now git-ignored.
- `paper/paper_overleaf/` and `paper_overleaf.zip` are an 18 September copy for Overleaf and are stale; regenerate or drop before sharing.
- `measuring_health/health_measures_note.tex` is the archived measure-search note and describes the retired measures; it is kept as a document only.
- `artifacts/scratch/contract_backup_oct5/` is the health contract as it was before fi10 was dropped (same rows; only the extra column differs). Safe to delete once the October fits have been read.
