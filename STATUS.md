# Status

Last updated 7 October 2026. The draft is `paper/paper_skeleton.tex` (85 pages,
builds clean); red `\todo` marks in the PDF are open decisions, grey `\note`
marks are drafting notes.

## Fits running or queued (5 October)

Three detached queues, two at a time (about 80% of the machine); each skips
finished fits, so a launcher can be rerun after an interruption. Logs are in
the output directory named.

| Queue | Fits, in order | Log | Hook that reads them |
|---|---|---|---|
| `clustering/runs/frailty_firstheld_launch.sh` | frailty base/ssm, logfrailty base/ssm, h base/ssm first row, theta and h base/ssm with the first two rows held out: all done by 6 Oct 22:41, clean (ROUND2 QUEUE COMPLETE) | `artifacts/health-frailty/queue.log` | `32_paper_bayes_fits.py combined --frailty` / `--logfrailty` (in the draft); `43_paper_type_from_initial.py --first2`, `--first2 --h` and `46_paper_targeting.py --first2`, `--first2 --h` (in the draft, 7 Oct) |
| `clustering/runs/cohort_first2_launch.sh` | theta and h ssm-cohort with the first two rows held out: both done 6 Oct, clean | `artifacts/health-firstheld/queue_cohort_first2.log` | `43 --first2` (the cohort typology is a row of `fig:typeinit`) |
| `clustering/runs/multidim_first2_launch.sh` (launched 7 Oct 14:08; sleeps until the windows queue logs WINDOWS QUEUE COMPLETE, then runs alongside K = 4/5) | theta multidimensional ssm + mortality with the first two rows held out, people with at least five rows | `artifacts/multidim-health/queue_first2.log` | `43 --first2` picks it up automatically (`theta-ssm-mort-first2ho`) and drops the first-row placeholder row; then rerun `43 --first2`, rewrite the multidimensional bullet under `fig:typeinit` and remove its todo |
| `clustering/runs/multidim_health_k45_launch.sh` (after the first queue) | theta multidimensional ssm at K = 4, then K = 5 | `artifacts/multidim-health/queue_k45.log` | `39_paper_multidim.py --k` (`fig_multidim_trajectories_by_K.png`) |
| `clustering/runs/windows_launch.sh` (alongside, launched 6 Oct 16:17) | theta ssm refitted on the ten age windows of the K-means windows figure (window contracts `health_window_{lo}_{hi}_v1`) | `artifacts/health-windows/queue.log` | a Bayesian version of `fig:windows` (script to write: `26` logic on `bayes_fit` with the window contracts); replaces the K-means figure in the main text, K-means version to the appendix (todo at the windows bullet) |

K-means twins for the first-rows exercises (`22_paper_kmeans.py --first-held 1 --variants h`,
`--first-held 2`) were run on 5 October; the labels sit in
`artifacts/descriptives/paper_kmeans_labels.parquet` as `h_firstheld`,
`theta_first2held`, `h_first2held`.

**When each lands:** run the hook, look at the figure, then write the bullets.
The draft places are: the two-row multidimensional typology as the
multidimensional row of `fig:typeinit` (placeholder now); the K = 4/5
multidimensional rows as a figure in Section 4 next to `fig:multidim`, with the
same vertical axis across panels (a feedback item).

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
| frailty and log-frailty AR(1)+"spike" at K = 4 and 5 (`health-frailty-ssm-k4/5`, `health-logfrailty-ssm-k4/5`) | the two "not fitted" cells per row of `fig:returnsmeasures` | 4 fits, 10-30 h each (K = 3 frailty ssm took 9.5 h) | open; suggested after the windows queue, so as not to run three jobs at once |
| cohort-shift versions of the first-*one*-row held-out fits (theta, h) | would complete the type-from-initial typologies | 2 fits, 7-13 h | on ice (5 Oct) |
| h in the multidimensional model at K = 4 and 5 | the h rows of a multidimensional K comparison | 2 fits, 40-60 h, mode risk | on ice (5 Oct) |
| the single-channel returns on the first-row-held-out fit's own classes (`42 --firstho`) | would make the targeting figure self-contained on one fit | no fit needed, a script flag | open; expected to move the targeting numbers by a few hundred pounds at most |

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
| Windows (`fig:windows`): class composition by age as the window moves | Done (6 Oct): lower row of `fig:windows`, the worst type's share by age per window and per window's people. Within shared ages the by-age shares coincide with the lifecycle typology's; the worst type's share of people rises as the window moves later because the window's people are older, and only windows confined to 60+ relabel late decliners (21-25% at 65-75 against 18-19%) |
| Returns with targeting on observables or on theta(t-1) (6 Oct: Section 5's main figure is now `fig:prevention`, a 2 x 2 of the class returns and the bill beside the targeting panels, AR(1)+"spike" theta only; the detailed figures moved to Appendix `app:returns_detail`, and `app:returns_models` shows the return by class under six typologies and on the four measures) | Done (6 Oct): `fig:targeting` / `tab:targeting` after the returns table, from `46_paper_targeting.py`: rules on what is known at the first observation (age; + observables; + one theta reading; both; the model's posterior; the oracle), treating the top q by predicted return and scored by the realised return. On the first-two-rows fits (7 Oct): one theta reading closes 40% of the age-to-oracle gap at 10% coverage, observables a fifth, two readings 53%; the model's posterior from one or two rows matches the regressions; the h version is `fig:targeting_h` in the measures appendix. Open todo: net of a screening cost; targeting on the predicted class rather than the predicted return |
| Treatment as the cost prevented; when the savings come (done 7 Oct: panel (c) of `fig:prevention` and `fig:timing` in the appendix, with the half-way ages and the present values at 3.5%); LE and inequality effects of prevention | Treatment as the cost prevented: done (6 Oct). The tau instrument and T(r) are gone; the framework and Section 5 now price the treatment bill still to come C(r), split into the cost at the mean path, type differences and realised deviations (the unpreventable part, 16% from 30), and prevention as the share of the bill it avoids (P/C 0.91, 0.67, 0.39 from 30, 50, 70). The note's T(r) is 2C(r) under the quadratic. Timing of the savings and the LE / inequality effects of prevention: open |

## Open decisions in the draft (the red todos)

- Section 2: which noise class to lead with (the data pick Case 4 at its rho -> 1 corner), how treatment and prevention relate to the "spike", one definition of "type", selection, measurement error in the policy scenarios (lines 132, 173, 197, 214-227).
- Section 3.3: which Exhibit 2 cases to present, pairwise versus balanced moments (line 378).
- Section 4: a bounded observation model for h and frailty (line 550); expanding windows on the Bayesian fits (line 536); the posterior uncertainty of theta in the clustering (line 280).
- Appendix prose: the relation to PCS, more on the GRM, the h-versus-theta figure, the proofs appendix (lines 682-752).
- The PSID replication (line 248) and the introduction (line 31).

## Housekeeping

- Nothing is committed since `set off multidim` (25 September) apart from what the next checkpoint commit covers.
- `paper/paper_overleaf/` and `paper_overleaf.zip` are an 18 September copy for Overleaf and are stale; regenerate or drop before sharing.
- `measuring_health/health_measures_note.tex` is the archived measure-search note and describes the retired measures; it is kept as a document only.
- `artifacts/scratch/contract_backup_oct5/` is the health contract as it was before fi10 was dropped (same rows; only the extra column differs). Safe to delete once the October fits have been read.
