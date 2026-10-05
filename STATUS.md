# Status

Last updated 5 October 2026. The draft is `paper/paper_skeleton.tex` (68 pages,
builds clean); red `\todo` marks in the PDF are open decisions, grey `\note`
marks are drafting notes.

## Fits running or queued (5 October)

Three detached queues, two at a time (about 80% of the machine); each skips
finished fits, so a launcher can be rerun after an interruption. Logs are in
the output directory named.

| Queue | Fits, in order | Log | Hook that reads them |
|---|---|---|---|
| `clustering/runs/frailty_firstheld_launch.sh` | frailty base (done), frailty ssm, logfrailty base, logfrailty ssm; then h base/ssm with the first row held out; then theta and h base/ssm with the first two rows held out | `artifacts/health-frailty/queue.log` | `32_paper_bayes_fits.py combined --frailty` / `--logfrailty` (figure drawn on the base fit already; the ssm column fills in); `43_paper_type_from_initial.py --h`, `--first2`, `--first2 --h` |
| `clustering/runs/cohort_first2_launch.sh` (alongside) | theta and h ssm-cohort with the first two rows held out | `artifacts/health-firstheld/queue_cohort_first2.log` | `43 --first2` (the fourth typology) |
| `clustering/runs/multidim_health_k45_launch.sh` (after the first queue) | theta multidimensional ssm at K = 4, then K = 5 | `artifacts/multidim-health/queue_k45.log` | `39_paper_multidim.py --k` (`fig_multidim_trajectories_by_K.png`) |

K-means twins for the first-rows exercises (`22_paper_kmeans.py --first-held 1 --variants h`,
`--first-held 2`) were run on 5 October; the labels sit in
`artifacts/descriptives/paper_kmeans_labels.parquet` as `h_firstheld`,
`theta_first2held`, `h_first2held`.

**When each lands:** run the hook, look at the figure, then write the bullets.
The draft places are: the frailty mixtures in Appendix `app:measures` next to
`fig:bayes_h` (and the main-text bullet "We have not yet fit the mixtures on
the frailty indices" goes); the first-two-rows and h versions of the
type-from-initial exercise next to `fig:typeinit`; the K = 4/5
multidimensional rows as a figure in Section 4 next to `fig:multidim`, with the
same vertical axis across panels (a feedback item).

On ice by decision (5 October): cohort-shift versions of the first-*one*-row
fits; h at K = 4/5 in the multidimensional model; Bayesian expanding windows.

## Feedback of 29 September (`paper/feedback_29sept.txt`)

| Item | Status |
|---|---|
| Predict the type from an initial, held-out health value, baseline and full clustering | Done: `fig:typeinit`, K-means / independent / AR(1)+"spike" / multidimensional; the first-two-rows and h versions follow the running fits |
| Relabel "spike" as "episodic" | Open (a global rename in the draft and the figure footers; `descriptives/` prints "spike" in about 30 places) |
| Is the C_Hd reversal driven by the small very-ill group? | Open; the bottom-decile and worst-class composition figures exist (`fig:decomp`, `fig:multidim`) but the statement has not been made |
| Result 2: focus on the by-type form and why the evolution of the mean matters | Open (Section 2 rewrite; see the framework todos below) |
| Same vertical axis across K panels | Open for `fig_class_trajectories_by_K` and the coming multidimensional K figure |
| More discrimination in the unhealthy type as K rises and as the stochastic structure is added? | Partly: the K = 4/5 single-channel rows are in; the multidimensional K = 4/5 fits are queued |
| Is there always a bad group, and does K-means understate it because the process is noisy? | Open; the ingredients are `fig:bayes` (dashed K-means lines) and the composition strips |
| Express type differences as LE / HLE | Open; the Gompertz-Makeham hazards by class in `fig:multidim` give LE directly, HLE needs a health threshold |
| Univariate mental-health clusters versus the multidimensional classes | Done: Appendix `app:mental` |
| Windows (`fig:windows`): class composition by age as the window moves | Open; `26_paper_windows.py` has the labels, the composition panel is not drawn |
| Returns (`fig:returnsbayes`) with targeting on observables or on theta(t-1) | Open; `42_paper_returns_bayes.py` has the class-level returns, targeting needs a rule |
| Treatment as the cost prevented; when the savings come; LE and inequality effects of prevention | Open (Section 5) |

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
