"""The model registry: one source of truth for model specifications.

The predecessor project spread model identity across seven parallel
name-keyed structures — 88 ``if model ==`` branches in ``bayesian/paths.py``,
246 model-name references in ``bayesian/ar1.py``, and 30 ``uses_*_model()``
predicates in the R runner. Missing one branch failed at run time, not parse
time.

Here a model is a dataclass. Adding a variant means adding a ``ModelSpec``, and
the Stan data payload is derived from it mechanically.
"""

from __future__ import annotations

from dataclasses import replace, dataclass, field
from pathlib import Path

STAN_DIR = Path(__file__).resolve().parent / "stan"

GAUSSIAN_PANEL = STAN_DIR / "mixture_gaussian_panel.stan"
MIXED_PANEL = STAN_DIR / "mixture_mixed_panel.stan"


@dataclass(frozen=True)
class ModelSpec:
    """A fully specified model. Everything the runner needs, in one object."""

    name: str
    stan_file: Path
    description: str

    # Structure
    n_classes: int = 3
    channels: tuple[str, ...] = ("sf12pcs_dv",)
    anchor_channel: str = "sf12pcs_dv"
    design: str = "quadratic"          # quadratic | linear
    ar_mode: int = 0                   # 0 none | 1 AR(1) | 2 AR(1) + meas. error
    homosigma: bool = True
    cohort: str = "none"               # none | decade
    cohort_by_class: bool = False
    standardize: bool = True
    person_weight_power: float = 0.0

    # Priors. Scales are on the standardised scale when standardize is True.
    alpha_prior_scale: float = 1.5
    slope_prior_scales: tuple[float, ...] = (1.0, 0.05)
    sigma_prior_location: float = 0.7
    sigma_prior_scale: float = 0.5
    theta_prior_concentration: float = 1.5
    cohort_prior_scale: float = 0.15
    rho_prior_alpha: float = 2.0
    rho_prior_beta: float = 2.0
    # Half-normal on the measurement-error scale; read only when ar_mode == 2.
    # Centred well below the residual scale: the default should be that most
    # variation is signal, and the data have to argue for noise.
    sigma_meas_prior_location: float = 0.0
    sigma_meas_prior_scale: float = 0.4

    # Sampling
    chains: int = 4
    iter_warmup: int = 1000
    iter_sampling: int = 1000
    adapt_delta: float = 0.95
    max_treedepth: int = 12
    seed: int = 20260720

    metadata: dict = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def design_width(self) -> int:
        """Number of design columns, including the intercept."""
        return {"linear": 2, "quadratic": 3}[self.design]

    @property
    def anchor_index(self) -> int:
        """One-based index of the anchor channel, as Stan expects."""
        if self.anchor_channel not in self.channels:
            raise ValueError(
                f"Anchor channel {self.anchor_channel!r} is not among the fitted "
                f"channels {self.channels}. Ordering a channel that carries no "
                "likelihood weight leaves class labels exchangeable."
            )
        return self.channels.index(self.anchor_channel) + 1

    def validate(self) -> None:
        _ = self.anchor_index
        if len(self.slope_prior_scales) != self.design_width - 1:
            raise ValueError(
                f"design {self.design!r} needs {self.design_width - 1} slope "
                f"prior scales, got {len(self.slope_prior_scales)}"
            )
        if self.n_classes < 1:
            raise ValueError("n_classes must be at least 1")


# ---------------------------------------------------------------------------
# Registered models
# ---------------------------------------------------------------------------

PCS_HEADLINE = ModelSpec(
    name="pcs-headline",
    stan_file=GAUSSIAN_PANEL,
    description=(
        "PCS-only K=3 quadratic growth mixture, shared residual scale, no AR "
        "term. Matches the published ar1-mixture-residual-rho0-homosigma fit."
    ),
    channels=("sf12pcs_dv",),
    anchor_channel="sf12pcs_dv",
)

MCS_HEADLINE = ModelSpec(
    name="mcs-headline",
    stan_file=GAUSSIAN_PANEL,
    description="MCS-only K=3, anchored on MCS itself.",
    channels=("sf12mcs_dv",),
    anchor_channel="sf12mcs_dv",
)

JOINT_HEADLINE = ModelSpec(
    name="joint-headline",
    stan_file=GAUSSIAN_PANEL,
    description="Joint PCS+MCS K=3 with one shared latent class, PCS-anchored.",
    channels=("sf12pcs_dv", "sf12mcs_dv"),
    anchor_channel="sf12pcs_dv",
    # The published joint fit used concentration 0.5 where the univariate fits
    # used 1.5. Negligible against 50k people, but kept for faithfulness.
    theta_prior_concentration=0.5,
)

PCS_AR1 = ModelSpec(
    name="pcs-ar1",
    stan_file=GAUSSIAN_PANEL,
    description="PCS-only K=3 with AR(1) persistence in the class residual.",
    channels=("sf12pcs_dv",),
    anchor_channel="sf12pcs_dv",
    ar_mode=1,
)

PHYSGRM_HEADLINE = ModelSpec(
    name="physgrm-headline",
    stan_file=GAUSSIAN_PANEL,
    description=(
        "Physical GRM (P-FULL theta: functioning + condition groups) K=3 "
        "quadratic growth mixture, no AR term."
    ),
    channels=("theta_phys_full",),
    anchor_channel="theta_phys_full",
)

PHYSGRM_AR1 = ModelSpec(
    name="physgrm-ar1",
    stan_file=GAUSSIAN_PANEL,
    description="Physical GRM (P-FULL theta) K=3 with AR(1) persistence.",
    channels=("theta_phys_full",),
    anchor_channel="theta_phys_full",
    ar_mode=1,
)

COMBGRM_HEADLINE = ModelSpec(
    name="combgrm-headline",
    stan_file=GAUSSIAN_PANEL,
    description=(
        "Combined physical+mental GRM theta K=3 quadratic growth mixture, "
        "no AR term. NOTE the combined bank's known unidimensionality caveat "
        "(docs/health_measures_note.pdf section 6)."
    ),
    channels=("theta_combined",),
    anchor_channel="theta_combined",
)

COMBGRM_AR1 = ModelSpec(
    name="combgrm-ar1",
    stan_file=GAUSSIAN_PANEL,
    description="Combined GRM theta K=3 with AR(1) persistence.",
    channels=("theta_combined",),
    anchor_channel="theta_combined",
    ar_mode=1,
)

PCS_COHORT = ModelSpec(
    name="pcs-cohort",
    stan_file=GAUSSIAN_PANEL,
    description="PCS-only K=3 with shared birth-decade level shifts.",
    channels=("sf12pcs_dv",),
    anchor_channel="sf12pcs_dv",
    cohort="decade",
)

MIXED_HEALTH = ModelSpec(
    name="mixed-health",
    stan_file=MIXED_PANEL,
    description=(
        "Five-channel mixed-outcome model: PCS, MCS (Gaussian), self-rated "
        "health (ordered logit), chronic count (neg. binomial), ADL "
        "(hurdle neg. binomial). One shared latent class, PCS-anchored, "
        "class 1 = worst physical health. No survival channel (five-channel "
        "family); no tanh caps, unlike the quarantined predecessor fits."
    ),
    channels=("sf12pcs_dv", "sf12mcs_dv"),  # gaussian block; other channels fixed
    anchor_channel="sf12pcs_dv",
    theta_prior_concentration=0.5,
)

JOINT_AR1_COHORT = ModelSpec(
    name="joint-ar1-cohort",
    stan_file=GAUSSIAN_PANEL,
    description="Joint PCS+MCS with AR(1) persistence and birth-decade shifts.",
    channels=("sf12pcs_dv", "sf12mcs_dv"),
    anchor_channel="sf12pcs_dv",
    ar_mode=1,
    cohort="decade",
)

# ---------------------------------------------------------------------------
# AR(1) state plus one-period measurement error (ar_mode = 2)
#
# Under ar_mode 1 the observation is the state, so rho absorbs both true
# persistence and any transient noise, and is biased towards zero whenever the
# instrument is noisy. Separating them asks how much of the year-to-year
# movement in a GRM score is real change in health and how much is the
# measurement. These three fits put the same question to the original GRM and
# to both physical variants.
# ---------------------------------------------------------------------------

GRM_SSM = ModelSpec(
    name="grm-ssm",
    stan_file=GAUSSIAN_PANEL,
    description=(
        "Original GRM theta, K=3 quadratic growth mixture, AR(1) latent state "
        "plus i.i.d. measurement error, marginalised by Kalman filter."
    ),
    channels=("grm_theta",),
    anchor_channel="grm_theta",
    ar_mode=2,
)

PHYSGRM_FUNC_SSM = ModelSpec(
    name="physgrm-func-ssm",
    stan_file=GAUSSIAN_PANEL,
    description=(
        "P-FUNC theta (functioning items only), K=3, AR(1) state plus "
        "measurement error."
    ),
    channels=("theta_phys_func",),
    anchor_channel="theta_phys_func",
    ar_mode=2,
)

PHYSGRM_FULL_SSM = ModelSpec(
    name="physgrm-full-ssm",
    stan_file=GAUSSIAN_PANEL,
    description=(
        "P-FULL theta (functioning plus diagnosed conditions), K=3, AR(1) "
        "state plus measurement error."
    ),
    channels=("theta_phys_full",),
    anchor_channel="theta_phys_full",
    ar_mode=2,
)


# K = 4 under the AR(1) plus measurement error specification. Everything else matches the K = 3
# fits, so the only difference is the number of classes. More classes means
# more adjacent pairs the ordered anchor intercept has to separate, which is
# the configuration that produced a mode split in the multidim smoke, so
# these are watched rather than assumed.

PHYSGRM_FUNC_SSM_K4 = ModelSpec(
    name="physgrm-func-ssm-k4",
    stan_file=GAUSSIAN_PANEL,
    description="P-FUNC theta, K=4, AR(1) state plus measurement error.",
    channels=("theta_phys_func",),
    anchor_channel="theta_phys_func",
    ar_mode=2,
    n_classes=4,
)

PHYSGRM_FULL_SSM_K4 = ModelSpec(
    name="physgrm-full-ssm-k4",
    stan_file=GAUSSIAN_PANEL,
    description="P-FULL theta, K=4, AR(1) state plus measurement error.",
    channels=("theta_phys_full",),
    anchor_channel="theta_phys_full",
    ar_mode=2,
    n_classes=4,
)

# K = 5 for P-FUNC under the AR(1) plus measurement error specification. Identical to the K=4
# fit apart from the class count, on the same contract and people.

PHYSGRM_FUNC_SSM_K5 = ModelSpec(
    name="physgrm-func-ssm-k5",
    stan_file=GAUSSIAN_PANEL,
    description="P-FUNC theta, K=5, AR(1) state plus measurement error.",
    channels=("theta_phys_func",),
    anchor_channel="theta_phys_func",
    ar_mode=2,
    n_classes=5,
)


# The paper's health measure (P-LIM3+CC), baseline K=3 on each reported
# variant: independent residuals, no AR term, on the one contract that carries
# all three, so the fits differ by the variant alone.

HEALTH_THETA_BASE = ModelSpec(
    name="health-theta-base",
    stan_file=GAUSSIAN_PANEL,
    description="Health measure theta, K=3 quadratic growth mixture, no AR term.",
    channels=("theta",),
    anchor_channel="theta",
)

HEALTH_H_BASE = ModelSpec(
    name="health-h-base",
    stan_file=GAUSSIAN_PANEL,
    description="Health measure h (expected score, 0-1), K=3, no AR term.",
    channels=("h",),
    anchor_channel="h",
)

HEALTH_FI10_BASE = ModelSpec(
    name="health-fi10-base",
    stan_file=GAUSSIAN_PANEL,
    description="Health measure ten-deficit index, K=3, no AR term.",
    channels=("fi10",),
    anchor_channel="fi10",
)


# The same three variants under AR(1) plus measurement error: an AR(1) latent
# state plus a one-period measurement error, the paper's stochastic specification.

HEALTH_THETA_SSM = ModelSpec(
    name="health-theta-ssm",
    stan_file=GAUSSIAN_PANEL,
    description="Health measure theta, K=3, AR(1) state plus measurement error.",
    channels=("theta",),
    anchor_channel="theta",
    ar_mode=2,
)

HEALTH_H_SSM = ModelSpec(
    name="health-h-ssm",
    stan_file=GAUSSIAN_PANEL,
    description="Health measure h, K=3, AR(1) state plus measurement error.",
    channels=("h",),
    anchor_channel="h",
    ar_mode=2,
)

HEALTH_FI10_SSM = ModelSpec(
    name="health-fi10-ssm",
    stan_file=GAUSSIAN_PANEL,
    description="Health measure ten-deficit index, K=3, AR(1) state plus measurement error.",
    channels=("fi10",),
    anchor_channel="fi10",
    ar_mode=2,
)


# The AR(1) plus measurement error fits with a shared birth-decade level shift (the same
# cohort intercept for every class), as pcs-cohort did on the baseline.

HEALTH_THETA_SSM_COHORT = ModelSpec(
    name="health-theta-ssm-cohort",
    stan_file=GAUSSIAN_PANEL,
    description="Health measure theta, K=3, AR(1) plus measurement error, shared birth-decade shifts.",
    channels=("theta",),
    anchor_channel="theta",
    ar_mode=2,
    cohort="decade",
)

HEALTH_H_SSM_COHORT = ModelSpec(
    name="health-h-ssm-cohort",
    stan_file=GAUSSIAN_PANEL,
    description="Health measure h, K=3, AR(1) plus measurement error, shared birth-decade shifts.",
    channels=("h",),
    anchor_channel="h",
    ar_mode=2,
    cohort="decade",
)

HEALTH_FI10_SSM_COHORT = ModelSpec(
    name="health-fi10-ssm-cohort",
    stan_file=GAUSSIAN_PANEL,
    description="Health measure ten-deficit index, K=3, AR(1) plus measurement error, shared birth-decade shifts.",
    channels=("fi10",),
    anchor_channel="fi10",
    ar_mode=2,
    cohort="decade",
)


# AR(1) on the observation itself (ar_mode 1), the specification the paper
# argues against: with no measurement-error term, rho must absorb both genuine
# persistence and instrument noise. These three exist to test that attenuation
# on the paper's own measure rather than on the archived one.

HEALTH_THETA_AR1 = ModelSpec(
    name="health-theta-ar1",
    stan_file=GAUSSIAN_PANEL,
    description="Health measure theta, K=3, AR(1) in the class residual, no measurement error.",
    channels=("theta",),
    anchor_channel="theta",
    ar_mode=1,
)

HEALTH_H_AR1 = ModelSpec(
    name="health-h-ar1",
    stan_file=GAUSSIAN_PANEL,
    description="Health measure h, K=3, AR(1) in the class residual, no measurement error.",
    channels=("h",),
    anchor_channel="h",
    ar_mode=1,
)

HEALTH_FI10_AR1 = ModelSpec(
    name="health-fi10-ar1",
    stan_file=GAUSSIAN_PANEL,
    description="Health measure ten-deficit index, K=3, AR(1), no measurement error.",
    channels=("fi10",),
    anchor_channel="fi10",
    ar_mode=1,
)


# K = 4 and K = 5 on the paper's health measure, theta and h, under independent
# residuals and under AR(1) plus measurement error. Each is its K = 3 spec with
# only the class count changed, so the comparison across K is clean.

def _with_k(spec: ModelSpec, k: int, tag: str) -> ModelSpec:
    return replace(spec, name=f"{spec.name}-k{k}", n_classes=k,
                   description=spec.description.replace("K=3", f"K={k}") + f" ({tag})")


HEALTH_K45 = tuple(
    _with_k(spec, k, "class-count comparison")
    for k in (4, 5)
    for spec in (HEALTH_THETA_BASE, HEALTH_H_BASE, HEALTH_THETA_SSM, HEALTH_H_SSM)
)


REGISTRY: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        PCS_HEADLINE,
        MCS_HEADLINE,
        JOINT_HEADLINE,
        PCS_AR1,
    PHYSGRM_HEADLINE,
    PHYSGRM_AR1,
    COMBGRM_HEADLINE,
    COMBGRM_AR1,
        PCS_COHORT,
        JOINT_AR1_COHORT,
        MIXED_HEALTH,
        GRM_SSM,
        PHYSGRM_FUNC_SSM,
        PHYSGRM_FULL_SSM,
        PHYSGRM_FUNC_SSM_K4,
        PHYSGRM_FULL_SSM_K4,
        PHYSGRM_FUNC_SSM_K5,
        HEALTH_THETA_BASE,
        HEALTH_H_BASE,
        HEALTH_FI10_BASE,
        HEALTH_THETA_SSM,
        HEALTH_H_SSM,
        HEALTH_FI10_SSM,
        HEALTH_THETA_SSM_COHORT,
        HEALTH_H_SSM_COHORT,
        HEALTH_FI10_SSM_COHORT,
        HEALTH_THETA_AR1,
        HEALTH_H_AR1,
        HEALTH_FI10_AR1,
        *HEALTH_K45,
    )
}


def get_model(name: str) -> ModelSpec:
    try:
        spec = REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Unknown model {name!r}. Registered: {known}") from None
    spec.validate()
    return spec


__all__ = ["REGISTRY", "ModelSpec", "get_model"]
