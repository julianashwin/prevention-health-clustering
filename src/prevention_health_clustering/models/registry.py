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

from dataclasses import dataclass, field
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
    ar_mode: int = 0                   # 0 none | 1 AR(1) on the class residual
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

REGISTRY: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        PCS_HEADLINE,
        MCS_HEADLINE,
        JOINT_HEADLINE,
        PCS_AR1,
        PCS_COHORT,
        JOINT_AR1_COHORT,
        MIXED_HEALTH,
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
