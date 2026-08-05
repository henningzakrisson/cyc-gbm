"""Root configuration model that composes all sub-configs."""

from typing import Annotated, Self

from pydantic import BaseModel, Field, model_validator

from cyc_gbm.utils.distributions import Distribution, initiate_distribution

from .bootstrap import BootstrapConfig
from .constants import ModelClass
from .data import LocalDataConfig, SimulationConfig
from .models import ModelConfig
from .output import DumpingConfig
from .tuning import TuningConfig


def _resolve_model_name(model_config: ModelConfig) -> str:
    """Return the effective display name for a model config."""
    return model_config.name or model_config.model_class


class NumericalIllustrationConfig(BaseModel):
    """Top-level configuration for the numerical illustration pipeline.

    Attributes:
        data: Data source configuration (simulation or local file).
        output: Output directory configuration.
        models: List of model configurations to fit and evaluate.
        bootstrap: Bootstrap iteration settings.
        tuning: Cross-validation tuning settings.
    """

    data: Annotated[
        SimulationConfig | LocalDataConfig,
        Field(discriminator="data_source"),
    ]
    output: DumpingConfig = Field(default_factory=DumpingConfig)
    models: list[ModelConfig]
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)

    @property
    def model_names(self) -> list[str]:
        """List of resolved model name strings."""
        return [_resolve_model_name(m) for m in self.models]

    @property
    def model_configs_by_name(self) -> dict[str, ModelConfig]:
        """Lookup of model configs keyed by their resolved name."""
        return {_resolve_model_name(m): m for m in self.models}

    def resolve_distribution(self, model_config: ModelConfig) -> Distribution:
        """Return the distribution for a model, respecting per-model overrides."""
        parameterization = (
            model_config.parameterization or self.data.parameterization
        )
        return initiate_distribution(
            self.data.distribution, parameterization=parameterization
        )

    @property
    def distributions_by_model_name(self) -> dict[str, Distribution]:
        """Mapping from resolved model name to its effective distribution."""
        return {
            _resolve_model_name(m): self.resolve_distribution(m)
            for m in self.models
        }

    @model_validator(mode="after")
    def no_bootstrap_with_local_data(self) -> Self:
        if self.bootstrap.n_bootstraps > 1 and isinstance(self.data, LocalDataConfig):
            raise ValueError(
                "Bootstrap (n_bootstraps > 1) is not supported with local file "
                "data source. Use simulation or set n_bootstraps to 1."
            )
        return self

    @model_validator(mode="after")
    def unique_model_names(self) -> Self:
        names = self.model_names
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(
                f"Duplicate model names: {set(duplicates)}. "
                "Use the 'name' field to disambiguate models of the same class."
            )
        return self
