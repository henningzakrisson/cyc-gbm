"""Root configuration model that composes all sub-configs."""

from typing import Annotated, Self, Union

from pydantic import BaseModel, Field, model_validator

from .bootstrap import BootstrapConfig
from .data import LocalDataConfig, SimulationConfig
from .models import ModelConfig
from .output import DumpingConfig
from .tuning import TuningConfig


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
        Union[SimulationConfig, LocalDataConfig],
        Field(discriminator="data_source"),
    ]
    output: DumpingConfig = Field(default_factory=DumpingConfig)
    models: list[ModelConfig]
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)

    @model_validator(mode="after")
    def no_bootstrap_with_local_data(self) -> Self:
        if (
            self.bootstrap.n_bootstraps > 1
            and isinstance(self.data, LocalDataConfig)
        ):
            raise ValueError(
                "Bootstrap (n_bootstraps > 1) is not supported with local file "
                "data source. Use simulation or set n_bootstraps to 1."
            )
        return self
