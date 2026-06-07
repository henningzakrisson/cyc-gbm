"""Pydantic configuration models for the numerical illustration pipeline."""

from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator


# ── Data ──────────────────────────────────────────────────────────────────────


class DataConfig(BaseModel):
    """Base data configuration shared by all data sources."""

    distribution: str
    test_size: float = 0.2
    normalize_features: bool = True


class SimulationConfig(DataConfig):
    """Configuration for simulated data."""

    data_source: Literal["simulation"]
    random_seed: int = 1
    n_samples: int
    n_features: int
    parameter_function: str


class LocalDataConfig(DataConfig):
    """Configuration for loading data from a local file."""

    data_source: Literal["file"]
    file_path: Path
    random_seed: int = 42


# ── Output ────────────────────────────────────────────────────────────────────


class DumpingConfig(BaseModel):
    """Configuration for output / result storage."""

    output_dir: Path = Path("data/results")


# ── Models ────────────────────────────────────────────────────────────────────


class GradientBoostingMachineConfig(BaseModel):
    """Hyperparameters for the (non-cyclical) gradient boosting machine."""

    model_class: Literal["gbm"]
    n_estimators: int = 600
    max_depth: int = 3
    learning_rate: float = 0.05


class CyclicalGradientBoostingMachineConfig(BaseModel):
    """Hyperparameters for the cyclical gradient boosting machine."""

    model_class: Literal["cgbm"]
    n_estimators: list[int] = Field(default_factory=lambda: [500, 500])
    max_depth: int = 3
    learning_rate: Union[float, list[float]] = 0.05


class NaturalGradientBoostingMachineConfig(BaseModel):
    """Hyperparameters for the natural gradient boosting machine (NGBoost)."""

    model_class: Literal["ngboost"]
    n_estimators: int = 600
    max_depth: int = 3
    learning_rate: float = 0.05


class CyclicalGeneralizedLinearModelConfig(BaseModel):
    """Hyperparameters for the cyclical generalized linear model."""

    model_class: Literal["cglm"]
    max_iter: int = 2000
    tolerance: float = 1e-5
    step_size: float = 0.1


class InterceptConfig(BaseModel):
    """Configuration for the intercept-only baseline model."""

    model_class: Literal["intercept"]


ModelConfig = Annotated[
    Union[
        GradientBoostingMachineConfig,
        CyclicalGradientBoostingMachineConfig,
        NaturalGradientBoostingMachineConfig,
        CyclicalGeneralizedLinearModelConfig,
        InterceptConfig,
    ],
    Field(discriminator="model_class"),
]


# ── Bootstrap ─────────────────────────────────────────────────────────────────


class BootstrapConfig(BaseModel):
    """Configuration for bootstrap iterations."""

    n_bootstraps: int = 1
    parallel: bool = False
    n_jobs: int = -1


# ── Tuning ────────────────────────────────────────────────────────────────────


class TuningConfig(BaseModel):
    """Configuration for hyperparameter tuning."""

    perform_tuning: bool = False
    n_splits: int = 4


# ── Root ──────────────────────────────────────────────────────────────────────


class NumericalIllustrationConfig(BaseModel):
    """Top-level configuration for the numerical illustration pipeline."""

    data: Annotated[
        Union[SimulationConfig, LocalDataConfig],
        Field(discriminator="data_source"),
    ]
    output: DumpingConfig = Field(default_factory=DumpingConfig)
    models: list[ModelConfig]
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)

    @model_validator(mode="after")
    def no_bootstrap_with_local_data(self) -> "NumericalIllustrationConfig":
        if (
            self.bootstrap.n_bootstraps > 1
            and isinstance(self.data, LocalDataConfig)
        ):
            raise ValueError(
                "Bootstrap (n_bootstraps > 1) is not supported with local file "
                "data source. Use simulation or set n_bootstraps to 1."
            )
        return self
