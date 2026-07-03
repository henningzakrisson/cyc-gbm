"""Data configuration models."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from cyc_gbm.utils.distributions import Distribution, initiate_distribution

from .constants import DataSource


class DataConfig(BaseModel):
    """Shared data configuration inherited by all data source variants.

    Attributes:
        distribution: Name of the parametric distribution (e.g. "normal", "gamma").
            Must match a key accepted by ``initiate_distribution``.
        test_size: Fraction of the data reserved for the test set.
        normalize_features: Whether to z-score normalize features before fitting.
    """

    distribution: str
    parameterization: str = "mean-dispersion"
    test_size: float = 0.2
    normalize_features: bool = True

    @property
    def distribution_object(self) -> Distribution:
        """Instantiate and return the ``Distribution`` object."""
        return initiate_distribution(self.distribution, parameterization=self.parameterization)


class SimulationConfig(DataConfig):
    """Configuration for generating synthetic data via simulation.

    Attributes:
        data_source: Discriminator literal, always ``"simulation"``.
        random_seed: Seed for the random number generator used in simulation
            and train/test splitting.
        n_samples: Number of observations to simulate.
        n_features: Number of features (columns of X) to generate.
        parameter_function: Python source code defining a ``parameter(X)``
            function that maps the feature matrix to distribution parameters.
    """

    data_source: Literal[DataSource.SIMULATION]
    random_seed: int = 1
    n_samples: int
    n_features: int
    parameter_function: str


class LocalDataConfig(DataConfig):
    """Configuration for loading data from a local CSV file.

    Attributes:
        data_source: Discriminator literal, always ``"file"``.
        file_path: Path to the CSV file.
        random_seed: Seed for the random number generator used in
            train/test splitting.
        categorical_features: Column names to cast to
            ``pd.CategoricalDtype`` after loading.  These are then
            automatically picked up by ``CyclicalGradientBooster``.
    """

    data_source: Literal[DataSource.FILE]
    file_path: Path
    random_seed: int = 42
    categorical_features: list[str] = []
