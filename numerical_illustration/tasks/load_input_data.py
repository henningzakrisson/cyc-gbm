import numpy as np
import pandas as pd

from cyc_gbm.utils.distributions import initiate_distribution

from ..config.config_models import (
    LocalDataConfig,
    NumericalIllustrationConfig,
    SimulationConfig,
)


def load_input_data(
    config: NumericalIllustrationConfig, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Load or create the data.

    Args:
        config: validated pipeline configuration
        rng: random number generator (used for simulation and train-test split)
    """
    if isinstance(config.data, SimulationConfig):
        return _simulate_data(config.data, rng)
    elif isinstance(config.data, LocalDataConfig):
        return _load_data_from_file(config.data)
    else:
        raise ValueError(f"Unknown data source type: {type(config.data)}")


def _simulate_data(data_config: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    """
    Simulate data according to the configuration.

    Args:
        data_config: simulation configuration
        rng: random number generator
    """
    distribution = initiate_distribution(data_config.distribution)
    parameter_function = _compile_function_from_string(data_config.parameter_function)

    X = rng.normal(size=(data_config.n_samples, data_config.n_features))
    theta = parameter_function(X)
    w = np.ones(data_config.n_samples)
    y = distribution.simulate(z=theta, w=w, rng=rng)

    data = pd.DataFrame(X, columns=[f"X_{i}" for i in range(data_config.n_features)])
    data["y"] = y
    data["w"] = w
    theta_dim = theta.shape[0]
    for i in range(theta_dim):
        data[f"theta_{i}"] = theta[i]

    return data


def _compile_function_from_string(function_string: str) -> callable:
    """
    Compile a function from a string.

    Args:
        function_string: string representation of the function
    """
    local_vars = {}
    exec(function_string, globals(), local_vars)
    return local_vars["parameter"]


def _load_data_from_file(data_config: LocalDataConfig) -> pd.DataFrame:
    """
    Load the data from a file.

    Args:
        data_config: local data configuration
    """
    data = pd.read_csv(data_config.file_path)
    # NOTE: Assumed schema is:
    # y: target variable
    # w: weights (optional)
    # other columns: features
    if "w" not in data.columns:
        data["w"] = 1

    return data
