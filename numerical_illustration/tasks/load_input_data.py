import numpy as np
import pandas as pd

from cyc_gbm.utils.distributions import initiate_distribution

DATA_SOURCE = "data_source"
SIMULATION = "simulation"
FILE = "file"
N_SAMPLES = "n_samples"
N_FEATURES = "n_features"
DISTRIBUTION = "distribution"
PARAMETER_FUNCTION = "parameter_function"
FILE_PATH = "file_path"
PARAMETER = "parameter"


def load_input_data(config: dict, rng: np.random.Generator) -> pd.DataFrame:
    """
    Load or create the data.

    Args:
        config: configuration dictionary
        rng: random number generator (used for simulation and train-test split)
    """
    data_source = config[DATA_SOURCE]
    if data_source == SIMULATION:
        # Simulate the data
        return _simulate_data(config, rng)
    elif data_source == FILE:
        # Load the data from a file
        return _load_data_from_file(config)
    else:
        raise ValueError(f"Unknown data source: {data_source}")


def _simulate_data(config: dict, rng: np.random.Generator) -> pd.DataFrame:
    """
    Load the simulation data.

    Args:
        config: configuration dictionary
        rng: random number generator
    """
    # Load neccessary simulation metadata
    n_samples = config[N_SAMPLES]
    n_features = config[N_FEATURES]
    distribution = initiate_distribution(config[DISTRIBUTION])
    local_vars = {}
    parameter_function = _compile_function_from_string(config[PARAMETER_FUNCTION])

    # Simulate data
    X = rng.normal(size=(n_samples, n_features))
    theta = parameter_function(X)
    w = np.ones(n_samples)
    y = distribution.simulate(z=theta, w=w, rng=rng)

    data = pd.DataFrame(X, columns=[f"X_{i}" for i in range(n_features)])
    data["y"] = y
    data["w"] = w
    theta_dim = theta.shape[0]
    data[[f"theta_{i}" for i in range(theta_dim)]] = theta.T
    return data


def _compile_function_from_string(function_string: str) -> callable:
    """
    Compile a function from a string.

    Args:
        function_string: string representation of the function
    """
    local_vars = {}
    exec(function_string, globals(), local_vars)
    return local_vars[PARAMETER]


def _load_data_from_file(config: dict) -> pd.DataFrame:
    """
    Load the data from a file.

    Args:
        config: configuration dictionary
    """
    # Load neccessary file metadata
    file_path = config[FILE_PATH]

    data = pd.read_csv(file_path)
    # NOTE: Asssumed schema is:
    # y: target variable
    # w: weights (optional)
    # other columns: features
    if "w" not in data.columns:
        data["w"] = 1

    return data
