from typing import Tuple, Union

import numpy as np
import pandas as pd

from ..schema import LocalDataConfig, SimulationConfig


def preprocess_input_data(
    data_config: Union[SimulationConfig, LocalDataConfig],
    data: pd.DataFrame,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess the input data.

    Args:
        data_config: data configuration (used for normalize_features and test_size)
        data: input data
        rng: random number generator
    """
    features = [
        col
        for col in data.columns
        if col not in ["w", "y"] and not col.startswith("theta")
    ]

    if data_config.normalize_features:
        data[features] = _normalize_data(data[features])

    train_data, test_data = _split_data(data, data_config.test_size, rng)

    return train_data, test_data


def _normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize the data.

    Args:
        data: input data
    """
    return (data - data.mean()) / data.std()


def _split_data(
    data: pd.DataFrame, test_size: float, rng: np.random.Generator
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into training and test data.

    Args:
        data: input data
        test_size: size of the test data
        rng: random number generator
    """
    n_test = int(test_size * data.shape[0])
    test_indices = rng.choice(data.index, size=n_test, replace=False)
    test_data = data.loc[test_indices]
    train_data = data.drop(test_indices)
    return train_data, test_data
