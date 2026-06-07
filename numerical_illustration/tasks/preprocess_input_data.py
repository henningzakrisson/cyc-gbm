from typing import List, Tuple

import numpy as np
import pandas as pd

from ..schema import DataConfig


def preprocess_input_data(
    data_config: DataConfig,
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

    train_data, test_data = _split_data(data, data_config.test_size, rng)

    if data_config.normalize_features:
        train_data, test_data = _normalize_features(train_data, test_data, features)

    return train_data, test_data


def _normalize_features(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize features using training-set statistics only.

    Args:
        train_data: training data
        test_data: test data
        features: list of feature column names to normalize
    """
    mean = train_data[features].mean()
    std = train_data[features].std()
    train_data[features] = (train_data[features] - mean) / std
    test_data[features] = (test_data[features] - mean) / std
    return train_data, test_data


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
