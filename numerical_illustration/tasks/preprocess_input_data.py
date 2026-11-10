import logging

import numpy as np
import pandas as pd

from ..schema import DataConfig

logger = logging.getLogger(__name__)


def preprocess_input_data(
    data_config: DataConfig,
    data: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    large_claim_quantile = getattr(data_config, "large_claim_quantile", None)
    if large_claim_quantile is not None:
        train_data, test_data = _remove_large_claims(
            train_data=train_data, test_data=test_data, quantile=large_claim_quantile
        )

    if data_config.normalize_features:
        train_data, test_data = _normalize_features(train_data, test_data, features)

    return train_data, test_data


def _remove_large_claims(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop observations whose ``y`` exceeds the *quantile* of ``y`` in the training data.

    The threshold is estimated on the training data only and applied to both sets.
    """
    threshold = train_data["y"].quantile(quantile)
    kept_train = train_data.loc[train_data["y"] <= threshold]
    kept_test = test_data.loc[test_data["y"] <= threshold]
    logger.info(
        f"Removing large claims above {threshold:.2f} ({quantile:.0%} training quantile): "
        f"{len(train_data) - len(kept_train)} train and "
        f"{len(test_data) - len(kept_test)} test observations dropped"
    )
    return kept_train, kept_test


def _normalize_features(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize numeric features using training-set statistics only.

    Categorical features (``pd.CategoricalDtype``) are skipped.

    Args:
        train_data: training data
        test_data: test data
        features: list of feature column names to consider
    """
    numeric_features = [
        f for f in features
        if not isinstance(train_data[f].dtype, pd.CategoricalDtype)
    ]
    if not numeric_features:
        return train_data, test_data
    mean = train_data[numeric_features].mean()
    std = train_data[numeric_features].std()
    train_data[numeric_features] = (train_data[numeric_features] - mean) / std
    test_data[numeric_features] = (test_data[numeric_features] - mean) / std
    return train_data, test_data


def _split_data(
    data: pd.DataFrame, test_size: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
