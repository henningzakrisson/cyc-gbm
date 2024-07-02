from collections import deque
from typing import Tuple

import numpy as np
import pandas as pd

from cyc_gbm.utils.distributions import initiate_distribution

from .utils.constants import DISTRIBUTION, MODELS


def evaluate_predictions(
    train_data: pd.DataFrame, test_data: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """
    Evaluate the predictions.

    Args:
        predictions: predictions from the models
        config: configuration dictionary
    """
    # Evaluate the predictions
    distribution = initiate_distribution(config[DISTRIBUTION])
    n_dim = distribution.n_dim

    model_names = deque(config[MODELS])
    # Check if the true parameters are present
    if "theta_0" in train_data.columns:
        model_names.appendleft("true")

    metrics = pd.DataFrame(columns=["train", "test"], index=model_names)
    for data_set, data_name in zip([train_data, test_data], metrics.columns):
        for model_name in model_names:
            if model_name == "true":
                theta_cols = ["theta_" + str(i) for i in range(n_dim)]
            else:
                theta_cols = [
                    col
                    for col in data_set.columns
                    if col.startswith(model_name + "_theta_")
                ]

            y = data_set["y"].values
            w = data_set["w"].values
            z = data_set[theta_cols].values.T
            metrics.at[model_name, data_name] = distribution.loss(y=y, z=z, w=w).sum()

    return metrics
