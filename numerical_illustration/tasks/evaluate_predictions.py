from collections import deque
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cyc_gbm.utils.distributions import initiate_distribution

from .utils.constants import DISTRIBUTION, MODELS


def evaluate_predictions(
    train_data: pd.DataFrame, test_data: pd.DataFrame, config: dict
) -> Tuple[pd.DataFrame, plt.Figure]:
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

    fig, ax = plt.subplots(2, 1)

    if "true" in model_names:
        data_sorted_mean = train_data.sort_values(by="theta_0").reset_index(drop=True)
    else:
        data_sorted_mean = train_data.sort_values(by="y").reset_index(drop=True)
    data_sorted_mean["y"].rolling(window=100, min_periods=1).mean().plot(
        ax=ax[0], label="True"
    )
    for model in model_names:
        if model == "true":
            data_sorted_mean["theta_0"].plot(ax=ax[0], label=model)
        else:
            data_sorted_mean[model + "_theta_0"].plot(ax=ax[0], label=model)

    if "true" in model_names:
        data_sorted_var = train_data.copy()
        data_sorted_var = train_data.sort_values(by="theta_1").reset_index(drop=True)
        (data_sorted_var["y"] - data_sorted_var["theta_0"]).pow(2).rolling(
            window=100, min_periods=1
        ).mean().plot(ax=ax[1], label="True")
    else:
        data_sorted_var = train_data.sort_values(by="y").reset_index(drop=True)
        (
            data_sorted_var["y"]
            - data_sorted_var["y"].rolling(window=100, min_periods=1).mean()
        ).pow(2).rolling(window=100, min_periods=1).mean().plot(ax=ax[1], label="True")

    for model in model_names:
        if model == "true":
            np.exp(data_sorted_var["theta_1"]).plot(ax=ax[1], label=model)
        else:
            np.exp(data_sorted_var[model + "_theta_1"]).plot(ax=ax[1], label=model)

    return metrics, fig


def _rolling_mean(y: np.ndarray, window: int = 100) -> np.ndarray:
    """
    Calculate the rolling mean of the input array.

    Args:
        y: input array
        window: window size for the rolling mean
    """
    return np.convolve(y, np.ones(window), "same") / window


def _rolling_var(y: np.ndarray, mu: np.ndarray, window: int = 100) -> np.ndarray:
    """
    Calculate the rolling variance of the input array.

    Args:
        y: input array
        mu: rolling mean
        window: window size for the rolling variance
    """
    return _rolling_mean((y - mu) ** 2, window=window)
