from collections import deque

import numpy as np
import pandas as pd

from cyc_gbm.utils.distributions import Distribution


def evaluate_predictions(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    distribution: Distribution,
    model_names: list[str],
    is_simulation: bool,
) -> pd.DataFrame:
    """Evaluate the predictions.

    Args:
        train_data: training data with prediction columns
        test_data: test data with prediction columns
        distribution: instantiated distribution object
        model_names: list of model class name strings
        is_simulation: whether the data was generated via simulation
            (if so, include the true parameter loss in the metrics)
    """
    n_dim = distribution.n_dim

    names = deque(model_names)
    if is_simulation:
        names.appendleft("true")

    metrics = pd.DataFrame(columns=["train", "test"], index=names)
    for data_set, data_name in zip([train_data, test_data], metrics.columns):
        for model_name in names:
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
            metrics.at[model_name, data_name] = distribution.loss(y=y, z=z, w=w).mean()

    return metrics
