from typing import Any, Dict

import numpy as np
import pandas as pd

from cyc_gbm.utils.distributions import initiate_distribution

from .baseline_models.intercept_model import InterceptModel

MODELS = "models"
INTERCEPT = "intercept"
DISTRIBUTION = "distribution"


def fit_models(
    config: Dict[str, Any], train_data: pd.DataFrame, rng: np.random.Generator
) -> Dict[str, Any]:
    """
    Fit the models specified in the config, using hyperparameters from the config.
    Uses the training data and the random number generator.
    Creates cross validation if the model training requires it.
    """
    distribution = initiate_distribution(config[DISTRIBUTION])

    model_names = config[MODELS]
    models = {}

    if INTERCEPT in model_names:
        models[INTERCEPT] = _fit_intercept_model(train_data, distribution)

    return models


def _fit_intercept_model(train_data: pd.DataFrame, distribution) -> InterceptModel:
    """
    Fit the intercept model.
    """
    intercept_model = InterceptModel(distribution=distribution)
    features = [
        col
        for col in train_data.columns
        if col not in ["y", "w"] or col.startswith("theta")
    ]
    intercept_model.fit(
        X=train_data[features].values,
        y=train_data["y"].values,
        w=train_data["w"].values,
    )
    return InterceptModel(train_data)
