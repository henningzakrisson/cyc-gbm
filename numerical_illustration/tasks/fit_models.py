from typing import Any, Dict

import numpy as np
import pandas as pd

from cyc_gbm import CyclicalGradientBooster
from cyc_gbm.utils.distributions import initiate_distribution

from .baseline_models import CyclicGeneralizedLinearModel, InterceptModel

MODELS = "models"
INTERCEPT = "intercept"
DISTRIBUTION = "distribution"
CGLM = "cglm"
MODEL_HYPERPARAMS = "model_hyperparameters"
MAX_ITER = "max_iter"
TOLERANCE = "tolerance"
STEP_SIZE = "step_size"


def fit_models(
    config: Dict[str, Any], train_data: pd.DataFrame, rng: np.random.Generator
) -> Dict[str, Any]:
    """
    Fit the models specified in the config, using hyperparameters from the config.
    Uses the training data and the random number generator.
    Creates cross validation if the model training requires it.
    """
    # Initiate distribution and get train data
    distribution = initiate_distribution(config[DISTRIBUTION])
    X_train, y_train, w_train = _get_train_data(train_data)

    # Add models
    models = {}
    if INTERCEPT in config[MODELS]:
        models[INTERCEPT] = InterceptModel(distribution=distribution)
    if CGLM in config[MODELS]:
        models[CGLM] = CyclicGeneralizedLinearModel(
            distribution=distribution,
            max_iter=int(config[MODEL_HYPERPARAMS][CGLM][MAX_ITER]),
            tol=float(config[MODEL_HYPERPARAMS][CGLM][TOLERANCE]),
            eps=float(config[MODEL_HYPERPARAMS][CGLM][STEP_SIZE]),
        )

    # Fit models
    for model_name in models:
        models[model_name].fit(X_train, y_train, w_train)

    return models


def _get_train_data(train_data: pd.DataFrame) -> np.ndarray:
    features = [
        col
        for col in train_data.columns
        if col not in ["y", "w"] or col.startswith("theta")
    ]
    X_train = train_data[features].values
    y_train = train_data["y"].values
    w_train = train_data["w"].values
    return X_train, y_train, w_train
