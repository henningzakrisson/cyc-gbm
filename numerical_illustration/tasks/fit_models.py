from typing import Any, Dict,List

import numpy as np
import pandas as pd

from cyc_gbm import CyclicalGradientBooster
from cyc_gbm.utils.distributions import initiate_distribution

from .baseline_models import CyclicGeneralizedLinearModel, InterceptModel
from .utils.constants import (
    CGBM,
    CGLM,
    DISTRIBUTION,
    GBM,
    INTERCEPT,
    LEARNING_RATE,
    MAX_DEPTH,
    MAX_ITER,
    MODEL_HYPERPARAMS,
    MODELS,
    N_ESTIMATORS,
    STEP_SIZE,
    TOLERANCE,
)
from .utils.utils import get_targets_features

def fit_models(
    config: Dict[str, Any], train_data: pd.DataFrame, rng: np.random.Generator,n_estimators: Dict[str, List[int]]
) -> Dict[str, Any]:
    """
    Fit the models specified in the config, using hyperparameters from the config.
    Uses the training data and the random number generator.
    Creates cross validation if the model training requires it.
    """
    # Initiate distribution and get train data
    distribution = initiate_distribution(config[DISTRIBUTION])
    X_train, y_train, w_train = get_targets_features(train_data)

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
    if CGBM in config[MODELS]:
        models[CGBM] = CyclicalGradientBooster(
            distribution=distribution,
            n_estimators=n_estimators[CGBM],
            learning_rate=config[MODEL_HYPERPARAMS][CGBM][LEARNING_RATE],
            max_depth=config[MODEL_HYPERPARAMS][CGBM][MAX_DEPTH],
        )
    if GBM in config[MODELS]:
        models[GBM] = CyclicalGradientBooster(
            distribution=distribution,
            n_estimators=n_estimators[GBM],
            learning_rate=config[MODEL_HYPERPARAMS][GBM][LEARNING_RATE],
            max_depth=config[MODEL_HYPERPARAMS][GBM][MAX_DEPTH],
        )

    # Fit models
    for model_name in models:
        models[model_name].fit(X_train, y_train, w_train)

    return models
