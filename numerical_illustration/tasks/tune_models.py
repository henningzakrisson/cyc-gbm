from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from cyc_gbm import CyclicalGradientBooster
from cyc_gbm.utils.distributions import initiate_distribution
from cyc_gbm.utils.tuning import tune_n_estimators

from .utils.constants import (CGBM, DISTRIBUTION, GBM, LEARNING_RATE,
                              MAX_DEPTH, MODEL_HYPERPARAMS, MODELS,
                              N_ESTIMATORS, N_SPLITS, PARALLEL, TUNING)
from .utils.utils import get_targets_features


def tune_models(
    config: Dict[str, Any], train_data: pd.DataFrame, rng: np.random.Generator
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[int]]]:
    distribution = initiate_distribution(config[DISTRIBUTION])
    X_train, y_train, w_train = get_targets_features(train_data)
    loss_results = {}
    n_estimators = {}
    if config[TUNING]:
        for model_name in set(config[MODELS]).intersection([GBM, CGBM]):
            model = CyclicalGradientBooster(
                distribution=distribution,
                learning_rate=config[MODEL_HYPERPARAMS][model_name][LEARNING_RATE],
                max_depth=config[MODEL_HYPERPARAMS][model_name][MAX_DEPTH],
            )

            tuning_results = tune_n_estimators(
                X=X_train,
                y=y_train,
                w=w_train,
                model=model,
                n_estimators_max=config[MODEL_HYPERPARAMS][model_name][N_ESTIMATORS],
                n_splits=config[N_SPLITS],
                rng=rng,
                parallel=config[PARALLEL],
            )
            loss_results[model_name] = tuning_results["loss"]
            n_estimators[model_name] = tuning_results["n_estimators"]

    else:
        n_estimators = {
            model_name: config[MODEL_HYPERPARAMS][model_name][N_ESTIMATORS]
            for model_name in set(config[MODELS]).intersection([GBM, CGBM])
        }
    return loss_results, n_estimators
