import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from cyc_gbm import CyclicalGradientBooster
from cyc_gbm.utils.distributions import initiate_distribution

from ..config.config_models import (
    CyclicalGeneralizedLinearModelConfig,
    CyclicalGradientBoostingMachineConfig,
    GradientBoostingMachineConfig,
    InterceptConfig,
    NaturalGradientBoostingMachineConfig,
    NumericalIllustrationConfig,
)
from .baseline_models import CyclicGeneralizedLinearModel, InterceptModel, NGBoostModel
from .utils.utils import get_targets_features

logger = logging.getLogger(__name__)


def fit_models(
    config: NumericalIllustrationConfig,
    train_data: pd.DataFrame,
    rng: np.random.Generator,
    n_estimators: Dict[str, List[int]],
    log_prefix: str = "",
) -> Dict[str, Any]:
    """
    Fit the models specified in the config, using hyperparameters from the config.
    Uses the training data and the random number generator.
    """
    distribution = initiate_distribution(config.data.distribution)
    X_train, y_train, w_train = get_targets_features(train_data)

    models = {}
    for model_config in config.models:
        mc = model_config.model_class

        if isinstance(model_config, InterceptConfig):
            models[mc] = InterceptModel(distribution=distribution)

        elif isinstance(model_config, CyclicalGeneralizedLinearModelConfig):
            models[mc] = CyclicGeneralizedLinearModel(
                distribution=distribution,
                max_iter=model_config.max_iter,
                tol=model_config.tolerance,
                eps=model_config.step_size,
            )

        elif isinstance(model_config, CyclicalGradientBoostingMachineConfig):
            models[mc] = CyclicalGradientBooster(
                distribution=distribution,
                n_estimators=n_estimators[mc],
                learning_rate=model_config.learning_rate,
                max_depth=model_config.max_depth,
            )

        elif isinstance(model_config, GradientBoostingMachineConfig):
            models[mc] = CyclicalGradientBooster(
                distribution=distribution,
                n_estimators=n_estimators[mc],
                learning_rate=model_config.learning_rate,
                max_depth=model_config.max_depth,
            )

        elif isinstance(model_config, NaturalGradientBoostingMachineConfig):
            models[mc] = NGBoostModel(
                distribution=distribution,
                n_estimators=n_estimators[mc],
                learning_rate=model_config.learning_rate,
                max_depth=model_config.max_depth,
            )

    # Fit models
    for model_name in models:
        logger.info(f"{log_prefix}Fitting {model_name}")
        models[model_name].fit(X_train, y_train, w_train)

    return models
