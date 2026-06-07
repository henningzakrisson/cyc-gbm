import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from cyc_gbm import CyclicalGradientBooster
from cyc_gbm.utils.distributions import Distribution

from ..schema import (
    CyclicalGeneralizedLinearModelConfig,
    CyclicalGradientBoostingMachineConfig,
    GradientBoostingMachineConfig,
    InterceptConfig,
    ModelClass,
    ModelConfig,
    NaturalGradientBoostingMachineConfig,
)
from .baseline_models import CyclicGeneralizedLinearModel, InterceptModel, NGBoostModel
from .utils.utils import get_targets_features

logger = logging.getLogger(__name__)


def fit_models(
    model_configs: Iterable[ModelConfig],
    distribution: Distribution,
    train_data: pd.DataFrame,
    rng: np.random.Generator,
    n_estimators: dict[str, list[int]],
    log_prefix: str = "",
) -> dict[str, Any]:
    """Fit the models specified in the config.

    Args:
        model_configs: list of model configuration objects
        distribution: instantiated distribution object
        train_data: training data
        rng: random number generator
        n_estimators: tuned n_estimators per model (keyed by model_class)
        log_prefix: prefix for log messages
    """
    X_train, y_train, w_train = get_targets_features(train_data)

    return {
        mc.model_class: _build_and_fit(
            model_config=mc,
            distribution=distribution,
            n_estimators=n_estimators,
            X=X_train,
            y=y_train,
            w=w_train,
            log_prefix=log_prefix,
        )
        for mc in model_configs
    }


def _build_and_fit(
    model_config: ModelConfig,
    distribution: Distribution,
    n_estimators: dict[str, list[int]],
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    log_prefix: str = "",
) -> Any:
    """Instantiate a model from its config, fit it, and return it."""
    mc = model_config.model_class
    logger.info(f"{log_prefix}Fitting {mc}")

    if isinstance(model_config, InterceptConfig):
        model = InterceptModel(distribution=distribution)

    elif isinstance(model_config, CyclicalGeneralizedLinearModelConfig):
        model = CyclicGeneralizedLinearModel(
            distribution=distribution,
            max_iter=model_config.max_iter,
            tol=model_config.tolerance,
            eps=model_config.step_size,
        )

    elif isinstance(model_config, CyclicalGradientBoostingMachineConfig):
        model = CyclicalGradientBooster(
            distribution=distribution,
            n_estimators=n_estimators[mc],
            learning_rate=model_config.learning_rate,
            max_depth=model_config.max_depth,
        )

    elif isinstance(model_config, GradientBoostingMachineConfig):
        model = CyclicalGradientBooster(
            distribution=distribution,
            n_estimators=n_estimators[mc],
            learning_rate=model_config.learning_rate,
            max_depth=model_config.max_depth,
        )

    elif isinstance(model_config, NaturalGradientBoostingMachineConfig):
        model = NGBoostModel(
            distribution=distribution,
            n_estimators=n_estimators[mc],
            learning_rate=model_config.learning_rate,
            max_depth=model_config.max_depth,
        )
    else:
        raise ValueError(f"Unknown model config type: {type(model_config)}")

    model.fit(X, y, w)
    return model
