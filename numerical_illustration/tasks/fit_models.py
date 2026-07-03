import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from cyc_gbm import CyclicalGradientBooster
from cyc_gbm.utils.distributions import Distribution, initiate_distribution

from ..schema import (
    CyclicalGeneralizedLinearModelConfig,
    CyclicalGradientBoostingMachineConfig,
    GradientBoostingMachineConfig,
    InterceptConfig,
    ModelClass,
    ModelConfig,
    NaturalGradientBoostingMachineConfig,
)
from ..schema.data import DataConfig
from .baseline_models import CyclicGeneralizedLinearModel, InterceptModel, NGBoostModel
from .utils.utils import get_targets_features

logger = logging.getLogger(__name__)


def _resolve_distribution(model_config: ModelConfig, data_config: DataConfig) -> Distribution:
    """Return the distribution for a model, honouring per-model parametrization overrides."""
    parameterization = getattr(model_config, "parameterization", None) or data_config.parameterization
    return initiate_distribution(data_config.distribution, parameterization=parameterization)


def fit_models(
    model_configs: Iterable[ModelConfig],
    data_config: DataConfig,
    train_data: pd.DataFrame,
    rng: np.random.Generator,
    n_estimators: dict[str, list[int]],
    log_prefix: str = "",
) -> dict[str, Any]:
    """Fit the models specified in the config.

    Args:
        model_configs: list of model configuration objects
        data_config: data configuration (used to resolve per-model distributions)
        train_data: training data
        rng: random number generator
        n_estimators: tuned n_estimators per model (keyed by model name)
        log_prefix: prefix for log messages
    """
    X_train, y_train, w_train = get_targets_features(train_data)

    return {
        mc.name: _build_and_fit(
            model_config=mc,
            distribution=_resolve_distribution(mc, data_config),
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
    name = model_config.name
    logger.info(f"{log_prefix}Fitting {name}")

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
            n_estimators=n_estimators[name],
            learning_rate=model_config.learning_rate,
            max_depth=model_config.max_depth,
            min_samples_leaf=model_config.min_samples_leaf,
        )

    elif isinstance(model_config, GradientBoostingMachineConfig):
        model = CyclicalGradientBooster(
            distribution=distribution,
            n_estimators=n_estimators[name],
            learning_rate=model_config.learning_rate,
            max_depth=model_config.max_depth,
            min_samples_leaf=model_config.min_samples_leaf,
        )

    elif isinstance(model_config, NaturalGradientBoostingMachineConfig):
        model = NGBoostModel(
            distribution=distribution,
            n_estimators=n_estimators[name],
            learning_rate=model_config.learning_rate,
            max_depth=model_config.max_depth,
        )
    else:
        raise ValueError(f"Unknown model config type: {type(model_config)}")

    model.fit(X, y, w)
    return model
