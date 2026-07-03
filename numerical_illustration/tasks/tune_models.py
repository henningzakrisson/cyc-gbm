import logging
from typing import Any

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder

from cyc_gbm import CyclicalGradientBooster
from cyc_gbm.utils.distributions import Distribution
from cyc_gbm.utils.tuning import _fold_split, tune_n_estimators

from ..schema import (
    CyclicalGradientBoostingMachineConfig,
    GradientBoostingMachineConfig,
    ModelClass,
    NaturalGradientBoostingMachineConfig,
    NumericalIllustrationConfig,
)
from .utils.utils import get_targets_features

logger = logging.getLogger(__name__)


def tune_models(
    config: NumericalIllustrationConfig,
    train_data: pd.DataFrame,
    rng: np.random.Generator,
    log_prefix: str = "",
) -> tuple[dict[str, pd.DataFrame], dict[str, list[int]]]:
    distribution = config.data.distribution_object
    X_train, y_train, w_train = get_targets_features(train_data)
    loss_results = {}
    n_estimators = {}

    model_configs = config.model_configs_by_class

    if config.tuning.perform_tuning:
        for mc_name in [ModelClass.GBM, ModelClass.CGBM]:
            if mc_name not in model_configs:
                continue
            mc = model_configs[mc_name]
            logger.info(f"{log_prefix}Tuning {mc_name}")

            model = CyclicalGradientBooster(
                distribution=distribution,
                learning_rate=mc.learning_rate,
                max_depth=mc.max_depth,
                min_samples_leaf=mc.min_samples_leaf,
            )

            if isinstance(mc, CyclicalGradientBoostingMachineConfig):
                n_estimators_max = mc.n_estimators
            elif isinstance(mc, GradientBoostingMachineConfig):
                n_estimators_max = mc.n_estimators_as_list(distribution.n_dim)

            tuning_results = tune_n_estimators(
                X=X_train,
                y=y_train,
                w=w_train,
                model=model,
                n_estimators_max=n_estimators_max,
                n_splits=config.tuning.n_folds,
                rng=rng,
                log_prefix=log_prefix,
            )
            loss_results[mc_name] = tuning_results["loss"]
            n_estimators[mc_name] = tuning_results["n_estimators"]

        if ModelClass.NGBOOST in model_configs:
            mc = model_configs[ModelClass.NGBOOST]
            logger.info(f"{log_prefix}Tuning {ModelClass.NGBOOST}")
            tuning_results = _tune_ngboost(
                X=X_train,
                y=y_train,
                w=w_train,
                distribution=distribution,
                n_estimators_max=mc.n_estimators,
                learning_rate=mc.learning_rate,
                max_depth=mc.max_depth,
                n_folds=config.tuning.n_folds,
                rng=rng,
                log_prefix=log_prefix,
            )
            loss_results[ModelClass.NGBOOST] = tuning_results["loss"]
            n_estimators[ModelClass.NGBOOST] = tuning_results["n_estimators"]

    else:
        if ModelClass.CGBM in model_configs:
            mc_cgbm = model_configs[ModelClass.CGBM]
            n_estimators[ModelClass.CGBM] = mc_cgbm.n_estimators
            if ModelClass.GBM in model_configs:
                mc_gbm = model_configs[ModelClass.GBM]
                n_estimators[ModelClass.GBM] = mc_gbm.n_estimators_as_list(
                    len(mc_cgbm.n_estimators)
                )
        elif ModelClass.GBM in model_configs:
            mc_gbm = model_configs[ModelClass.GBM]
            n_estimators[ModelClass.GBM] = mc_gbm.n_estimators_as_list(
                distribution.n_dim
            )

        if ModelClass.NGBOOST in model_configs:
            mc_ngb = model_configs[ModelClass.NGBOOST]
            n_estimators[ModelClass.NGBOOST] = mc_ngb.n_estimators

    return loss_results, n_estimators


def _tune_ngboost(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    distribution: Distribution,
    n_estimators_max: int,
    learning_rate: float,
    max_depth: int,
    n_folds: int,
    rng: np.random.Generator,
    log_prefix: str = "",
) -> dict[str, Any]:
    """Tune NGBoost n_estimators via k-fold CV using staged_pred_dist,
    mirroring the same fold split and loss computation as CGBM tuning."""

    # NGBoost requires numeric input; one-hot encode categoricals.
    if isinstance(X, pd.DataFrame):
        cat_cols = [
            c for c in X.columns
            if isinstance(X[c].dtype, pd.CategoricalDtype)
        ]
        if cat_cols:
            num_cols = [c for c in X.columns if c not in cat_cols]
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = enc.fit_transform(X[cat_cols])
            numeric = X[num_cols].to_numpy(dtype=float, na_value=0.0)
            X = np.hstack([numeric, encoded])
        else:
            X = X.to_numpy(dtype=float, na_value=0.0)
    folds = _fold_split(X=X, y=y, w=w, n_splits=n_folds, rng=rng)

    fold_train_losses = []
    fold_valid_losses = []
    fold_best_iters = []

    for i, fold in folds.items():
        X_tr, y_tr, w_tr, X_val, y_val, w_val = fold
        logger.info(f"{log_prefix}NGBoost fold {i + 1}/{n_folds}")

        model = NGBRegressor(
            Dist=Normal,
            Score=MLE,
            Base=DecisionTreeRegressor(criterion="squared_error", max_depth=max_depth),
            n_estimators=n_estimators_max,
            learning_rate=learning_rate,
            verbose=False,
            random_state=int(rng.integers(1e6)),
        )
        model.fit(X_tr, y_tr, sample_weight=w_tr)

        train_losses = np.zeros((n_estimators_max, 2))
        valid_losses = np.zeros((n_estimators_max, 2))

        for k, (pred_tr, pred_val) in enumerate(
            zip(model.staged_pred_dist(X_tr), model.staged_pred_dist(X_val))
        ):
            z_tr = np.stack([pred_tr.loc, np.log(pred_tr.scale)])
            z_val = np.stack([pred_val.loc, np.log(pred_val.scale)])
            step_loss_tr = distribution.loss(y=y_tr, z=z_tr, w=w_tr).sum()
            step_loss_val = distribution.loss(y=y_val, z=z_val, w=w_val).sum()
            train_losses[k, :] = step_loss_tr
            valid_losses[k, :] = step_loss_val

        fold_train_losses.append(train_losses)
        fold_valid_losses.append(valid_losses)
        fold_best_iters.append(int(np.argmin(valid_losses[:, 0])))

    best_n_estimators = int(np.mean(fold_best_iters))
    logger.info(f"{log_prefix}NGBoost per-fold best iters: {fold_best_iters}")
    logger.info(f"{log_prefix}NGBoost optimal n_estimators: {best_n_estimators}")

    return {
        "n_estimators": best_n_estimators,
        "loss": {
            "train": fold_train_losses,
            "valid": fold_valid_losses,
        },
    }
