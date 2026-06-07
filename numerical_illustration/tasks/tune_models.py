import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE

from cyc_gbm import CyclicalGradientBooster
from cyc_gbm.utils.distributions import initiate_distribution
from cyc_gbm.utils.tuning import _fold_split, tune_n_estimators

from .utils.constants import (CGBM, DISTRIBUTION, EARLY_STOPPING_ROUNDS, GBM,
                               LEARNING_RATE, MAX_DEPTH, MODEL_HYPERPARAMS,
                               MODELS, N_ESTIMATORS, N_SPLITS, NGBOOST,
                               TUNING)
from .utils.utils import get_targets_features

logger = logging.getLogger(__name__)


def tune_models(
    config: Dict[str, Any],
    train_data: pd.DataFrame,
    rng: np.random.Generator,
    log_prefix: str = "",
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[int]]]:
    distribution = initiate_distribution(config[DISTRIBUTION])
    X_train, y_train, w_train = get_targets_features(train_data)
    loss_results = {}
    n_estimators = {}
    if config[TUNING]:
        for model_name in set(config[MODELS]).intersection([GBM, CGBM]):
            logger.info(f"{log_prefix}Tuning {model_name}")
            model = CyclicalGradientBooster(
                distribution=distribution,
                learning_rate=config[MODEL_HYPERPARAMS][model_name][LEARNING_RATE],
                max_depth=config[MODEL_HYPERPARAMS][model_name][MAX_DEPTH],
            )

            if model_name == CGBM:
                n_estimators_max = config[MODEL_HYPERPARAMS][model_name][N_ESTIMATORS]
            elif model_name == GBM:
                n_estimators_max = [
                    config[MODEL_HYPERPARAMS][model_name][N_ESTIMATORS],
                    0,
                ]

            tuning_results = tune_n_estimators(
                X=X_train,
                y=y_train,
                w=w_train,
                model=model,
                n_estimators_max=n_estimators_max,
                n_splits=config[N_SPLITS],
                rng=rng,
                log_prefix=log_prefix,
            )
            loss_results[model_name] = tuning_results["loss"]
            n_estimators[model_name] = tuning_results["n_estimators"]

        if NGBOOST in config[MODELS]:
            logger.info(f"{log_prefix}Tuning {NGBOOST}")
            tuning_results = _tune_ngboost(
                X=X_train,
                y=y_train,
                w=w_train,
                distribution=distribution,
                n_estimators_max=int(config[MODEL_HYPERPARAMS][NGBOOST][N_ESTIMATORS]),
                learning_rate=float(config[MODEL_HYPERPARAMS][NGBOOST][LEARNING_RATE]),
                early_stopping_rounds=int(config[MODEL_HYPERPARAMS][NGBOOST][EARLY_STOPPING_ROUNDS]),
                n_splits=config[N_SPLITS],
                rng=rng,
                log_prefix=log_prefix,
            )
            loss_results[NGBOOST] = tuning_results["loss"]
            n_estimators[NGBOOST] = tuning_results["n_estimators"]

    else:
        n_estimators[CGBM] = config[MODEL_HYPERPARAMS][CGBM][N_ESTIMATORS]
        n_estimators[GBM] = [config[MODEL_HYPERPARAMS][GBM][N_ESTIMATORS]] + [0] * (len(n_estimators[CGBM]) - 1)
        if NGBOOST in config[MODELS]:
            n_estimators[NGBOOST] = int(config[MODEL_HYPERPARAMS][NGBOOST][N_ESTIMATORS])
    return loss_results, n_estimators


def _tune_ngboost(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    distribution,
    n_estimators_max: int,
    learning_rate: float,
    early_stopping_rounds: int,
    n_splits: int,
    rng: np.random.Generator,
    log_prefix: str = "",
) -> Dict[str, Any]:
    """Tune NGBoost n_estimators via k-fold CV using staged_pred_dist,
    mirroring the same fold split and loss computation as CGBM tuning."""
    folds = _fold_split(X=X, y=y, w=w, n_splits=n_splits, rng=rng)

    fold_train_losses = []
    fold_valid_losses = []
    fold_best_iters = []

    for i, fold in folds.items():
        X_tr, y_tr, w_tr, X_val, y_val, w_val = fold
        logger.info(f"{log_prefix}NGBoost fold {i + 1}/{n_splits}")

        model = NGBRegressor(
            Dist=Normal,
            Score=MLE,
            n_estimators=n_estimators_max,
            learning_rate=learning_rate,
            verbose=False,
            random_state=int(rng.integers(1e6)),
        )
        model.fit(X_tr, y_tr, sample_weight=w_tr)

        # Record loss at every boosting step using our own distribution
        train_losses = np.zeros((n_estimators_max, 2))
        valid_losses = np.zeros((n_estimators_max, 2))

        for k, (pred_tr, pred_val) in enumerate(
            zip(model.staged_pred_dist(X_tr), model.staged_pred_dist(X_val))
        ):
            z_tr = np.stack([pred_tr.loc, np.log(pred_tr.scale)])
            z_val = np.stack([pred_val.loc, np.log(pred_val.scale)])
            step_loss_tr = distribution.loss(y=y_tr, z=z_tr, w=w_tr).sum()
            step_loss_val = distribution.loss(y=y_val, z=z_val, w=w_val).sum()
            # Both parameter dims updated jointly — fill both columns identically
            train_losses[k, :] = step_loss_tr
            valid_losses[k, :] = step_loss_val

        fold_train_losses.append(train_losses)
        fold_valid_losses.append(valid_losses)
        fold_best_iters.append(int(np.argmin(valid_losses[:, 0])))

    # Average best iteration across folds
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
