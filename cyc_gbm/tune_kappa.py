from typing import Union, List, Dict, Tuple, Optional
import logging

import numpy as np

from cyc_gbm import CyclicalGradientBooster
from cyc_gbm.distributions import initiate_distribution, Distribution
from cyc_gbm.logger import CycGBMLogger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
formatter = logging.Formatter("[%(asctime)s][%(message)s]", datefmt="%Y-%m-%d %H:%M")
logger.handlers[0].setFormatter(formatter)


def _fold_split(
    X: np.ndarray,
    n_splits: int = 4,
    random_state: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split data into k folds.

    :param X: The input data matrix of shape (n_samples, n_features).
    :param n_splits: The number of folds to use for k-fold cross-validation.
    :param random_state: The seed used by the random number generator.
    :param rng: The random number generator.
    :return List of tuples containing (idx_train, idx_test) for each fold.
    """
    if rng is None:
        rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    idx = rng.permutation(n_samples)
    idx_folds = np.array_split(idx, n_splits)
    folds = []
    for i in range(n_splits):
        idx_test = idx_folds[i]
        idx_train = np.concatenate(idx_folds[:i] + idx_folds[i + 1 :])
        folds.append((idx_train, idx_test))
    return folds


def tune_kappa(
    X: np.ndarray,
    y: np.ndarray,
    w: Union[np.ndarray, float] = 1.0,
    kappa_max: Union[int, List[int]] = 1000,
    eps: Union[float, List[float]] = 0.1,
    max_depth: Union[int, List[int]] = 2,
    min_samples_leaf: Union[int, List[int]] = 20,
    distribution: Union[str, Distribution] = "normal",
    n_splits: int = 4,
    random_state: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    logger: Optional[CycGBMLogger] = None,
) -> Dict[str, Union[List[int], np.ndarray]]:
    """Tunes the kappa parameter of a CycGBM model using k-fold cross-validation.

    :param X: The input data matrix of shape (n_samples, n_features).
    :param y: The target vector of shape (n_samples,).
    :param w: The weights for the training data, of shape (n_samples,). Default is 1 for all samples.
    :param kappa_max: The maximum value of the kappa parameter to test. Dimension-wise or same for all parameter dimensions.
    :param eps: The epsilon parameters for the CycGBM model.Dimension-wise or same for all parameter dimensions.
    :param max_depth: The maximum depth of the decision trees in the GBM model. Dimension-wise or same for all parameter dimensions.
    :param min_samples_leaf: The minimum number of samples required to be at a leaf node in the CycGBM model. Dimension-wise or same for all parameter dimensions.
    :param distribution: The distribution of the target variable.
    :param n_splits: The number of folds to use for k-fold cross-validation.
    :param random_state: The random state to use for the k-fold split.
    :param rng: The random number generator.
    :param logger: The simulation logger to use for logging.
    :return: A dictionary containing the following keys:
        - "kappa": The optimal kappa parameter value for each parameter dimension.
        - "loss": The loss values for each kappa parameter value.
    """
    if logger is None:
        logger = CycGBMLogger(verbose=0)
    if isinstance(w, float):
        w = np.ones(len(y)) * w
    if rng is None:
        rng = np.random.default_rng(random_state)
    folds = _fold_split(X=X, n_splits=n_splits, rng=rng)
    if isinstance(distribution, str):
        distribution = initiate_distribution(distribution=distribution)
    d = distribution.d
    kappa_max = kappa_max if isinstance(kappa_max, list) else [kappa_max] * d
    loss_train = np.ones((n_splits, max(kappa_max) + 1, d)) * np.nan
    loss_valid = np.ones((n_splits, max(kappa_max) + 1, d)) * np.nan
    for i, idx in enumerate(folds):
        logger.append_format_level(f"fold {i+1}/{n_splits}")
        logger.log("tuning", verbose=1)
        idx_train, idx_valid = idx
        X_train, y_train, w_train = X[idx_train], y[idx_train], w[idx_train]
        X_valid, y_valid, w_valid = X[idx_valid], y[idx_valid], w[idx_valid]

        gbm = CyclicalGradientBooster(
            kappa=0,
            eps=eps,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            distribution=distribution,
        )
        gbm.fit(X_train, y_train, w_train)
        z_train = gbm.predict(X_train)
        z_valid = gbm.predict(X_valid)
        loss_train[i, 0, :] = gbm.dist.loss(y=y_train, z=z_train, w=w_train).sum()
        loss_valid[i, 0, :] = gbm.dist.loss(y=y_valid, z=z_valid, w=w_valid).sum()

        for k in range(1, max(kappa_max) + 1):
            for j in range(d):
                if k < kappa_max[j]:
                    gbm.update(X=X_train, y=y_train, z=z_train, w=w_train, j=j)
                    z_train[j] += gbm.eps[j] * gbm.trees[j][-1].predict(X_train)
                    z_valid[j] += gbm.eps[j] * gbm.trees[j][-1].predict(X_valid)
                    loss_train[i, k, j] = gbm.dist.loss(
                        y=y_train, z=z_train, w=w_train
                    ).sum()
                    loss_valid[i, k, j] = gbm.dist.loss(
                        y=y_valid, z=z_valid, w=w_valid
                    ).sum()
                else:
                    if j == 0:
                        loss_train[i, k, j] = loss_train[i, k - 1, j + 1]
                        loss_valid[i, k, j] = loss_valid[i, k - 1, j + 1]
                    else:
                        loss_train[i, k, j] = loss_train[i, k, j - 1]
                        loss_valid[i, k, j] = loss_valid[i, k, j - 1]

            # Stop if no improvement was made
            if k != max(kappa_max) and np.all(
                [loss_valid[i, k, 0] >= loss_valid[i, k - 1, 1]]
                + [loss_valid[i, k, j] >= loss_valid[i, k, j - 1] for j in range(1, d)]
            ):
                loss_valid[i, k + 1 :, :] = loss_valid[i, k, -1]
                logger.log(
                    msg=f"tuning converged after {k} steps",
                    verbose=1,
                )
                break

            if k == max(kappa_max):
                logger.log(
                    msg="tuning did not converge",
                    verbose=1,
                )
            logger.log_progress(step=k, total_steps=max(kappa_max) + 1, verbose=2)
        logger.reset_progress()
        logger.remove_format_level()

    loss_total = loss_valid.sum(axis=0)
    loss_delta = np.zeros((d, max(kappa_max) + 1))
    loss_delta[0, 1:] = loss_total[1:, 0] - loss_total[:-1, -1]
    for j in range(1, d):
        loss_delta[j, 1:] = loss_total[1:, j] - loss_total[1:, j - 1]
    kappa = np.maximum(0, np.argmax(loss_delta > 0, axis=1) - 1)
    did_not_converge = (loss_delta > 0).sum(axis=1) == 0
    for j in range(d):
        if did_not_converge[j] and kappa_max[j] > 0:
            logger.log(f"tuning did not converge for dimension {j}", verbose=1)
            kappa[j] = kappa_max[j]

    return {"kappa": kappa, "loss": {"train": loss_train, "valid": loss_valid}}
