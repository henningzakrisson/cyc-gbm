from typing import Union, List, Dict, Tuple
import logging

import numpy as np

from src.cyc_gbm import CycGBM
from src.cyc_gbm.distributions import initiate_distribution, Distribution
from src.cyc_gbm.utils import SimulationLogger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
formatter = logging.Formatter("[%(asctime)s][%(message)s]", datefmt="%Y-%m-%d %H:%M")
logger.handlers[0].setFormatter(formatter)


def _fold_split(
    X: np.ndarray,
    n_splits: int = 4,
    random_state: Union[int, None] = None,
    rng: Union[np.random.Generator, None] = None,
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
    random_state: Union[int, None] = None,
    rng: Union[np.random.Generator, None] = None,
    verbose: int = 0,
    logger: Union[SimulationLogger, None] = None,
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
    :param verbose: The verbosity level of the logger. 0: no logging, 1: fold logging, 2: tree logging.
    :param logger: The simulation logger to use for logging.
    :return: Dictionary containing 'kappa' as the optimal number of bossting steps and 'loss' as loss array over all folds.

    """
    if logger is None:
        logger = SimulationLogger()
    if isinstance(w, float):
        w = np.ones(len(y)) * w
    if rng is None:
        rng = np.random.default_rng(random_state)
    folds = _fold_split(X=X, n_splits=n_splits, rng=rng)
    if isinstance(distribution, str):
        distribution = initiate_distribution(distribution=distribution)
    d = distribution.d
    kappa_max = kappa_max if isinstance(kappa_max, list) else [kappa_max] * d
    loss = np.ones((n_splits, max(kappa_max) + 1, d)) * np.nan
    if verbose > 0:
        logger.log_info(
            f"Starting tuning of kappa with {n_splits}-fold cross-validation"
        )
    for i, idx in enumerate(folds):
        if verbose == 1:
            logger.log_info(f"Fold {i+1}/{n_splits}")
        idx_train, idx_valid = idx
        X_train, y_train, w_train = X[idx_train], y[idx_train], w[idx_train]
        X_valid, y_valid, w_valid = X[idx_valid], y[idx_valid], w[idx_valid]

        gbm = CycGBM(
            kappa=0,
            eps=eps,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            distribution=distribution,
        )
        gbm.fit(X_train, y_train, w_train)
        z_valid = gbm.predict(X_valid)
        loss[i, 0, :] = gbm.dist.loss(y=y_valid, z=z_valid, w=w_valid).sum()

        for k in range(1, max(kappa_max) + 1):
            if verbose > 1:
                logger.log_info(
                    f"Fold {i+1}/{n_splits}, Boosting step {k}/{max(kappa_max)}"
                )
            for j in range(d):
                if k < kappa_max[j]:
                    gbm.update(X=X_train, y=y_train, w=w_train, j=j)
                    z_valid[j] += gbm.eps[j] * gbm.trees[j][-1].predict(X_valid)
                    loss[i, k, j] = gbm.dist.loss(y=y_valid, z=z_valid, w=w_valid).sum()
                else:
                    if j == 0:
                        loss[i, k, j] = loss[i, k - 1, j + 1]
                    else:
                        loss[i, k, j] = loss[i, k, j - 1]

            # Stop if no improvement was made
            if np.all(
                [loss[i, k, 0] >= loss[i, k - 1, 1]]
                + [loss[i, k, j] >= loss[i, k, j - 1] for j in range(1, d)]
            ):
                loss[i, k + 1 :, :] = loss[i, k, -1]
                break

    loss_total = loss.sum(axis=0)
    loss_delta = np.zeros((d, max(kappa_max) + 1))
    loss_delta[0, 1:] = loss_total[1:, 0] - loss_total[:-1, -1]
    for j in range(1, d):
        loss_delta[j, 1:] = loss_total[1:, j] - loss_total[1:, j - 1]
    kappa = np.maximum(0, np.argmax(loss_delta > 0, axis=1) - 1)
    did_not_converge = (loss_delta > 0).sum(axis=1) == 0
    for j in range(d):
        if did_not_converge[j] and kappa_max[j] > 0:
            if verbose > 0:
                logger.warning(f"Tuning did not converge for dimension {j}")
            kappa[j] = kappa_max[j]

    if verbose > 0:
        logger.log_info(
            f"Finished tuning of kappa with {n_splits}-fold cross-validation"
        )
    results = {"kappa": kappa, "loss": loss}

    return results


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    n = 100000
    p = 9
    random_state = 11
    dist = "normal"

    rng = np.random.default_rng(seed=random_state)
    distribution = initiate_distribution(distribution=dist)

    X = np.concatenate([np.ones((1, n)), rng.normal(0, 1, (p - 1, n))]).T
    z0 = (
        1.5 * X[:, 1]
        + 2 * X[:, 3]
        - 0.65 * X[:, 2] ** 2
        + 0.5 * np.abs(X[:, 3]) * np.sin(0.5 * X[:, 2])
        + 0.45 * X[:, 4] * X[:, 5] ** 2
    )
    z1 = 1 + 0.02 * X[:, 2] + 0.5 * X[:, 1] * (X[:, 1] < 2) + 1.8 * (X[:, 5] > 0)
    z = np.stack([z0, z1])
    y = distribution.simulate(z=z, random_state=12)

    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
        X, y, z.T, test_size=0.2, random_state=random_state
    )
    z_train = z_train.T
    z_test = z_test.T

    # Set hyperparameters
    kappa_max = [100, 0]
    eps = 0.1
    max_depth = 3
    min_samples_leaf = 5
    n_splits = 10

    # Tune kappa
    kappa_uni = tune_kappa(
        X=X_train,
        y=y_train,
        n_splits=n_splits,
        eps=eps,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        kappa_max=kappa_max,
        dist=dist,
        random_state=13,
        log=True,
    )["kappa"]

    print(kappa_uni)
