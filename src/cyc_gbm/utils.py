import numpy as np
from typing import Union, List, Dict
from src.cyc_gbm import CycGBM
from src.cyc_gbm.distributions import initiate_distribution
from sklearn.model_selection import KFold
import warnings


def tune_kappa(
    X: np.ndarray,
    y: np.ndarray,
    kappa_max: Union[int, List[int]] = 1000,
    eps: Union[float, List[float]] = 0.1,
    max_depth: Union[int, List[int]] = 2,
    min_samples_leaf: Union[int, List[int]] = 20,
    dist: str = "normal",
    n_splits: int = 4,
    random_state: Union[int, None] = None,
) -> Dict[str, Union[List[int], np.ndarray]]:
    """Tunes the kappa parameter of a CycGBM model using k-fold cross-validation.

    :param X: The input data matrix of shape (n_samples, n_features).
    :param y: The target vector of shape (n_samples,).
    :param kappa_max: The maximum value of the kappa parameter to test. Dimension-wise or same for all parameter dimensions.
    :param eps: The epsilon parameters for the CycGBM model.Dimension-wise or same for all parameter dimensions.
    :param max_depth: The maximum depth of the decision trees in the GBM model. Dimension-wise or same for all parameter dimensions.
    :param min_samples_leaf: The minimum number of samples required to be at a leaf node in the CycGBM model. Dimension-wise or same for all parameter dimensions.
    :param dist: The distribution of the target variable.
    :param n_splits: The number of folds to use for k-fold cross-validation.
    :param random_state: The random state to use for the k-fold split.
    :param return_loss: boolean to indicate if the loss from the folds should be returned
    :return: Dictionary containing 'kappa' as the optimal number of bossting steps and 'loss' as loss array over all folds.

    """
    distribution = initiate_distribution(dist = dist)
    d = distribution.d
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    kappa_max = kappa_max if isinstance(kappa_max, list) else [kappa_max] * d
    loss = np.ones((n_splits, max(kappa_max) + 1, d)) * np.nan

    for i, idx in enumerate(kf.split(X)):
        idx_train, idx_valid = idx
        X_train, y_train = X[idx_train], y[idx_train]
        X_valid, y_valid = X[idx_valid], y[idx_valid]

        gbm = CycGBM(
            kappa=0,
            eps=eps,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            dist=dist,
        )
        gbm.fit(X_train, y_train)
        z_valid = gbm.predict(X_valid)
        loss[i, 0, :] = gbm.dist.loss(y=y_valid, z=z_valid).sum()

        for k in range(1, max(kappa_max) + 1):
            for j in range(d):
                if k < kappa_max[j]:
                    gbm.update(X=X_train, y=y_train, j=j)
                    z_valid[j] += gbm.eps[j] * gbm.trees[j][-1].predict(X_valid)
                    loss[i, k, j] = gbm.dist.loss(y=y_valid, z=z_valid).sum()
                else:
                    if j == 0:
                        loss[i, k, j] = loss[i, k - 1, j + 1]
                    else:
                        loss[i, k, j] = loss[i, k, j - 1]

            # Stop if no improvement was made
            if np.all([loss[i, k, 0] >= loss[i, k - 1, 1]] + [loss[i, k, j] >= loss[i, k, j-1] for j in range(1,d)]):
                loss[i, k + 1 :, :] = loss[i, k, -1]
                break

    loss_total = loss.sum(axis=0)
    loss_delta = np.zeros((d,max(kappa_max) + 1))
    loss_delta[0,1:] = loss_total[1:, 0] - loss_total[:-1, -1]
    for j in range(1,d):
        loss_delta[j,1:] = loss_total[1:, j] - loss_total[1:, j-1]
    kappa = np.argmax(loss_delta > 0, axis=1) - 1
    did_not_converge = (loss_delta > 0).sum(axis=1) == 0
    for j in range(d):
        if did_not_converge[j] and kappa_max[j] > 0:
            warnings.warn(f"Tuning did not converge for dimension {j}")
            kappa[j] = kappa_max[j]

    results = {"kappa": kappa, "loss": loss}

    return results

if __name__ == "__main__":
    n = 100
    expected_kappa = [12, 16]
    rng = np.random.default_rng(seed=10)
    X0 = np.arange(0, n)
    X1 = np.arange(0, n)
    rng.shuffle(X1)
    mu = 10 * (X0 > 0.3 * n) + 5 * (X1 > 0.5 * n)
    sigma = np.exp(1 + 1 * (X0 < 0.4 * n))

    X = np.stack([X0, X1]).T
    y = rng.normal(mu, sigma)

    kappa_max = [1000, 100]
    eps = 0.1
    max_depth = 2
    min_samples_leaf = 20
    random_state = 5
    tuning_results = tune_kappa(
        X=X,
        y=y,
        kappa_max=kappa_max,
        eps=eps,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        dist="normal",
        n_splits=4,
        random_state=random_state,
    )