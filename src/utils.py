import numpy as np
from typing import Union, List
from src.CycGBM import CycGBM
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
    return_loss: bool = False,
) -> List[int]:
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
    :return: The optimal value of the kappa parameter."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # Assume two dimensions
    kappa_max = kappa_max if isinstance(kappa_max, list) else [kappa_max] * 2
    loss = np.ones((n_splits, max(kappa_max) + 1, 2)) * np.nan

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
            # Assume 2 dimensions
            for j in [0, 1]:
                if k < kappa_max[j]:
                    gbm.update(X=X_train, y=y_train, j=j)
                    z_valid[j] += gbm.eps[j] * gbm.trees[j][-1].predict(X_valid)
                    loss[i, k, j] = gbm.dist.loss(y=y_valid, z=z_valid).sum()
                else:
                    if j == 0:
                        loss[i, k, j] = loss[i, k - 1, j + 1]
                    else:
                        loss[i, k, j] = loss[i, k, j - 1]

            # Assume two dimensions
            if loss[i, k, 1] >= loss[i, k, 0] >= loss[i, k - 1, 1]:
                loss[i, k + 1 :, :] = loss[i, k, -1]
                break

    loss_total = loss.sum(axis=0)
    # Assume two dimensions
    loss_delta_0 = loss_total[1:, 0] - loss_total[:-1, -1]
    loss_delta_1 = loss_total[1:, 1] - loss_total[1:, 0]
    loss_delta = np.stack([loss_delta_0, loss_delta_1])
    kappa = np.argmax(loss_delta > 0, axis=1)

    did_not_converge = (loss_delta > 0).sum(axis=1) == 0
    # Assume two dimensions
    for j in range(0, 2):
        if did_not_converge[j] and kappa_max[j] > 0:
            warnings.warn(f"Tuning did not converge for dimension {j}")
            kappa[j] = kappa_max[j]
    if return_loss:
        return kappa, loss
    return kappa
