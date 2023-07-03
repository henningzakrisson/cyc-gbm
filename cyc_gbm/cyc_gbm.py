from typing import List, Union, Optional

import numpy as np

from cyc_gbm.distributions import Distribution, initiate_distribution
from cyc_gbm.logger import CycGBMLogger
from cyc_gbm.boosting_tree import BoostingTree


class CyclicalGradientBooster:
    """
    Class for cyclical gradient boosting regressors
    """

    def __init__(
        self,
        kappa: Union[int, List[int]] = 100,
        eps: Union[float, List[float]] = 0.1,
        max_depth: Union[int, List[int]] = 2,
        min_samples_leaf: Union[int, List[int]] = 20,
        distribution: Union[str, Distribution] = "normal",
    ):
        """
        :param kappa: Number of boosting steps. Dimension-wise or global for all parameter dimensions.
        :param eps: Shrinkage factors, which scales the contribution of each tree. Dimension-wise or global for all parameter dimensions.
        :param max_depth: Maximum depths of each decision tree. Dimension-wise or global for all parameter dimensions.
        :param min_samples_leaf: Minimum number of samples required at a leaf node. Dimension-wise or global for all parameter dimensions.
        :param distribution: distribution for losses and gradients. String or Distribution object.
        """
        if isinstance(distribution, str):
            self.dist = initiate_distribution(distribution=distribution)
        else:
            self.dist = distribution
        self.d = self.dist.d
        self.kappa = (
            kappa
            if isinstance(kappa, list) or isinstance(kappa, np.ndarray)
            else [kappa] * self.d
        )
        self.eps = eps if isinstance(eps, list) else [eps] * self.d
        self.max_depth = (
            max_depth if isinstance(max_depth, list) else [max_depth] * self.d
        )
        self.min_samples_leaf = (
            min_samples_leaf
            if isinstance(min_samples_leaf, list)
            else [min_samples_leaf] * self.d
        )

        self.z0 = None
        self.trees = [[None] * self.kappa[j] for j in range(self.d)]

    def _adjust_mle(
        self, X: np.ndarray, y: np.ndarray, w: Union[np.ndarray, float] = 1
    ) -> None:
        """
        Adjust the initial values of the model for parameters not modeled by the GBM

        :param X: Input data matrix of shape (n_samples, n_features).
        :param y: True response values for the input data.
        :param w: Weights for the training data, of shape (n_samples,). Default is 1 for all samples.
        """
        z = self.predict(X=X)
        for j in range(self.d):
            if self.kappa[j] == 0:
                self.z0[j] += self.dist.opt_step(y=y, z=z, w=w, j=j)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: Union[np.ndarray, float] = 1.0,
        logger: Optional[CycGBMLogger] = None,
    ) -> None:
        """
        Train the model on the given training data.

        :param X: Input data matrix of shape (n_samples, n_features).
        :param y: True response values for the input data.
        :param w: Weights for the training data, of shape (n_samples,). Default is 1 for all samples.
        """
        if logger is None:
            logger = CycGBMLogger(verbose=0)
        if isinstance(w, float):
            w = np.ones(len(y)) * w
        self.z0 = self.dist.mle(y=y, w=w)[:, None]
        z = np.tile(self.z0, (1, len(y)))
        for k in range(0, max(self.kappa)):
            for j in range(self.d):
                if k >= self.kappa[j]:
                    continue
                tree = BoostingTree(
                    max_depth=self.max_depth[j],
                    min_samples_leaf=self.min_samples_leaf[j],
                    distribution=self.dist,
                )
                tree.fit_gradients(X=X, y=y, z=z, w=w, j=j)
                z[j] += self.eps[j] * tree.predict(X)
                self.trees[j][k] = tree
                logger.log_progress(
                    step=(k + 1) * (j + 1),
                    total_steps=(max(self.kappa) * self.d),
                    verbose=2,
                )

        self._adjust_mle(X=X, y=y, w=w)

    def update(
        self, X: np.ndarray, y: np.ndarray, j: int, w: Union[np.ndarray, float] = 1
    ) -> None:
        """
        Updates the current boosting model with one additional tree

        :param X: The training input data, shape (n_samples, n_features).
        :param y: The target values for the training data.
        :param j: Parameter dimension to update
        :param w: Weights for the training data, of shape (n_samples,). Default is 1 for all samples.
        """
        if isinstance(w, float):
            w = np.ones(len(y)) * w
        z = self.predict(X)
        tree = BoostingTree(
            max_depth=self.max_depth[j],
            min_samples_leaf=self.min_samples_leaf[j],
            distribution=self.dist,
        )
        tree.fit_gradients(X=X, y=y, z=z, w=w, j=j)
        self.trees[j] += [tree]
        self.kappa[j] += 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict response values for the input data using the trained model.

        :param X: Input data matrix of shape (n_samples, n_features).
        :return: Predicted response values of shape (d,n_samples).
        """
        z_hat = np.zeros((self.d, len(X)))
        for j in range(self.d):
            if len(self.trees[j]) > 0:
                z_hat[j] = self.eps[j] * sum(
                    [tree.predict(X) for tree in self.trees[j]]
                )
        return self.z0 + z_hat

    def feature_importances(
        self, j: Union[str, int] = "all", normalize: bool = True
    ) -> np.ndarray:
        """
        Computes the feature importances for parameter dimension j

        :param j: Parameter dimension. If 'all', calculate importance over all parameter dimensions.
        :return: Feature importance of shape (n_features,)
        """
        if j == "all":
            feature_importances = np.array(
                [
                    [tree.feature_importances() for tree in self.trees[j]]
                    for j in range(self.d)
                ]
            ).sum(axis=(0, 1))
        else:
            feature_importances = np.array(
                [tree.feature_importances() for tree in self.trees[j]]
            ).sum(axis=0)
        if normalize:
            feature_importances /= feature_importances.sum()
        return feature_importances


if __name__ == "__main__":
    rng = np.random.default_rng(seed=10)
    n = 1000
    p = 9
    X = np.concatenate([np.ones((1, n)), rng.normal(0, 1, (p - 1, n))]).T
    z0 = (
        1.5 * X[:, 1]
        + 2 * X[:, 3]
        - 0.65 * X[:, 2] ** 2
        + 0.5 * np.abs(X[:, 3]) * np.sin(0.5 * X[:, 2])
        + 0.45 * X[:, 4] * X[:, 5] ** 2
    )
    z1 = 1 + 0.02 * X[:, 2] + 0.5 * X[:, 1] * (X[:, 1] < 2) + 1.8 * (X[:, 5] > 0)
    z2 = 0.2 * X[:, 3] + 0.03 * X[:, 2] ** 2
    z = np.stack([z0, z1, z2])
    distribution = initiate_distribution(distribution="multivariate_normal")
    y = distribution.simulate(z=z, random_state=5)

    kappa = [23, 17, 79]
    eps = [0.5, 0.25, 0.1]
    gbm = CyclicalGradientBooster(distribution="multivariate_normal", kappa=kappa, eps=eps)
    gbm.fit(X, y)
