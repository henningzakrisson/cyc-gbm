import numpy as np
from typing import List, Union
from src.cyc_gbm.distributions import initiate_dist
from src.cyc_gbm.GBMTree import GBMTree


class CycGBM:
    """
    Class for cyclical gradient boosting machine regressors
    """

    def __init__(
        self,
        # Assume 2 dimensions
        kappa: Union[int, List[int]] = [100, 100],
        eps: Union[float, List[float]] = [0.1, 0.1],
        max_depth: Union[int, List[int]] = [2, 2],
        min_samples_leaf: Union[int, List[int]] = [20, 20],
        dist: str = "normal",
    ):
        """
        :param kappa: Number of boosting steps. Dimension-wise or global for all parameter dimensions.
        :param eps: Shrinkage factors, which scales the contribution of each tree. Dimension-wise or global for all parameter dimensions.
        :param max_depth: Maximum depths of each decision tree. Dimension-wise or global for all parameter dimensions.
        :param min_samples_leaf: Minimum number of samples required at a leaf node. Dimension-wise or global for all parameter dimensions.
        :param dist: distribution for losses and gradients
        """
        # Assume 2 dimensions
        self.kappa = kappa if isinstance(kappa, list) else [kappa] * 2
        self.eps = eps if isinstance(eps, list) else [eps] * 2
        self.max_depth = max_depth if isinstance(max_depth, list) else [max_depth] * 2
        self.min_samples_leaf = (
            min_samples_leaf
            if isinstance(min_samples_leaf, list)
            else [min_samples_leaf] * 2
        )

        self.dist = initiate_dist(dist=dist)

        # Assume 2 dimensions
        self.z0s = [np.nan, np.nan]
        self.trees = [None, None]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model on the given training data.

        :param X: Input data matrix of shape (n_samples, n_features).
        :param y: True response values for the input data, of shape (n_samples,).
        """
        self.z0 = self.dist.estimate_mle(y)[:, None]

        # Assume 2 dimensions
        z = self.z0.repeat(len(y)).reshape((2, len(y)))
        self.trees = [[[]] * self.kappa[0], [[]] * self.kappa[1]]

        for k in range(0, max(self.kappa)):
            # Assume 2 dimensions
            for j in [0, 1]:
                if k >= self.kappa[j]:
                    continue
                tree = GBMTree(
                    max_depth=self.max_depth[j],
                    min_samples_leaf=self.min_samples_leaf[j],
                    dist=self.dist,
                )
                tree.fit_gradients(X=X, y=y, z=z, j=j)
                z[j] += self.eps[j] * tree.predict(X)
                self.trees[j][k] = tree

    def update(self, X: np.ndarray, y: np.ndarray, j: int):
        """
        Updates the current boosting model with one additional tree

        :param X: The training input data, shape (n_samples, n_features).
        :param y: The target values for the training data, shape (n_samples,).
        :param j: Parameter dimension to update
        """
        z = self.predict(X)
        tree = GBMTree(
            max_depth=self.max_depth[j],
            min_samples_leaf=self.min_samples_leaf[j],
            dist=self.dist,
        )
        tree.fit_gradients(X=X, y=y, z=z, j=j)
        self.trees[j] += [tree]
        self.kappa[j] += 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict response values for the input data using the trained model.

        :param X: Input data matrix of shape (n_samples, n_features).
        :return: Predicted response values of shape (d,n_samples).
        """
        z_hat = np.zeros((len(self.z0), len(X)))
        for j in range(len(self.z0)):
            if len(self.trees[j]) > 0:
                z_hat[j] = self.eps[j] * sum(
                    [tree.predict(X) for tree in self.trees[j]]
                )
        return self.z0 + z_hat


if __name__ == "__main__":
    n = 100
    expected_loss = 166.58751236236634
    rng = np.random.default_rng(seed=10)
    X0 = np.arange(0, n)
    X1 = np.arange(0, n)
    rng.shuffle(X1)
    mu = 10 * (X0 > 0.3 * n) + 5 * (X1 > 0.5 * n)

    X = np.stack([X0, X1]).T
    y = rng.normal(mu, 1.5)

    kappa = [100, 0]
    gbm = CycGBM(dist="normal", kappa=kappa)
    gbm.fit(X, y)
    z_hat = gbm.predict(X)
