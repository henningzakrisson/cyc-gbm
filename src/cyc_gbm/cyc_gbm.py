import numpy as np
from typing import List, Union
from src.cyc_gbm.distributions import initiate_distribution
from src.cyc_gbm.gbm_tree import GBMTree
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger.handlers[0].setFormatter(formatter)


class CycGBM:
    """
    Class for cyclical gradient boosting machine regressors
    """

    def __init__(
        self,
        kappa: Union[int, List[int]] = 100,
        eps: Union[float, List[float]] = 0.1,
        max_depth: Union[int, List[int]] = 2,
        min_samples_leaf: Union[int, List[int]] = 20,
        dist: str = "normal",
    ):
        """
        :param kappa: Number of boosting steps. Dimension-wise or global for all parameter dimensions.
        :param eps: Shrinkage factors, which scales the contribution of each tree. Dimension-wise or global for all parameter dimensions.
        :param max_depth: Maximum depths of each decision tree. Dimension-wise or global for all parameter dimensions.
        :param min_samples_leaf: Minimum number of samples required at a leaf node. Dimension-wise or global for all parameter dimensions.
        :param dist: distribution for losses and gradients
        """
        self.dist = initiate_distribution(dist=dist)
        self.d = self.dist.d
        self.kappa = kappa if isinstance(kappa, list) else [kappa] * self.d
        self.eps = eps if isinstance(eps, list) else [eps] * self.d
        self.max_depth = (
            max_depth if isinstance(max_depth, list) else [max_depth] * self.d
        )
        self.min_samples_leaf = (
            min_samples_leaf
            if isinstance(min_samples_leaf, list)
            else [min_samples_leaf] * self.d
        )

    def fit(self, X: np.ndarray, y: np.ndarray, log: bool = False) -> None:
        """
        Train the model on the given training data.

        :param X: Input data matrix of shape (n_samples, n_features).
        :param y: True response values for the input data, of shape (n_samples,).
        """
        if log:
            logger.info("Initializing model")
        self.z0 = self.dist.mle(y)[:, None]

        z = np.tile(self.z0, (1, len(y)))
        self.trees = [[None] * self.kappa[j] for j in range(self.d)]
        for k in range(0, max(self.kappa)):
            for j in range(self.d):
                if log:
                    logger.info(f"Boosting step {k+1}, parameter dimension {j+1}")
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
        if log:
            logger.info("Finished training")

    def update(self, X: np.ndarray, y: np.ndarray, j: int) -> None:
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
        z_hat = np.zeros((self.d, len(X)))
        for j in range(self.d):
            if len(self.trees[j]) > 0:
                z_hat[j] = self.eps[j] * sum(
                    [tree.predict(X) for tree in self.trees[j]]
                )
        return self.z0 + z_hat

    def feature_importances(self, j: Union[str,int] = 'all', normalize: bool = True) -> np.ndarray:
        """
        Computes the feature importances for parameter dimension j

        :param j: Parameter dimension. If 'all', calculate importance over all parameter dimensions.
        :return: Feature importance of shape (n_features,)
        """
        if j == 'all':
            feature_importances = np.array([[tree.feature_importances() for tree in self.trees[j]] for j in range(self.d)]).sum(axis=(0,1))
        else:
            feature_importances = np.array([tree.feature_importances() for tree in self.trees[j]]).sum(axis=0)
        if normalize:
            feature_importances /= feature_importances.sum()
        return feature_importances


if __name__ == "__main__":
    rng = np.random.default_rng(seed=10)
    n = 10000
    p = 5
    X = np.concatenate([np.ones((1, n)), rng.normal(0, 1, (p - 1, n))]).T
    z0 = (
            1.5 * X[:, 1]
            + 2 * X[:, 2]
    )
    z1 = 1 + 1.2 * X[:, 1]
    z = np.stack([z0, z1])
    distribution = initiate_distribution(dist="normal")
    y = distribution.simulate(z=z, random_state=5)

    kappa = 100
    eps = 0.1
    max_depth = 2
    gbm = CycGBM(dist="normal", kappa=kappa, eps=eps, max_depth=max_depth)
    gbm.fit(X, y)

    feature_importances = gbm.feature_importances(j=0)
    print(feature_importances.round(5))
    feature_importances = gbm.feature_importances(j=1)
    print(feature_importances.round(5))
    feature_importances = gbm.feature_importances(j='all')
    print(feature_importances.round(5))