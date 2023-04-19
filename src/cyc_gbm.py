import numpy as np
from sklearn.tree import DecisionTreeRegressor
from typing import List, Union
from src.distributions import initiate_dist
from sklearn.model_selection import KFold
import warnings


class GBMTree(DecisionTreeRegressor):
    """
    A Gradient Boosting Machine tree.

    :param max_depth: The maximum depth of the tree.
    :param min_samples_leaf: The minimum number of samples required to be at a leaf node.
    :param dist: The distribution function used for calculating the gradients.

    """

    def __init__(
        self,
        max_depth: int,
        min_samples_leaf: int,
        dist,  # TODO: Add type annotation here when the wrapper class is introduced"?
    ):
        """
        Constructs a new GBMTree instance.

        :param max_depth: The maximum depth of the tree.
        :param min_samples_leaf: The minimum number of samples required to be at a leaf node.
        :param dist: The distribution function used for calculating the gradients.
        """
        super().__init__(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        self.dist = dist

    def _adjust_node_values(
        self,
        X: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        j: int,
    ) -> None:
        """
        Adjust the predicted node values of the decision tree to optimal step sizes

        :param X: The input training data for the model as a numpy array
        :param y: The output training data for the model as a numpy array
        :param z: The current parameter estimates
        :param j: Parameter dimension to update
        :return: The decision tree regressor object with adjusted node values
        """
        g_hat = self.predict(X)
        gs = np.unique(g_hat)
        for g in gs:
            # Find optimal step size for this node
            index = g_hat == g
            g_opt = self.dist.opt_step(y=y[index], z=z[:, index], j=j, g_0=g)
            # Manipulate tree
            self.tree_.value[self.tree_.value == g] = g_opt

    def fit_gradients(self, X, y, z, j: int) -> None:
        """
        Fits the GBMTree to the gradients and adjusts node values to minimize loss

        :param X: The training input samples.
        :param y: The target values.
        :param z: The predicted parameter values from the previous iteration.
        :param j: The index of the current iteration.
        """
        g = self.dist.grad(z=z, y=y, j=j)
        self.fit(X, -g)
        self._adjust_node_values(X=X, y=y, z=z, j=j)


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
        dist="normal",
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

        self.dist = initiate_dist(dist)

        # Assume 2 dimensions
        self.z0s = [np.nan, np.nan]
        self.trees = [None, None]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model on the given training data.

        :param X: Input data matrix of shape (n_samples, n_features).
        :param y: True response values for the input data, of shape (n_samples,).
        """
        self.z0 = self.dist.mle(y)[:, None]

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


def tune_kappa(
    X: np.ndarray,
    y: np.ndarray,
    kappa_max: Union[int, List[int]] = 1000,
    eps: Union[float, List[float]] = 0.1,
    max_depth: Union[int, List[int]] = 2,
    min_samples_leaf: Union[int, List[int]] = 20,
    dist="normal",
    n_splits: int = 4,
    random_state=None,
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
        loss[i, 0, :] = gbm.dist.loss(z_valid, y_valid).sum()

        for k in range(1, max(kappa_max) + 1):
            # Assume 2 dimensions
            for j in [0, 1]:
                if k < kappa_max[j]:
                    gbm.update(X=X_train, y=y_train, j=j)
                    z_valid[j] += gbm.eps[j] * gbm.trees[j][-1].predict(X_valid)
                    loss[i, k, j] = gbm.dist.loss(z_valid, y_valid).sum()
                else:
                    if j == 0:
                        loss[i, k, j] = loss[i, k - 1, j + 1]
                    else:
                        loss[i, k, j] = loss[i, k, j - 1]

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
    # Asssume two dimensions
    for j in range(0, 2):
        if did_not_converge[j] and kappa_max[j] > 0:
            warnings.warn(f"Tuning did not converge for dimension {j}")
            kappa[j] = kappa_max[j]
    # TODO: remove the loss return (it is only for testing purposes)
    return kappa, loss


if __name__ == "__main__":
    n = 1000
    rng = np.random.default_rng(seed=10)
    X0 = np.arange(0, n) / n
    X1 = np.arange(0, n) / n
    rng.shuffle(X1)
    mu = np.exp(1 * (X0 > 0.3 * n) + 0.5 * (X1 > 0.5 * n))
    v = np.exp(1 + 1 * X0 - 3 * np.abs(X1))

    X = np.stack([X0, X1]).T
    alpha = mu * (1 + v)
    beta = v + 2
    y0 = rng.beta(alpha, beta)
    y = y0 / (1 - y0)

    max_depth = 2
    min_samples_leaf = 20
    eps = [0.1, 0.1]
    kappa = [20, 100]

    gbm = CycGBM(
        kappa=kappa,
        eps=eps,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        dist="beta_prime",
    )
    gbm.fit(X, y)
    z_hat = gbm.predict(X)

    print(f"new model loss: {gbm.dist.loss(z_hat, y).sum().round(2)}")
