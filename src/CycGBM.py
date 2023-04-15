import numpy as np
from sklearn.tree import DecisionTreeRegressor
from typing import List
from src.exceptions import UnknownDistribution


class CycGBM:
    """
    Class for cyclical gradient boosting machine regressors
    """

    def __init__(
        self,
        # Assume 2 dimensions
        kappas: List[int] = [100, 100],
        eps: List[float] = [0.1, 0.1],
        max_depths: List[int] = [2, 2],
        min_samples_leafs: List[int] = [20, 20],
        dist="normal",
    ):
        """
        :param kappas: Number of boosting steps.
        :param eps: Shrinkage factors, which scales the contribution of each tree.
        :param max_depths: Maximum depths of each decision tree.
        :param min_samples_leafs: Minimum number of samples required at a leaf node.
        :param dist: distribution for losses and gradients
        """
        self.kappas = kappas
        self.eps = eps
        self.max_depths = max_depths
        self.min_samples_leafs = min_samples_leafs
        self.dist = dist

        # Assume 2 dimensions
        self.z0s = [np.nan, np.nan]
        self.trees = [None, None]

        if self.dist == "normal":
            self.loss = lambda z, y: z[1] + 0.5 * np.exp(-2 * z[1]) * (y - z[0]) ** 2
            self.grads = [
                lambda z, y: -np.exp(-2 * z[1]) * (y - z[0]),
                lambda z, y: 1 - np.exp(-2 * z[1]) * (y - z[0]) ** 2,
            ]
        else:
            raise UnknownDistribution("Unknown distribution")

    def _initiate_param(self, y: np.ndarray) -> np.ndarray:
        """
        Compute the initial parameter estimate.

        :param y: Training responses, of shape (n_samples,).
        :return: Initial prediction for the response variable.
        """

        if self.dist == "normal":
            z0 = np.array([y.mean(), y.std()])
        return z0

    def _train_tree(self, X: np.ndarray, y: np.ndarray, z: np.ndarray, j: int):
        """
        Train a decision tree regressor to predict the residual of the current model.

        :param X: Input data matrix of shape (n_samples, n_features).
        :param y: True values of the response variable for the given input data X.
        :param z: Parameter estimates of the current model
        :param j: Parameter dimension to fit the tree to
        :return: Trained decision tree regressor.
        """
        tree = DecisionTreeRegressor(
            max_depth=self.max_depths[j], min_samples_leaf=self.min_samples_leafs[j]
        )
        g = self.grads[j](z, y)
        tree.fit(X, -g)

        return tree

    def _optimize_step_size(self, y: np.ndarray, z: np.ndarray, j: int) -> float:
        """
        Compute the optimal step size (gamma) for updating the predictions.

        :param y: True response values for the current iteration, of shape (n_samples,).
        :param z: Estimated parameters for the current iteration, of shape (n_samples,).
        :param j: Parameter dimension to update
        :return: Optimal step size for updating the parameter estimates.
        """
        if self.dist == "normal":
            if j == 0:
                gamma = np.mean(y - z)
            elif j == 1:
                gamma = 0.5 * np.log(np.mean((np.exp(-2 * z[1]) * (y - z[0]) ** 2)))
        return gamma

    def _adjust_node_values(
        self,
        tree: DecisionTreeRegressor,
        X: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        j: int,
    ) -> DecisionTreeRegressor:
        """
        Adjust the predicted node values of the decision tree to optimal step sizes

        :param tree: A decision tree regressor object to adjust the node values for
        :param X: The input training data for the model as a numpy array
        :param y: The output training data for the model as a numpy array
        :param z: The current parameter estimates
        :param j: Parameter dimension to update
        :return: The decision tree regressor object with adjusted node values
        """

        # Predict gradients
        g_hat = tree.predict(X)

        # Look at all unique gradient predictions values and adjust them to optimal step size
        gs = np.unique(g_hat)
        for g in gs:
            # Find optimal step size for this node
            index = g_hat == g
            gamma = self._optimize_step_size(y=y[index], z=z[:, index], j=j)
            # Manipulate tree
            tree.tree_.value[tree.tree_.value == g] = gamma

        return tree

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model on the given training data.

        :param X: Input data matrix of shape (n_samples, n_features).
        :param y: True response values for the input data, of shape (n_samples,).
        """
        self.z0 = self._initiate_param(y)

        # Assume 2 dimensions
        z = self.z0.repeat(len(y)).reshape((2, len(y)))
        self.trees = [[[]] * self.kappas[0], [[]] * self.kappas[1]]

        for k in range(0, max(self.kappas)):
            # Assume 2 dimensions
            for j in [0, 1]:
                if k >= self.kappas[j]:
                    continue
                tree = self._train_tree(X=X, y=y, z=z, j=j)
                tree = self._adjust_node_values(tree=tree, X=X, y=y, z=z, j=j)
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
        tree = self._train_tree(X=X, y=y, z=z, j=j)
        tree = self._adjust_node_values(tree=tree, X=X, y=y, z=z, j=j)
        self.trees += [tree]
        self.kappas[j] += 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict response values for the input data using the trained model.

        :param X: Input data matrix of shape (n_samples, n_features).
        :return: Predicted response values of shape (d,n_samples).
        """
        # Assume two dimensions
        z_hat = (
            self.z0.repeat(len(X)).reshape((2, len(X)))
            + (
                np.array(self.eps)
                * np.array(
                    [sum([tree.predict(X) for tree in trees]) for trees in self.trees]
                ).T
            ).T
        )
        return z_hat


if __name__ == "__main__":
    n = 100
    rng = np.random.default_rng(seed=10)
    X0 = np.arange(0, n)
    X1 = np.arange(0, n)
    rng.shuffle(X1)
    mu = 10 * (X0 > 0.3 * n) + 5 * (X1 > 0.5 * n)
    sigma = np.exp(1 + 1 * (X0 < 0.4 * n))

    X = np.stack([X0, X1]).T
    y = rng.normal(mu, sigma)

    kappas = [100, 10]
    eps = [0.1, 0.01]
    gbm = CycGBM(kappas=kappas, eps=eps)
    gbm.train(X, y)
    z_hat = gbm.predict(X)

    loss = gbm.loss(z_hat, y).sum()

    print(loss)
