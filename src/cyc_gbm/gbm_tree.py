import numpy as np
from sklearn.tree import DecisionTreeRegressor
from .distributions import Distribution


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
        dist: Distribution,
    ):
        """
        Constructs a new GBMTree instance.

        :param max_depth: The maximum depth of the tree.
        :param min_samples_leaf: The minimum number of samples required to be at a leaf node.
        :param dist: The distribution used for calculating the gradients and losses
        """
        super().__init__(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        self.dist = dist

    def _adjust_node_value(
        self, X: np.ndarray, y: np.ndarray, z: np.ndarray, j: int, node_index: int
    ) -> None:
        """
        Adjust the predicted node values of an individual node to its optimal step size.

        :param X: The input training data for the model as a numpy array
        :param y: The output training data for the model as a numpy array
        :param z: The current parameter estimates
        :param j: Parameter dimension to update
        :param node_index: The index of the node to update
        """
        feature = self.tree_.feature[node_index]
        threshold = self.tree_.threshold[node_index]
        index = X[:, feature] <= threshold
        g_0 = self.tree_.value[node_index]
        g_opt = self.dist.opt_step(y=y[index], z=z[:, index], j=j, g_0=g_0)
        self.tree_.value[node_index] = g_opt

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
        """
        g_hat = self.predict(X)
        gs = np.unique(g_hat)
        for g in gs:
            # Find optimal step size for this node
            index = g_hat == g
            g_opt = self.dist.opt_step(y=y[index], z=z[:, index], j=j, g_0=g)
            # Manipulate tree
            self.tree_.value[self.tree_.value == g] = g_opt

    def fit_gradients(
        self, X: np.ndarray, y: np.ndarray, z: np.ndarray, j: int
    ) -> None:
        """
        Fits the GBMTree to the negative gradients and adjusts node values to minimize loss.

        :param X: The training input samples.
        :param y: The target values.
        :param z: The predicted parameter values from the previous iteration.
        :param j: The index of the current iteration.
        """
        g = self.dist.grad(y=y, z=z, j=j)
        self.fit(X, -g)
        self._adjust_node_values(X=X, y=y, z=z, j=j)
