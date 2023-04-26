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

    def _adjust_node_values(
        self, X: np.ndarray, y: np.ndarray, z: np.ndarray, j: int, node_index: int = 0
    ) -> None:
        """
        Adjust the predicted node values of the node outputs to its optimal step size.
        Adjustment is performed recursively starting at the top of the tree.
        The impurity is also changed to the loss of the new node values.

        :param X: The input training data for the model as a numpy array
        :param y: The output training data for the model as a numpy array
        :param z: The current parameter estimates
        :param j: Parameter dimension to update
        :param node_index: The index of the node to update
        """
        if node_index == -1:
            # This is nota node, but the child of a leaf
            return
        # Optimize node and update impurity
        g_0 = self.tree_.value[node_index]
        g_opt = self.dist.opt_step(y=y, z=z, j=j, g_0=g_0)
        self.tree_.value[node_index] = g_opt
        e = np.eye(self.dist.d)[:, j:j + 1] # Indicator vector
        self.tree_.impurity[node_index] = self.dist.loss(y=y, z=z + e * g_opt).sum()

        # Tend to the children
        feature = self.tree_.feature[node_index]
        threshold = self.tree_.threshold[node_index]
        index_left = X[:, feature] <= threshold
        child_left = self.tree_.children_left[node_index]
        child_right = self.tree_.children_right[node_index]
        self._adjust_node_values(X=X[index_left], y=y[index_left], z=z[:, index_left], j=j, node_index=child_left)
        self._adjust_node_values(X=X[~index_left], y=y[~index_left], z=z[:, ~index_left], j=j, node_index=child_right)

    def feature_importances(self) -> np.ndarray:
        """
        Returns the feature importances of the tree.

        :return: The feature importances of the tree.
        """
        return self.tree_.compute_feature_importances(normalize=False)

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
