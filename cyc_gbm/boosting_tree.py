import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize

from cyc_gbm.utils.distributions import Distribution


class BoostingTree(DecisionTreeRegressor):
    """
    A Gradient Boosting Machine tree.
    It is a subclass of sklearn.tree.DecisionTreeRegressor that first computes the current negative gradient of the loss function and then fits a regression tree to the negative gradient.
    Then, the node values of the tree are adjusted to the step size that minimizes the loss.

    :param max_depth: The maximum depth of the tree.
    :param min_samples_leaf: The minimum number of samples required for a split to be valid.
    :param distribution: The distribution function used for calculating the gradients and optimal step sizes.
    """

    def __init__(
        self,
        distribution: Distribution,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
    ):
        """
        Constructs a new BoostingTree instance.

        :param distribution: The distribution used for calculating the gradients and losses
        :param max_depth: The maximum depth of the tree.
        :param min_samples_split: The minimum number of samples required to split an internal node.
        :param min_samples_leaf: The minimum number of samples required to be at a leaf node.
        """
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )
        self.distribution = distribution

    def fit_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        w: np.ndarray,
        j: int,
    ) -> None:
        """
        Fits the BoostingTree to the negative gradients and adjusts node values to minimize loss.

        :param X: The training input samples.
        :param y: The target values.
        :param z: The predicted parameter values from the previous iteration.
        :param w: Weights for the training data, of shape (n_samples,).
        :param j: The parameter dimension to update.
        """
        g = self.distribution.grad(y=y, z=z, w=w, j=j)
        self.fit(X, -g)
        self._adjust_node_values(X=X, y=y, z=z, w=w, j=j)

    def _adjust_node_values(
        self,
        X: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        w: np.ndarray,
        j: int,
        node_index: int = 0,
    ) -> None:
        """
        Adjust the predicted node values of the node outputs to its optimal step size.
        Adjustment is performed recursively starting at the top of the tree.
        The impurity is also changed to the loss of the new node values.

        :param X: The input training data for the model as a numpy array
        :param y: The output training data for the model as a numpy array
        :param z: The current parameter estimates
        :param w: Weights for the training data, of shape (n_samples,). Default is 1 for all samples.
        :param j: Parameter dimension to update
        :param node_index: The index of the node to update
        """
        # Optimize node and update impurity
        node_value_0 = self.tree_.value[node_index][0][0]
        node_value = self._optimize_node_value(
            y=y, z=z, w=w, j=j, node_value_0=node_value_0
        )
        self.tree_.value[node_index] = node_value
        e = np.eye(self.distribution.n_dim)[:, j : j + 1]  # Indicator vector
        self.tree_.impurity[node_index] = self.distribution.loss(
            y=y, z=z + e * node_value, w=w
        ).sum()

        # Tend to the children
        feature = self.tree_.feature[node_index]
        if feature == -2:
            # This is a leaf
            return
        threshold = self.tree_.threshold[node_index]
        index_left = X[:, feature] <= threshold
        child_left = self.tree_.children_left[node_index]
        child_right = self.tree_.children_right[node_index]
        self._adjust_node_values(
            X=X[index_left],
            y=y[index_left],
            z=z[:, index_left],
            w=w[index_left],
            j=j,
            node_index=child_left,
        )
        self._adjust_node_values(
            X=X[~index_left],
            y=y[~index_left],
            z=z[:, ~index_left],
            w=w[~index_left],
            j=j,
            node_index=child_right,
        )

    def _optimize_node_value(
        self,
        y: np.ndarray,
        z: np.ndarray,
        w: np.ndarray,
        j: int,
        node_value_0: float = 0,
    ):
        """
        Numerically optimize the node value for the data in specified dimension

        :param y: Target values.
        :param z: Current parameter estimates.
        :param w: Weights of the observations.
        :param j: Index of the dimension to optimize.
        :param node_value_0: Initial guess for the optimal step size. Default is 0.
        :return: The optimal node value size.
        """
        # Indicator vector for adding step to dimension j
        e = np.eye(self.distribution.n_dim)[:, j : j + 1]
        node_value = minimize(
            fun=lambda step: self.distribution.loss(y=y, z=z + e * step, w=w).sum(),
            jac=lambda step: self.distribution.grad(
                y=y, z=z + e * step, j=j, w=w
            ).sum(),
            x0=node_value_0,
        )["x"][0]
        return node_value

    def compute_feature_importances(self) -> np.ndarray:
        """
        Returns the feature importances of the tree.

        :return: The feature importances of the tree.
        """
        return self.tree_.compute_feature_importances(normalize=False)
