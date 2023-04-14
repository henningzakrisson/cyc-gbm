import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingMachine:
    """
    Class for standard 1D gradient boosting machine regressors
    """

    def __init__(
        self,
        kappa: int = 100,
        eps: float = 0.1,
        max_depth: int = 2,
        min_samples_leaf: int = 20,
    ):
        """
        :param kappa: Number of boosting steps.
        :param eps: Shrinkage factor, which scales the contribution of each tree.
        :param max_depth: Maximum depth of each decision tree.
        :param min_samples_leaf: Minimum number of samples required at a leaf node.
        """
        self.kappa = kappa
        self.eps = eps
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        # Assume normal distribition
        self.loss = lambda z, y: (y - z[0]) ** 2

        # Assume normal distribution
        self.grad = lambda z, y: z[0] - y

    def _train_tree(self, X: np.ndarray, y: np.ndarray, z: np.ndarray):
        """
        Train a decision tree regressor to predict the residual of the current model.

        :param z: Predicted values of the current model for the given input data X.
        :param y: True values of the response variable for the given input data X.
        :param X: Input data matrix of shape (n_samples, n_features).
        :return: Trained decision tree regressor.
        """
        tree = DecisionTreeRegressor(
            max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf
        )
        g = self.grad(z, y)
        tree.fit(X, -g)
        return tree

    # def train(self,X: np.ndarray, y: np.ndarray):
