import numpy as np
from sklearn.tree import DecisionTreeRegressor


class UniGBM:
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

        self.z0 = np.nan
        self.trees = [[]] * self.kappa

        # Assume normal distribition
        self.loss = lambda z, y: (y - z) ** 2

        # Assume normal distribution
        self.grad = lambda z, y: z - y

    def _initiate_param(self, y: np.ndarray) -> float:
        """
        Compute the initial parameter estimate

        :param y: Training responses, of shape (n_samples,).
        :return: Initial prediction for the response variable.
        """

        # Assume normal distribution
        return y.mean()

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

    def _optimize_step_size(self, y: np.ndarray, z: np.ndarray) -> float:
        """
        Compute the optimal step size (gamma) for updating the predictions.

        :param y: True response values for the current iteration, of shape (n_samples,).
        :param z: Estimated parameters for the current iteration, of shape (n_samples,).
        :return: Optimal step size for updating the parameter estimates.
        """
        # Assume normal distribution
        gamma = np.mean(y - z)
        return gamma

    def _adjust_node_values(
        self, tree: DecisionTreeRegressor, X: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> DecisionTreeRegressor:
        """
        Adjust the predicted node values of the decision tree to optimal step sizes

        :param tree: A decision tree regressor object to adjust the node values for
        :param X: The input training data for the model as a numpy array
        :param y: The output training data for the model as a numpy array
        :param z: The current parameter estimates
        :return: The decision tree regressor object with adjusted node values
        """

        # Predict gradients
        g_hat = tree.predict(X)

        # Look at all unique gradient predictions values and adjust them to optimal step size
        gs = np.unique(g_hat)
        for g in gs:
            # Find optimal step size for this node
            index = g_hat == g
            gamma = self._optimize_step_size(y[index], z[index])
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
        z = self.z0.repeat(len(y))

        for k in range(0, self.kappa):
            tree = self._train_tree(X=X, y=y, z=z)
            tree = self._adjust_node_values(tree=tree, X=X, y=y, z=z)
            z += self.eps * tree.predict(X)
            self.trees[k] = tree

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict response values for the input data using the trained model.

        :param X: Input data matrix of shape (n_samples, n_features).
        :return: Predicted response values of shape (n_samples,).
        """
        z_hat = self.z0 + self.eps * sum([tree.predict(X) for tree in self.trees])
        return z_hat


if __name__ == "__main__":
    rng = np.random.default_rng(seed=10)
    X0 = np.arange(0, 100)
    X1 = np.arange(0, 100)
    rng.shuffle(X1)
    mu = 10 * (X0 > 30) + 5 * (X1 > 50)

    X = np.stack([X0, X1]).T
    y = rng.normal(mu, 1.5)

    gbm = UniGBM()
    gbm.train(X, y)
    y_hat = gbm.predict(X)

    print(sum(y - y_hat) ** 2)
