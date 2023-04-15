import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from src.exceptions import UnknownDistribution


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
        dist: str = "normal",
    ):
        """
        :param kappa: Number of boosting steps.
        :param eps: Shrinkage factor, which scales the contribution of each tree.
        :param max_depth: Maximum depth of each decision tree.
        :param min_samples_leaf: Minimum number of samples required at a leaf node.
        :param dist: distribution for losses and gradients
        """
        self.kappa = kappa
        self.eps = eps
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.dist = dist

        self.z0 = np.nan
        self.trees = None

        if self.dist == "normal":
            # The normal distribution uses a mean-dispersion parametrization
            self.loss = lambda z, y: (y - z) ** 2
            self.grad = lambda z, y: z - y
        elif self.dist == "gamma":
            # The gamma distribution uses a mean-dispersion parametrization
            self.loss = lambda z, y: y * np.exp(-z) + z
            self.grad = lambda z, y: 1 - y * np.exp(-z)
        else:
            raise UnknownDistribution("Unknown distribution")

    def _initiate_param(self, y: np.ndarray) -> float:
        """
        Compute the initial parameter estimate

        :param y: Training responses, of shape (n_samples,).
        :return: Initial parameter estimate.
        """

        if self.dist == "normal":
            z0 = y.mean()
        elif self.dist == "gamma":
            z0 = np.log(y.mean())
        return z0

    def _train_tree(self, X: np.ndarray, y: np.ndarray, z: np.ndarray):
        """
        Train a decision tree regressor to predict the residual of the current model.

        :param X: Input data matrix of shape (n_samples, n_features).
        :param y: True values of the response variable for the given input data X.
        :param z: Parameter estimates of the current model
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
        if self.dist == "normal":
            gamma = np.mean(y - z)
        elif self.dist == "gamma":
            gamma = np.log((y * np.exp(-z)).mean())
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
            gamma = self._optimize_step_size(y=y[index], z=z[index])
            # Manipulate tree
            tree.tree_.value[tree.tree_.value == g] = gamma

        return tree

    def _train_trees(self, X: np.ndarray, y: np.ndarray, z: np.ndarray, kappa: int):
        """
        Trains kappa trees using the training data X and y and the current model's parameter estimates z as inputs.

        Args:
            X: The input training data, shape (n_samples, n_features).
            y: The target values for the training data, shape (n_samples,).
            z: The current model's parameter estimates
            kappa: The number of trees to train.
        """
        trees = [[]] * kappa
        for k in range(0, kappa):
            tree = self._train_tree(X=X, y=y, z=z)
            tree = self._adjust_node_values(tree=tree, X=X, y=y, z=z)
            z += self.eps * tree.predict(X)
            trees[k] = tree

        return trees

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model on the given training data.

        :param X: Input data matrix of shape (n_samples, n_features).
        :param y: True response values for the input data, of shape (n_samples,).
        """
        self.z0 = self._initiate_param(y)
        z = self.z0.repeat(len(y))

        self.trees = self._train_trees(X=X, y=y, z=z, kappa=self.kappa)

    def update(self, X: np.ndarray, y: np.ndarray, k_add: int = 100):
        """
        Updates the current boosting model with additional trees, trained on the specified training data X and y.

        :param X: The training input data, shape (n_samples, n_features).
        :param y: The target values for the training data, shape (n_samples,).
        :param k_add: The number of trees to add to the model. Defaults to 100.
        """
        z = self.predict(X)

        self.trees += self._train_trees(X=X, y=y, z=z, kappa=k_add)
        self.kappa += k_add

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict response values for the input data using the trained model.

        :param X: Input data matrix of shape (n_samples, n_features).
        :return: Predicted parameter estimates of shape (n_samples,).
        """
        z_hat = np.ones(len(X)) * self.z0 + self.eps * sum(
            [tree.predict(X) for tree in self.trees]
        )
        return z_hat


def tune_kappa(
    X: np.ndarray,
    y: np.ndarray,
    kappa_max: int = 1000,
    eps: int = 0.1,
    max_depth: int = 2,
    min_samples_leaf: int = 20,
    dist="normal",
    n_splits: int = 4,
    random_state=None,
):
    """Tunes the kappa parameter of a UniGBM model using k-fold cross-validation.

    :param X: The input data matrix of shape (n_samples, n_features).
    :param y: The target vector of shape (n_samples,).
    :param kappa_max: The maximum value of the kappa parameter to test.
    :param eps: The epsilon parameter for the UniGBM model.
    :param max_depth: The maximum depth of the decision trees in the UniGBM model.
    :param min_samples_leaf: The minimum number of samples required to be at a leaf node in the UniGBM model.
    :param dist: The distribution of the target variable.
    :param n_splits: The number of folds to use for k-fold cross-validation.
    :param random_state: The random state to use for the k-fold split.
    :return: The optimal value of the kappa parameter."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    loss = np.ones((n_splits, kappa_max)) * np.nan

    for i, idx in enumerate(kf.split(X)):
        idx_train, idx_valid = idx
        X_train, y_train = X[idx_train], y[idx_train]
        X_valid, y_valid = X[idx_valid], y[idx_valid]

        gbm = UniGBM(
            kappa=0,
            eps=eps,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            dist=dist,
        )
        gbm.train(X_train, y_train)

        for k in range(0, kappa_max):
            gbm.update(X_train, y_train, 1)
            loss[i, k] = gbm.loss(gbm.predict(X_valid), y_valid).sum()
            if loss[i, k] > loss[i, k - 1]:
                loss[i, k:] = loss[i, k]
                break

    kappa = np.argmin(loss.sum(axis=0))

    return kappa


if __name__ == "__main__":
    n = 1000
    rng = np.random.default_rng(seed=10)
    X0 = np.arange(0, n)
    X1 = np.arange(0, n)
    rng.shuffle(X1)
    mu = 10 * (X0 > 0.3 * n) + 5 * (X1 > 0.5 * n)

    X = np.stack([X0, X1]).T
    y = rng.normal(mu, 1.5)

    kappa = tune_kappa(X=X, y=y, random_state=5)
    print(kappa)
