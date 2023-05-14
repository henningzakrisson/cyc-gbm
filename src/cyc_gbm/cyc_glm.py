import numpy as np
from typing import List, Union
from src.cyc_gbm.distributions import initiate_distribution, Distribution


class CycGLM:
    """
    Class for cyclical generalized linear model regressors.
    """

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-5,
        eps: Union[List[float], float] = 1e-7,
        distribution: Union[str,Distribution] = "normal",
    ):
        """
        Initialize the model.

        :param max_iter: Maximum number of iterations.
        :param tol: Tolerance for convergence.
        :param eps: Learning rate.
        :param distribution: distribution for losses and gradients
        :param dist: distribution for losses and gradients
        """
        if isinstance(distribution, str):
            self.dist = initiate_distribution(distribution=distribution)
        else:
            self.dist = distribution
        self.d = self.dist.d
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps if isinstance(eps, list) else [eps] * self.d
        self.beta = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: Union[np.ndarray, float] = 1,
    ) -> None:
        """
        Train the model on the given training data.

        :param X: Input data matrix of shape (n_samples, n_features).
        :param y: True response values for the input data, of shape (n_samples,).
        :param w: Weights for the training data, of shape (n_samples,). Default is 1 for all samples.
        """
        z0 = self.dist.mle(y)[:, None]
        z = np.tile(z0, (1, len(y)))
        p = X.shape[1]

        beta_hat = np.zeros((self.max_iter, self.d, p))
        for j in range(self.d):
            beta_hat[0, j, 0] = z0[j]

        for i in range(self.max_iter):
            for j in range(self.d):
                g = self.dist.grad(y=y, z=z, w=w, j=j)
                beta_hat[i, j] = beta_hat[i - 1, j] - self.eps[j] * g @ X
                # Update score
                z[j] = beta_hat[i, j] @ X.T

            if i > 0:
                if (
                    np.sum(
                        np.abs(beta_hat[i] - beta_hat[i - 1]) / np.abs(beta_hat[i - 1])
                    )
                    < self.tol
                ):
                    break

        self.beta = beta_hat[i]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the response for the input data.

        :param X: Input data matrix of shape (n_samples, n_features).
        :return: Predicted response of shape (n_samples,).
        """
        z = np.zeros((self.d, X.shape[0]))
        for j in range(self.d):
            z[j] = self.beta[j] @ X.T

        return z


if __name__ == "__main__":
    rng = np.random.default_rng(seed=10)
    n = 10000
    p = 5
    X = np.concatenate([np.ones((1, n)), rng.normal(0, 1, (p - 1, n))]).T
    z0 = 2 + 4 * X[:, 1]
    z1 = 1 + 0.2 * X[:, 2]
    z = np.stack([z0, z1])
    distribution = initiate_distribution(distribution="normal")
    y = distribution.simulate(z=z, random_state=5)

    max_iter = 10000
    eps = 1e-5
    tol = 1e-5
    glm = CycGLM(max_iter=max_iter, eps=eps, tol=tol, distribution="normal")
    glm.fit(X, y)

    loss_glm = distribution.loss(y=y, z=glm.predict(X)).mean()
    z0 = distribution.mle(y=y)[:, None]
    loss_intercept = distribution.loss(y=y, z=z0).mean()
    loss_true = distribution.loss(y=y, z=z).mean()
    print(f"Loss true: {loss_true}")
    print(f"Loss intercept: {loss_intercept}")
    print(f"Loss GLM: {loss_glm}")
