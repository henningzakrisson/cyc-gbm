import numpy as np
from typing import List, Union
from src.cyc_gbm.distributions import initiate_distribution

class CycGLM:
    """
    Class for cyclical generalized linear model regressors.
    """

    def __init__(
            self,
            iter_max: int = 1000,
            tol: float = 1e-2,
            eps: float: 1e-7,
            dist: str = "normal",
    ):
        """
        Initialize the model.

        :param iter_max: Maximum number of iterations.
        :param tol: Tolerance for convergence.
        :param eps: Learning rate.
        :param dist: distribution for losses and gradients
        """

        self.dist = initiate_distribution(dist=dist)
        self.d = self.dist.d

    def fit(self, X: np.ndarray,
            y: np.ndarray,
            log: bool = False,
            ) -> None:
        """
        Train the model on the given training data.

        :param X: Input data matrix of shape (n_samples, n_features).
        :param y: True response values for the input data, of shape (n_samples,).
        """
        z0 = self.dist.mle(y)[:, None]
        p = X.shape[1]

        beta_hat = np.zeros((BMax,d,p))
        beta_hat[0,0,0] = z0[0]
        beta_hat[0,1,0] = z0[1]

        for i in range(self.iter_max):
            for j in range(self.d):
                g = self.dist.grad[j](y = y, z = z)
                for k in range(p):
                    beta_hat[i, j, k] = betaHat[i - 1, j, k] - self.eps[j] * sum(X[:, k] * g)
                # Update score
                z[j] = beta_hat[b, j] @ X.T

            # Check absolute change in all parameters
            if i > 0:
                if np.sum(np.abs(beta_hat[b, :, :] - beta_hat[b - 1, :, :]) / np.abs(beta_hat[b - 1, :, :])) < self.tol:
                    break

        self.beta = beta_hat[b, :, :]

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
