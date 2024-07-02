from typing import List, Union

import numpy as np
from scipy.optimize import minimize

from cyc_gbm.utils.distributions import Distribution


class CyclicGeneralizedLinearModel:
    def __init__(
        self,
        distribution: Distribution,
        max_iter: int = 1000,
        tol: float = 1e-5,
        eps: Union[List[float], float] = 1e-7,
    ):
        """
        Initialize the model.
        """
        self.distribution = distribution
        self.d = self.distribution.n_dim
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.beta = None
        self.z0 = None

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        """
        Fit the model.
        """
        z = np.zeros((self.d, len(y)))
        self.z0 = minimize(
            fun=lambda z0: self.distribution.loss(y=y, z=z0[:, None] + z, w=w).sum(),
            x0=self.distribution.mme(y=y, w=w),
        )["x"]
        z = np.tile(self.z0, (X.shape[0], 1)).T
        p = X.shape[1]

        beta = np.zeros((self.max_iter, self.d, p))

        for i in range(self.max_iter):
            for j in range(self.d):
                g = self.distribution.grad(y=y, z=z, w=w, j=j)
                # Update patameter estimate
                beta[i, j] = beta[i - 1, j] - self.eps * g @ X
                # Update parameter estimate
                z[j] = self.z0[j] + beta[i, j] @ X.T

            # Check convergence
            if i > 0 and np.linalg.norm(beta[i] - beta[i - 1]) < self.tol:
                break

            if i == self.max_iter - 1:
                Warning("CGLM model did not converge")

        self.beta = beta[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the response for the given input data.
        """
        z = np.zeros((self.d, X.shape[0]))
        for j in range(self.d):
            z[j] = self.z0[j] + self.beta[j] @ X.T
        return z
