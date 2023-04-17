import numpy as np
from typing import Union
from src.exceptions import UnknownDistribution
from scipy.special import loggamma, polygamma
from scipy.optimize import minimize


class Distribution:
    """
    Class for distributions for the GBM classes
    """

    def __init__(
        self,
        dist: str,
        d: int = 2,
    ):
        """
        Initialize a distribution object.

        :param dist: The distribution name.
        :param d: The dimensionality of the distribution (default=2).
        :raises UnknownDistribution: If`distribution is not implemented
        """
        if dist not in ["normal", "gamma"]:
            raise UnknownDistribution("Unknown distribution")
        self.dist = dist
        self.d = d

    def loss(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates the loss of the parameter estimates and the response.

        :param z: The predicted parameters.
        :param y: The target values.
        :return: The loss function value(s) for the given `z` and `y`.
        """
        if self.dist == "normal":
            if self.d == 1:
                return (y - z) ** 2
            elif self.d == 2:
                return z[1] + 0.5 * np.exp(-2 * z[1]) * (y - z[0]) ** 2
        elif self.dist == "gamma":
            if self.d == 1:
                return y * np.exp(-z) + z
            elif self.d == 2:
                return loggamma(np.exp(-z[1])) + np.exp(-z[1]) * (
                    y * np.exp(-z[0]) - np.log(y) + z[0] + z[1]
                )

    def grad(self, z: np.ndarray, y: np.ndarray, j: int = 0) -> np.ndarray:
        """
        Calculates the gradients of the loss function with respect to the parameters.

        :param z: The predicted parameters.
        :param y: The target values.
        :param j: The parameter dimension to compute the gradient for (default=0).
        :return: The gradient(s) of the loss function for the given `z`, `y`, and `j`.
        """
        if self.dist == "normal":
            if self.d == 2:
                if j == 0:
                    return -np.exp(-2 * z[1]) * (y - z[0])
                elif j == 1:
                    return 1 - np.exp(-2 * z[1]) * (y - z[0]) ** 2
            elif self.d == 1:
                return z - y
        elif self.dist == "gamma":
            if self.d == 2:
                if j == 0:
                    return np.exp(-z[1]) * (1 - y * np.exp(-z[0]))
                elif j == 1:
                    np.exp(-z[1]) * (
                        1
                        + np.log(y)
                        - z[0]
                        - z[1]
                        - y * np.exp(-z[0])
                        - polygamma(0, np.exp(-z[1]))
                    )
            elif self.d == 1:
                return 1 - y * np.exp(-z)

    def _mle_numeric(
        self, y: np.ndarray, z: np.ndarray, j: int, z_j_0: np.ndarray = np.array([0, 0])
    ):
        """Compute maximum likelihood estimator numerically

        :param y: Target values.
        :param z: Current parameter estimates.
        :param j: Index of the dimension to optimize.
        :param z_j_0: Initial guess for the MLE
        :return: The MLE of the loss for this dimension

        """
        # Dimension indicator (assume two dimensions)
        if j == 0:
            e = np.array([1, 0])
        elif j == 1:
            e = np.array([0, 1])

        to_min = lambda z_j: self.loss(z + e[j] * z_j, y).sum()
        z_j_opt = minimize(to_min, z_j_0)["x"][0]

        return z_j_opt

    def mle(self, y: np.ndarray) -> Union[float, np.ndarray]:
        """
        Calculates the d-dimensional maximum likelihood estimator of the observations.

        :param y: The target values.
        :return: The maximum likelihood estimator of the parameters.
        """
        if self.dist == "normal":
            if self.d == 2:
                return np.array([y.mean(), np.log(y.std())])
            elif self.d == 1:
                return y.mean()
        elif self.dist == "gamma":
            if self.d == 2:
                z_0 = np.log(y.mean())
                z_j_0 = np.log(y.var() / (y.mean() ** 2))
                z_1 = self._mle_numeric(y, np.array([z_0, 0]), j=1, z_j_0=z_j_0)
                return np.array([z_0, z_1])
            elif self.d == 1:
                return np.log(y.mean())

    def _opt_step_numeric(self, y: np.ndarray, z: np.ndarray, j: int, g_0: float = 0):
        """
        Compute the optimal step size for the data in specific dimension

        :param y: Target values.
        :param z: Current parameter estimates.
        :param j: Index of the dimension to optimize.
        :param g_0: Initial guess for the optimal step size. Default is 0.
        :return: The optimal step size.
        """
        # Dimension indicator (assume two dimensions)
        if j == 0:
            e = np.array([1, 0])
        elif j == 1:
            e = np.array([0, 1])

        to_min = lambda gamma: self.loss(z + e * gamma, y).sum()
        gamma_opt = minimize(to_min, g_0)["x"][0]
        return gamma_opt

    def opt_step(self, y: np.ndarray, z: np.ndarray, j: int = 0, g_0=0) -> np.ndarray:
        """
        Calculate the optimal step length for these parameter estimates and responses

        :param y: The target values
        :param z: The current parameter estimates
        :param j: The parameter dimension to update, defaults to 0

        :return: The optimal step length for the given parameter estimates and responses
        """
        if self.dist == "normal":
            if self.d == 2:
                if j == 0:
                    return np.mean(y - z[0])
                elif j == 1:
                    return 0.5 * np.log(np.mean((np.exp(-2 * z[1]) * (y - z[0]) ** 2)))
            elif self.d == 1:
                return np.mean(y - z)
        if self.dist == "gamma":
            if self.d == 2:
                if j == 0:
                    return np.log(np.exp(-z[0] - z[1]).sum() / np.exp(-z[1]).sum())
                if j == 1:
                    g_opt = self._opt_step_numeric(y=y, z=z, j=1, g_0=g_0)
                    return g_opt
            elif self.d == 1:
                return np.log((y * np.exp(-z)).mean())
