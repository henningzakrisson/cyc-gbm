import numpy as np
from typing import Union
from src.exceptions import UnknownDistribution


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
                # TODO: Add these
                return None

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
                # TODO: add these
                return None
            elif self.d == 1:
                return 1 - y * np.exp(-z)

    def mle(self, y: np.ndarray) -> Union[float, np.ndarray]:
        """
        Calculates the d-dimensional maximum likelihood estimator of the observations.

        :param y: The target values.
        :return: The maximum likelihood estimator of the parameters.
        """
        if self.dist == "normal":
            if self.d == 2:
                return np.array([y.mean(), y.std()])
            elif self.d == 1:
                return y.mean()
        elif self.dist == "gamma":
            if self.d == 2:
                # TODO: add this
                return None
            elif self.d == 1:
                return np.log(y.mean())

    def opt_step(self, y: np.ndarray, z: np.ndarray, j: int = 0) -> np.ndarray:
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
                # TODO: add these
                return None
            elif self.d == 1:
                return np.log((y * np.exp(-z)).mean())
