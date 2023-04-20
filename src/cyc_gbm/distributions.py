import numpy as np
from typing import Union, Type
from src.cyc_gbm.exceptions import UnknownDistributionError
from scipy.special import loggamma, polygamma
from scipy.optimize import minimize


def inherit_docstrings(cls: Type) -> Type:
    """
    Decorator to copy docstrings from base class to derived class methods.

    :param cls: The class to decorate.
    :return: The decorated class.
    """
    for name, method in vars(cls).items():
        if method.__doc__ is None:
            for parent in cls.__bases__:
                parent_method = getattr(parent, name)
                if parent_method.__doc__ is not None:
                    method.__doc__ = parent_method.__doc__
                break
    return cls


class Distribution:
    def __init__(
        self,
    ):
        """Initialize a distribution object."""

    def loss(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Calculates the loss of the parameter estimates and the response.

        :param z: The predicted parameters.
        :param y: The target values.
        :return: The loss function value(s) for the given `z` and `y`.
        """
        pass

    def grad(self, y: np.ndarray, z: np.ndarray, j: int) -> np.ndarray:
        """
        Calculates the gradients of the loss function with respect to the parameters.

        :param z: The predicted parameters.
        :param y: The target values.
        :param j: The parameter dimension to compute the gradient for (default=0).
        :return: The gradient(s) of the loss function for the given `z`, `y`, and `j`.
        """
        return self.grad_functions[j](y=y, z=z)

    def mme(self, y: np.ndarray) -> np.ndarray:
        """
        Calculates the method of moments estimator for the parameter vector

        :param y: The target values.
        :return: A moment-based initial guess for the MLE parameter
        """
        pass

    def mle(self, y: np.ndarray) -> np.ndarray:
        """
        Calculates the maximum likelihood estimator of the parameter given observations.

        :param y: The target values.
        :return: The maximum likelihood estimator of the parameters.
        """
        z_0 = self.mme(y=y)
        to_min = lambda z: self.loss(y=y, z=z).sum()
        z_opt = minimize(to_min, z_0)["x"]
        return z_opt

    def opt_step(
        self,
        y: np.ndarray,
        z: np.ndarray,
        j: int,
        g_0: float = 0,
    ):
        """
        Numerically optimize the step size for the data in specified dimension

        :param y: Target values.
        :param z: Current parameter estimates.
        :param j: Index of the dimension to optimize.
        :param g_0: Initial guess for the optimal step size. Default is 0.
        :return: The optimal step size.
        """

        # Indicator vector for adding step to dimension j (assume two dimensions)
        e = np.eye(2)[:, j : j + 1]
        to_min = lambda step: self.loss(y=y, z=z + e * step).sum()
        step_opt = minimize(fun=to_min, x0=g_0)["x"][0]
        return step_opt


@inherit_docstrings
class NormalDistribution(Distribution):
    def __init__(
        self,
    ):
        """Initialize a normal distribution object."""

    def loss(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return z[1] + 0.5 * np.exp(-2 * z[1]) * (y - z[0]) ** 2

    # loss.__doc__ = super().loss.__doc__

    def grad(self, y: np.ndarray, z: np.ndarray, j: int) -> np.ndarray:
        if j == 0:
            return -np.exp(-2 * z[1]) * (y - z[0])
        elif j == 1:
            return 1 - np.exp(-2 * z[1]) * (y - z[0]) ** 2

    # grad.__doc__ = Distribution.grad.__doc__

    def mme(self, y: np.ndarray) -> np.ndarray:
        return np.array([y.mean(), np.log(y.std())])

    # initial_mle.__doc__ = Distribution.initial_mle.__doc__


@inherit_docstrings
class InverseGaussianDistribution(Distribution):
    def __init__(
        self,
    ):
        """Initialize a normal distribution object."""

    def loss(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (
            np.exp(z[1]) * (y * np.exp(-2 * z[0]) - 2 * np.exp(-z[0]) + y ** (-1))
            - z[1]
        )

    def grad(self, y: np.ndarray, z: np.ndarray, j: int) -> np.ndarray:
        if j == 0:
            return 2 * np.exp(z[1] - z[0]) * (1 - y * np.exp(-z[0]))
        elif j == 1:
            return (
                np.exp(z[1]) * (y * np.exp(-2 * z[0]) - 2 * np.exp(-z[0]) + y ** (-1))
                - 1
            )

    def mme(self, y: np.ndarray) -> np.ndarray:
        return np.array([np.log(np.mean(y)), 3 * np.log(np.mean(y)) - np.log(y.var())])


@inherit_docstrings
class GammaDistribution(Distribution):
    def __init__(
        self,
    ):
        """
        Initialize a gamma distribution object.
        """

    def loss(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return loggamma(np.exp(-z[1])) + np.exp(-z[1]) * (
            y * np.exp(-z[0]) - np.log(y) + z[0] + z[1]
        )

    def grad(self, y: np.ndarray, z: np.ndarray, j: int) -> np.ndarray:
        if j == 0:
            return np.exp(-z[1]) * (1 - y * np.exp(-z[0]))
        elif j == 1:
            return np.exp(-z[1]) * (
                1
                + np.log(y)
                - z[0]
                - z[1]
                - y * np.exp(-z[0])
                - polygamma(0, np.exp(-z[1]))
            )

    def mme(self, y: np.ndarray) -> np.ndarray:
        return np.array(
            [
                np.log(np.mean(y)),
                np.log(np.var(y) / (np.mean(y) ** 2)),
            ]
        )


@inherit_docstrings
class BetaPrimeDistribution(Distribution):
    def __init__(
        self,
    ):
        """
        Initialize a beta prime distribution object.
        """

    def loss(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (
            (np.exp(z[0]) + np.exp(z[1]) + np.exp(z[0] + z[1])) * np.log(y + 1)
            - np.exp(z[0]) * (np.exp(z[1]) + 1) * np.log(y)
            + loggamma(np.exp(z[0]) * (np.exp(z[1]) + 1))
            + loggamma(np.exp(z[1]) + 2)
            - loggamma(np.exp(z[0]) + np.exp(z[1]) + np.exp(z[0] + z[1]) + 2)
        )

    def grad(self, y: np.ndarray, z: np.ndarray, j: int) -> np.ndarray:
        if j == 0:
            return (
                np.exp(z[0])
                * (1 + np.exp(z[1]))
                * (
                    polygamma(0, np.exp(z[0]) * (1 + np.exp(z[1])))
                    - polygamma(
                        0, np.exp(z[0]) + np.exp(z[1]) + np.exp(z[0] + z[1]) + 2
                    )
                    + np.log((1 + y) / y)
                )
            )
        elif j == 1:
            return np.exp(z[1]) * (
                np.exp(z[0]) * np.log((y + 1) / y)
                + np.log(y + 1)
                + np.exp(z[0]) * polygamma(0, np.exp(z[0]) * (np.exp(z[1]) + 1))
                + polygamma(0, np.exp(z[1]) + 2)
                - (np.exp(z[0]) + 1)
                * polygamma(0, np.exp(z[0]) * (np.exp(z[1]) + 1) + np.exp(z[1]) + 2)
            )

    def mme(self, y: np.ndarray) -> np.ndarray:
        return np.array(
            [
                np.log(np.mean(y)),
                np.log(y.mean() * (1 + y.mean()) / y.var()),
            ]
        )


def initiate_dist(
    dist: str,
) -> Distribution:
    """
    Returns a probability distribution object based on the distribution name.

    :param dist: A string representing the name of the distribution to create.
                 Valid values are "normal", "gamma", or "beta_prime".
    :return: A probability distribution object based on the distribution name.
    :raises UnknownDistribution: If the input distribution name is not recognized.
    """
    if dist == "normal":
        return NormalDistribution()
    if dist == "gamma":
        return GammaDistribution()
    if dist == "beta_prime":
        return BetaPrimeDistribution()
    if dist == "inv_gauss":
        return InverseGaussianDistribution()
    raise UnknownDistributionError("Unknown distribution")


if __name__ == "__main__":
    normal_dist = initiate_dist(dist="normal")
    print(normal_dist.loss.__doc__)
