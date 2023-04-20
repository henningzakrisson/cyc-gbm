import numpy as np
from typing import Union
from src.cyc_gbm.exceptions import UnknownDistributionError
from scipy.special import loggamma, polygamma
from scipy.optimize import minimize


class Distribution:
    def __init__(self):
        """Initialize a distribution object.
        """

    def calculate_loss(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Calculates the loss of the parameter estimates and the response.

        :param z: The predicted parameters.
        :param y: The target values.
        :return: The loss function value(s) for the given `z` and `y`.
        """
        return self.loss_function(y=y, z=z)

    def calculate_grad(self, y: np.ndarray, z: np.ndarray, j: int) -> np.ndarray:
        """
        Calculates the gradients of the loss function with respect to the parameters.

        :param z: The predicted parameters.
        :param y: The target values.
        :param j: The parameter dimension to compute the gradient for (default=0).
        :return: The gradient(s) of the loss function for the given `z`, `y`, and `j`.
        """
        return self.grad_functions[j](y=y, z=z)

    def estimate_mle(self, y: np.ndarray) -> Union[float, np.ndarray]:
        """
        Calculates the maximum likelihood estimator of the parameter given observations.

        :param y: The target values.
        :return: The maximum likelihood estimator of the parameters.
        """
        return self.mle_estimator(y=y)

    def calculate_mle_numerically(
        self,
        y: np.ndarray,
        z_0: np.ndarray = np.array([0, 0]),
    ):
        """Compute maximum likelihood estimator numerically

        :param y: Target values.
        :param z_0: Initial guess for the MLE
        :return: The MLE of the parameter vector

        """
        to_min = lambda z: self.calculate_loss(y=y, z=z).sum()
        z_opt = minimize(to_min, z_0)["x"]
        return z_opt

    def calculate_step(self, y: np.ndarray, z: np.ndarray, j: int, g_0=0) -> np.ndarray:
        """
        Calculate the optimal step length for these parameter estimates and responses

        :param y: The target values
        :param z: The current parameter estimates
        :param j: The parameter dimension to update, defaults to 0
        :param g_0: initial guess (not used for this distribution)

        :return: The optimal step length for the given parameter estimates and responses
        """
        return self.step_functions[j](y=y, z=z, g_0=g_0)

    def calculate_step_numerically(
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
        e = np.eye(2)[:,j:j+1]
        to_min = lambda step: self.calculate_loss(y=y, z=z + e * step).sum()
        step_opt = minimize(fun=to_min, x0=g_0)["x"][0]
        return step_opt


class NormalDistribution(Distribution):
    def __init__(
        self,
    ):
        """Initialize a normal distribution object."""
        self.loss_function = (
            lambda y, z: z[1] + 0.5 * np.exp(-2 * z[1]) * (y - z[0]) ** 2
        )
        self.grad_functions = (
            lambda y, z: -np.exp(-2 * z[1]) * (y - z[0]),
            lambda y, z: 1 - np.exp(-2 * z[1]) * (y - z[0]) ** 2,
        )
        self.mle_estimator = lambda y: np.array([y.mean(), np.log(y.std())])
        self.step_functions = (
            lambda y, z, g_0: np.mean(y - z[0]),
            lambda y, z, g_0: 0.5
            * np.log(np.mean((np.exp(-2 * z[1]) * (y - z[0]) ** 2))),
        )


class GammaDistribution(Distribution):
    def __init__(
        self,
    ):
        """
        Initialize a gamma distribution object.
        """
        self.loss_function = lambda y, z: loggamma(np.exp(-z[1])) + np.exp(-z[1]) * (
            y * np.exp(-z[0]) - np.log(y) + z[0] + z[1]
        )
        self.grad_functions = (
            lambda y, z: np.exp(-z[1]) * (1 - y * np.exp(-z[0])),
            lambda y, z: np.exp(-z[1])
            * (
                1
                + np.log(y)
                - z[0]
                - z[1]
                - y * np.exp(-z[0])
                - polygamma(0, np.exp(-z[1]))
            ),
        )
        self.mle_estimator = lambda y: self.calculate_mle_numerically(
            y=y,
            z_0=[np.log(np.mean(y)), np.log(np.var(y) / (np.mean(y) ** 2))],
        )
        self.step_functions = (
            lambda y, z, g_0: np.log(
                (y * np.exp(-z[0] - z[1])).sum() / np.exp(-z[1]).sum()
            ),
            lambda y, z, g_0: self.calculate_step_numerically(y=y, z=z, j=1, g_0=g_0),
        )


class BetaPrimeDistribution(Distribution):
    def __init__(
        self,
    ):
        """
        Initialize a beta prime distribution object.
        """
        self.loss_function = lambda y, z: (
            (np.exp(z[0]) + np.exp(z[1]) + np.exp(z[0] + z[1])) * np.log(y + 1)
            - np.exp(z[0]) * (np.exp(z[1]) + 1) * np.log(y)
            + loggamma(np.exp(z[0]) * (np.exp(z[1]) + 1))
            + loggamma(np.exp(z[1]) + 2)
            - loggamma(np.exp(z[0]) + np.exp(z[1]) + np.exp(z[0] + z[1]) + 2)
        )
        self.grad_functions = (
            lambda y, z: (
                np.exp(z[0])
                * (1 + np.exp(z[1]))
                * (
                    polygamma(0, np.exp(z[0]) * (1 + np.exp(z[1])))
                    - polygamma(
                        0, np.exp(z[0]) + np.exp(z[1]) + np.exp(z[0] + z[1]) + 2
                    )
                    + np.log((1 + y) / y)
                )
            ),
            lambda y, z: np.exp(z[1])
            * (
                np.exp(z[0]) * np.log((y + 1) / y)
                + np.log(y + 1)
                + np.exp(z[0]) * polygamma(0, np.exp(z[0]) * (np.exp(z[1]) + 1))
                + polygamma(0, np.exp(z[1]) + 2)
                - (np.exp(z[0]) + 1)
                * polygamma(0, np.exp(z[0]) * (np.exp(z[1]) + 1) + np.exp(z[1]) + 2)
            ),
        )
        self.mle_estimator = lambda y: self.calculate_mle_numerically(
            y=y,
            z_0=[np.log(np.mean(y)), np.log(y.mean() * (1 + y.mean()) / y.var())],
        )
        self.step_functions = (
            lambda y, z, g_0: self.calculate_step_numerically(y=y, z=z, j=0, g_0=g_0),
            lambda y, z, g_0: self.calculate_step_numerically(y=y, z=z, j=1, g_0=g_0),
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
    raise UnknownDistributionError("Unknown distribution")
