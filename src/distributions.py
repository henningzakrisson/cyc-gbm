import numpy as np
from typing import Union
from src.exceptions import UnknownDistribution
from scipy.special import loggamma, polygamma
from scipy.optimize import minimize


class NormalDistribution:
    def __init__(
        self,
    ):
        """
        Initialize a normal distribution object.
        """

    def loss(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates the loss of the parameter estimates and the response.

        :param z: The predicted parameters.
        :param y: The target values.
        :return: The loss function value(s) for the given `z` and `y`.
        """
        return z[1] + 0.5 * np.exp(-2 * z[1]) * (y - z[0]) ** 2

    def grad(self, z: np.ndarray, y: np.ndarray, j: int) -> np.ndarray:
        """
        Calculates the gradients of the loss function with respect to the parameters.

        :param z: The predicted parameters.
        :param y: The target values.
        :param j: The parameter dimension to compute the gradient for (default=0).
        :return: The gradient(s) of the loss function for the given `z`, `y`, and `j`.
        """
        if j == 0:
            return -np.exp(-2 * z[1]) * (y - z[0])
        elif j == 1:
            return 1 - np.exp(-2 * z[1]) * (y - z[0]) ** 2

    def mle(self, y: np.ndarray) -> Union[float, np.ndarray]:
        """
        Calculates the d-dimensional maximum likelihood estimator of the observations.

        :param y: The target values.
        :return: The maximum likelihood estimator of the parameters.
        """
        return np.array([y.mean(), np.log(y.std())])

    def opt_step(self, y: np.ndarray, z: np.ndarray, j: int, g_0=0) -> np.ndarray:
        """
        Calculate the optimal step length for these parameter estimates and responses

        :param y: The target values
        :param z: The current parameter estimates
        :param j: The parameter dimension to update, defaults to 0
        :param g_0: initial guess (not used for this distribution)

        :return: The optimal step length for the given parameter estimates and responses
        """
        if j == 0:
            return np.mean(y - z[0])
        elif j == 1:
            return 0.5 * np.log(np.mean((np.exp(-2 * z[1]) * (y - z[0]) ** 2)))


class GammaDistribution:
    def __init__(
        self,
    ):
        """
        Initialize a gamma distribution object.
        """

    def loss(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates the loss of the parameter estimates and the response.

        :param z: The predicted parameters.
        :param y: The target values.
        :return: The loss function value(s) for the given `z` and `y`.
        """
        return loggamma(np.exp(-z[1])) + np.exp(-z[1]) * (
            y * np.exp(-z[0]) - np.log(y) + z[0] + z[1]
        )

    def grad(self, z: np.ndarray, y: np.ndarray, j: int) -> np.ndarray:
        """
        Calculates the gradients of the loss function with respect to the parameters.

        :param z: The predicted parameters.
        :param y: The target values.
        :param j: The parameter dimension to compute the gradient for (default=0).
        :return: The gradient(s) of the loss function for the given `z`, `y`, and `j`.
        """
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

    def mle(self, y: np.ndarray) -> Union[float, np.ndarray]:
        """
        Calculates the d-dimensional maximum likelihood estimator of the observations.

        :param y: The target values.
        :return: The maximum likelihood estimator of the parameters.
        """
        z_0 = np.log(y.mean())
        z_1_0 = np.log(y.var() / (y.mean() ** 2))
        z_1 = mle_numeric(
            distribution=self, y=y, z=np.array([z_0, 0]), j=1, z_j_0=z_1_0
        )
        return np.array([z_0, z_1])

    def opt_step(self, y: np.ndarray, z: np.ndarray, j: int, g_0=0) -> np.ndarray:
        """
        Calculate the optimal step length for these parameter estimates and responses

        :param y: The target values
        :param z: The current parameter estimates
        :param j: The parameter dimension to update, defaults to 0

        :return: The optimal step length for the given parameter estimates and responses
        """
        if j == 0:
            return np.log((y * np.exp(-z[0] - z[1])).sum() / np.exp(-z[1]).sum())
        if j == 1:
            return opt_step_numeric(distribution=self, y=y, z=z, j=1, g_0=g_0)


class BetaPrimeDistribution:
    def __init__(
        self,
    ):
        """
        Initialize a beta prime distribution object.
        """

    def loss(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates the loss of the parameter estimates and the response.

        :param z: The predicted parameters.
        :param y: The target values.
        :return: The loss function value(s) for the given `z` and `y`.
        """
        mu = np.exp(z[0])
        v = np.exp(z[1])
        loss = (
            (mu + v + mu * v) * np.log(y + 1)
            - mu * (v + 1) * np.log(y)
            + loggamma(mu * (v + 1))
            + loggamma(v + 2)
            - loggamma(mu + v + mu * v + 2)
        )
        return loss

    def grad(self, z: np.ndarray, y: np.ndarray, j: int) -> np.ndarray:
        """
        Calculates the gradients of the loss function with respect to the parameters.

        :param z: The predicted parameters.
        :param y: The target values.
        :param j: The parameter dimension to compute the gradient for (default=0).
        :return: The gradient(s) of the loss function for the given `z`, `y`, and `j`.
        """
        mu = np.exp(z[0])
        v = np.exp(z[1])
        if j == 0:
            return (
                mu
                * (1 + v)
                * (
                    polygamma(0, mu * (1 + v))
                    - polygamma(0, mu + v + mu * v + 2)
                    + np.log((1 + y) / y)
                )
            )
        elif j == 1:
            return v * (
                mu * np.log((y + 1) / y)
                + np.log(y + 1)
                + mu * polygamma(0, mu * (v + 1))
                + polygamma(0, v + 2)
                - (mu + 1) * polygamma(0, mu * (v + 1) + v + 2)
            )

    def mle(self, y: np.ndarray) -> Union[float, np.ndarray]:
        """
        Calculates the d-dimensional maximum likelihood estimator of the observations.

        :param y: The target values.
        :return: The maximum likelihood estimator of the parameters.
        """
        z_0_0 = np.log(y.mean())
        z_1_0 = np.log(y.mean() * (1 + y.mean()) / y.var())
        z_1 = mle_numeric(
            distribution=self, y=y, z=np.array([z_0, 0]), j=1, z_j_0=z_1_0
        )
        return np.array([z_0, z_1])

    def opt_step(self, y: np.ndarray, z: np.ndarray, j: int, g_0=0) -> np.ndarray:
        """
        Calculate the optimal step length for these parameter estimates and responses

        :param y: The target values
        :param z: The current parameter estimates
        :param j: The parameter dimension to update, defaults to 0
        :param g_0: initial guess

        :return: The optimal step length for the given parameter estimates and responses
        """
        return opt_step_numeric(distribution=self, y=y, z=z, j=j, g_0=g_0)


def mle_numeric(
    distribution: Union[NormalDistribution, GammaDistribution, BetaPrimeDistribution],
    y: np.ndarray,
    z: np.ndarray,
    j: int,
    z_j_0: np.ndarray = np.array([0, 0]),
):
    """Compute maximum likelihood estimator numerically

    :param distribution: the distribution to minimize the loss over
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

    to_min = lambda z_j: distribution.loss(z + e[j] * z_j, y).sum()
    z_j_opt = minimize(to_min, z_j_0)["x"][0]

    return z_j_opt


def step_loss(
    distribution: Union[NormalDistribution, GammaDistribution, BetaPrimeDistribution],
    z: np.ndarray,
    y: np.ndarray,
    step: float,
    j: int,
) -> float:
    """
    Loss function evaluated when adding step gamma to dimension j of z.

    :param distribution: the distribution to calculate the loss over
    :param z: An array of shape (n_samples, n_features) representing the input features.
    :param y: An array of shape (n_samples,) representing the target values.
    :param step: A float representing the amount to add to the jth dimension of z.
    :param j: An integer representing the dimension of z to add gamma to.
    :return: A float representing the sum of the loss values across all samples.
    """
    z[j] += step
    return distribution.loss(z, y).sum()


def opt_step_numeric(
    distribution: Union[NormalDistribution, GammaDistribution, BetaPrimeDistribution],
    y: np.ndarray,
    z: np.ndarray,
    j: int,
    g_0: float = 0,
):
    """
    Numerically optimize the step size for the data in specified dimension

    :param distribution: the distribution to calculate the loss over
    :param y: Target values.
    :param z: Current parameter estimates.
    :param j: Index of the dimension to optimize.
    :param g_0: Initial guess for the optimal step size. Default is 0.
    :return: The optimal step size.
    """

    to_min = lambda step: step_loss(distribution=distribution, z=z, y=y, step=step, j=j)
    step_opt = minimize(to_min, g_0)["x"][0]
    return step_opt


def initiate_dist(
    dist: str,
) -> Union[NormalDistribution, GammaDistribution, BetaPrimeDistribution]:
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
    raise UnknownDistribution("Unknown distribution")
