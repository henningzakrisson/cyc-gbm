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

    def loss(
        self, y: np.ndarray, z: np.ndarray, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        """
        Calculates the loss of the parameter estimates and the response.

        :param z: The predicted parameters.
        :param y: The target values.
        :param w: The weights of the observations. Default is 1.0.
        :return: The loss function value(s) for the given `z` and `y`.
        """
        pass

    def grad(
        self, y: np.ndarray, z: np.ndarray, j: int, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        """
        Calculates the gradients of the loss function with respect to the parameters.

        :param z: The predicted parameters.
        :param y: The target values.
        :param j: The parameter dimension to compute the gradient for (default=0).
        :param w: The weights of the observations. Default is 1.0.
        :return: The gradient(s) of the loss function for the given `z`, `y`, and `j`.
        """
        pass

    def mme(self, y: np.ndarray) -> np.ndarray:
        """
        Calculates the method of moments estimator for the parameter vector

        :param y: The target values.
        :return: the method of moments estimator
        """
        pass

    def simulate(
        self,
        z: np.ndarray,
        random_state: Union[int, None] = None,
        rng: Union[np.random.Generator, None] = None,
    ) -> np.ndarray:
        """Simulate values given parameter values in z

        :param z: Parameter values of shape (n_parameters, n_samples).
        :param random_state: Random seed to use in simulation.
        :param rng: Random number generator to use in simulation.
        :return: Simulated values of shape (n_samples,).
        """
        pass

    def moment(self, z: np.ndarray, k: int) -> np.ndarray:
        """
        Calculates the k:th moment given parameter estimates z.

        :param z: The predicted parameters of shape (d,n_samples).
        :param k: The number of the moment
        :return: Array with the k:th moments of shape (n_samples,)
        """
        pass

    def mle(self, y: np.ndarray, w: Union[np.ndarray, float] = 1.0) -> np.ndarray:
        """
        Calculates the maximum likelihood estimator of the parameter given observations.

        :param y: The target values.
        :param w: The weights of the observations. Default is 1.0.
        :return: The maximum likelihood estimator of the parameters.
        """
        z_0 = self.mme(y=y)
        to_min = lambda z: self.loss(y=y, z=z, w=w).sum()
        z_opt = minimize(to_min, z_0)["x"]
        return z_opt

    def opt_step(
        self,
        y: np.ndarray,
        z: np.ndarray,
        j: int,
        w: Union[np.ndarray, float] = 1.0,
        g_0: float = 0,
    ):
        """
        Numerically optimize the step size for the data in specified dimension

        :param y: Target values.
        :param z: Current parameter estimates.
        :param j: Index of the dimension to optimize.
        :param w: Weights of the observations. Default is 1.0.
        :param g_0: Initial guess for the optimal step size. Default is 0.
        :return: The optimal step size.
        """

        # Indicator vector for adding step to dimension j
        e = np.eye(self.d)[:, j : j + 1]
        to_min = lambda step: self.loss(y=y, z=z + e * step, w=w).sum()
        grad = lambda step: self.grad(y=y, z=z + e * step, j=j, w=w).sum()
        step_opt = minimize(
            fun=to_min,
            jac=grad,
            x0=g_0,
        )[
            "x"
        ][0]
        return step_opt


@inherit_docstrings
class MultivariateNormalDistribution(Distribution):
    def __init__(
        self,
    ):
        """Initialize a multivariate normal distribution object with equal means and variances and
        correlation

        Parameterization: z[0] = mu, z[1] = 2*log(sigma), z[2] = inv_sigm(rho)
        where
        E[X] = [w*mu,w*mu]
        Cov(X) = w*[sigma^2, rho*sigma^2; rho*sigma^2, sigma^2]
        """
        self.d = 3

    def sigm(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid function

        :param x: Input array.
        :return: Sigmoid of input array.
        """
        return 1 / (1 + np.exp(-x))

    def sigm_inv(self, x: np.ndarray) -> np.ndarray:
        """Inverse sigmoid function

        :param x: Input array.
        :return: Inverse sigmoid of input array.
        """
        return np.log(x / (1 - x))

    def loss(
        self, y: np.ndarray, z: np.ndarray, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        mu = w * z[0]
        s2 = w * np.exp(z[1])
        rho = self.sigm(z[2])

        mu_term = (
            (y[:, 0] - mu) ** 2
            - 2 * rho * (y[:, 0] - mu) * (y[:, 1] - mu)
            + (y[:, 1] - mu) ** 2
        )
        rho_term = 1 / (1 - rho**2)
        return z[1] + 0.5 * mu_term * rho_term / s2

    def grad(
        self, y: np.ndarray, z: np.ndarray, j: int, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        if j == 0:
            mu = w * z[0]
            s2 = w * np.exp(z[1])
            rho = self.sigm(z[2])

            return (1 / (s2 * (1 + rho))) * (2 * mu - y[:, 0] - y[:, 1])
        elif j == 1:
            mu = w * z[0]
            s2 = w * np.exp(z[1])
            rho = self.sigm(z[2])

            mu_term = (
                (y[:, 0] - mu) ** 2
                - 2 * rho * (y[:, 0] - mu) * (y[:, 1] - mu)
                + (y[:, 1] - mu) ** 2
            )
            a = mu_term / (2 * (1 - rho**2))
            grad = 1 - a / s2
            if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
                raise Exception("NaN in gradient")
            return grad

        elif j == 2:
            mu = w * z[0]
            s2 = w * np.exp(z[1])
            rho = self.sigm(z[2])

            m_1 = (y[:, 0] - mu) ** 2 + (y[:, 1] - mu) ** 2
            m_2 = (y[:, 0] - mu) * (y[:, 1] - mu)
            return 1 * (
                (rho / ((1 + rho)))
                * (
                    (1 / (s2 * (1 - rho**2))) * (-m_2 * rho**2 + m_1 * rho - m_2)
                    - rho
                )
            )

    def mme(self, y: np.ndarray, w: Union[np.ndarray, float] = 1.0) -> np.ndarray:
        # Return the mean, variance and correlation of a 2d normal distribution
        if isinstance(w, float):
            w = np.array([w] * len(y))
        mu = y.sum() / w.sum()
        s2 = sum((y[0] - mu) ** 2 + (y[1] - mu) ** 2) / 2 * w.sum()
        rho = np.corrcoef(y.T)[0, 1]
        return np.array([mu, np.log(s2), self.sigm_inv(rho)])

    def simulate(
        self,
        z: np.ndarray,
        w: Union[np.ndarray, float] = 1.0,
        random_state: Union[int, None] = None,
        rng: Union[np.random.Generator, None] = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(seed=random_state)
        mu = w * np.stack([z[0]] * 2)
        s2 = w * np.exp(z[1])
        rho = self.sigm(z[2])
        Sigma = np.stack([np.stack([s2, s2 * rho]), np.stack([s2 * rho, s2])])

        return np.stack(
            [
                rng.multivariate_normal(mu[:, i], Sigma[:, :, i])
                for i in range(0, z.shape[1])
            ]
        )

    def moment(
        self, z: np.ndarray, k: int, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        if k == 1:
            return w * np.array([z[0], z[0]])
        elif k == 2:
            rho = self.sigm(z[2])
            s2 = w * np.exp(z[1])
            return np.stack([np.stack([s2, s2 * rho]), np.stack([s2 * rho, s2])])


@inherit_docstrings
class NormalDistribution(Distribution):
    def __init__(
        self,
    ):
        """Initialize a normal distribution object.

        Parameterization: z[0] = mu, z[1] = log(sigma), where E[X] = w*mu, Var(X) = w*sigma^2
        """
        self.d = 2

    def loss(
        self, y: np.ndarray, z: np.ndarray, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        return (
            0.5 * np.log(w)
            + z[1]
            + 1 / (2 * w) * np.exp(-2 * z[1]) * (y - w * z[0]) ** 2
        )

    def grad(
        self, y: np.ndarray, z: np.ndarray, j: int, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        if j == 0:
            return -np.exp(-2 * z[1]) * (y - w * z[0])
        elif j == 1:
            return 1 - np.exp(-2 * z[1]) * (y - w * z[0]) ** 2 / w

    def mme(self, y: np.ndarray, w: Union[np.ndarray, float] = 1.0) -> np.ndarray:
        if not isinstance(w, np.ndarray):
            w = np.array([w] * len(y))
        mean = y.sum() / w.sum()
        log_sigma = 0.5 * np.log(((y - w * mean) ** 2 / w).mean())
        return np.array([mean, log_sigma])

    def simulate(
        self,
        z: np.ndarray,
        w: Union[np.ndarray, float] = 1.0,
        random_state: Union[int, None] = None,
        rng: Union[np.random.Generator, None] = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(seed=random_state)
        return rng.normal(w * z[0], w**0.5 * np.exp(z[1]))

    def moment(
        self, z: np.ndarray, k: int, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        if k == 1:
            return w * z[0]
        elif k == 2:
            return w * np.exp(2 * z[1])


@inherit_docstrings
class NegativeBinomialDistribution(Distribution):
    def __init__(
        self,
    ):
        """Initialize a negative binomial distribution object.

        Parameterization: z[0] = mu, z[1] = log(theta), where E[X] = w*mu, Var(X) = w*mu*(1+mu/theta)
        """
        self.d = 2

    def loss(
        self, y: np.ndarray, z: np.ndarray, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        return (
            loggamma(w * np.exp(z[1]))
            - loggamma(w * np.exp(z[1]) + y)
            - y * z[0]
            - w * z[1] * np.exp(z[1])
            + (w * np.exp(z[1]) + y) * np.log(np.exp(z[0]) + np.exp(z[1]))
        )

    def grad(
        self, y: np.ndarray, z: np.ndarray, j: int, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        if j == 0:
            return (w * np.exp(z[1]) + y) * np.exp(z[0]) / (
                np.exp(z[0]) + np.exp(z[1])
            ) - y
        elif j == 1:
            return (
                w
                * np.exp(z[1])
                * (
                    polygamma(0, w * np.exp(z[1]))
                    - polygamma(0, w * np.exp(z[1]) + y)
                    - 1
                    - z[1]
                    + np.log(np.exp(z[0]) + np.exp(z[1]))
                    + (np.exp(z[1]) + y / w) / (np.exp(z[0]) + np.exp(z[1]))
                )
            )

    def mme(self, y: np.ndarray, w: Union[np.ndarray, float] = 1.0) -> np.ndarray:
        if isinstance(w, float):
            w = np.array([w] * len(y))
        mean = y.sum() / w.sum()
        var = sum((y - mean) ** 2) / w.sum()
        theta = mean**2 / (var - mean)
        return np.array([np.log(mean), np.log(theta)])

    def simulate(
        self,
        z: np.ndarray,
        w: Union[np.ndarray, float] = 1.0,
        random_state: Union[int, None] = None,
        rng: Union[np.random.Generator, None] = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(seed=random_state)
        mu = np.exp(z[0])
        theta = np.exp(z[1])
        p = theta / (mu + theta)
        r = w * theta
        y = rng.negative_binomial(r, p)
        return y.astype(float)

    def moment(
        self, z: np.ndarray, k: int, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        if k == 1:
            return w * np.exp(z[0])
        elif k == 2:
            return w * np.exp(z[0]) * (1 + np.exp(z[0] - z[1]))


@inherit_docstrings
class InverseGaussianDistribution(Distribution):
    def __init__(
        self,
    ):
        """Initialize a inverse Gaussian distribution object.

        Parameterization: z[0] = log(mu), z[1] = log(lambda), where E[X] = w*mu, Var(X) =w*mu^3 / lambda
        """
        self.d = 2

    def loss(
        self, y: np.ndarray, z: np.ndarray, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        return (
            np.exp(z[1])
            * (y * np.exp(-2 * z[0]) - 2 * w * np.exp(-z[0]) + w**2 * y ** (-1))
            - z[1]
            - 2 * np.log(w)
        )

    def grad(
        self, y: np.ndarray, z: np.ndarray, j: int, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        if j == 0:
            return 2 * w * np.exp(z[1] - z[0]) * (1 - y * np.exp(-z[0]) / w)
        elif j == 1:
            return (
                np.exp(z[1])
                * (y * np.exp(-2 * z[0]) - 2 * w * np.exp(-z[0]) + w**2 * y ** (-1))
                - 1
            )

    def mme(self, y: np.ndarray, w: Union[np.ndarray, float] = 1.0) -> np.ndarray:
        if isinstance(w, float):
            w = np.array([w] * len(y))
        mean = y.sum() / w.sum()
        var = sum((y - mean) ** 2) / w.sum()
        z0 = np.log(mean)
        z1 = 3 * np.log(mean) - np.log(var)
        return np.array([z0, z1])

    def simulate(
        self,
        z: np.ndarray,
        w: Union[np.ndarray, float] = 1.0,
        random_state: Union[int, None] = None,
        rng: Union[np.random.Generator, None] = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(seed=random_state)
        return rng.wald(w * np.exp(z[0]), w**2 * np.exp(z[1]))

    def moment(
        self, z: np.ndarray, k: int, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        if k == 1:
            return w * np.exp(z[0])
        elif k == 2:
            return w * np.exp(3 * z[0] - z[1])


@inherit_docstrings
class GammaDistribution(Distribution):
    def __init__(
        self,
    ):
        """
        Initialize a gamma distribution object.

        Parameterization: z[0] = log(mu), z[1] = log(phi), where E[X] = w*mu, Var(X) =w*phi * mu^2
        """
        self.d = 2

    def loss(
        self, y: np.ndarray, z: np.ndarray, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        return (
            loggamma(w * np.exp(-z[1]))
            + y * np.exp(-z[0] - z[1])
            + w * np.exp(-z[1]) * (z[0] + z[1] - np.log(y))
        )

    def grad(
        self, y: np.ndarray, z: np.ndarray, j: int, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        if j == 0:
            return np.exp(-z[1]) * (w - y * np.exp(-z[0]))
        elif j == 1:
            return (
                w
                * np.exp(-z[1])
                * (
                    -polygamma(0, w * np.exp(-z[1]))
                    - y * np.exp(-z[0]) / w
                    - z[0]
                    - z[1]
                    + np.log(y)
                    + 1
                )
            )

    def mme(self, y: np.ndarray, w: Union[np.ndarray, float] = 1.0) -> np.ndarray:
        if isinstance(w, float):
            w = np.array([w] * len(y))
        mean = y.sum() / w.sum()
        var = sum((y - mean) ** 2) / w.sum()
        z0 = np.log(mean)
        z1 = np.log(var) - 2 * np.log(mean)

        return np.array([z0, z1])

    def simulate(
        self,
        z: np.ndarray,
        w: Union[np.ndarray, float] = 1.0,
        random_state: Union[int, None] = None,
        rng: Union[np.random.Generator, None] = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(seed=random_state)
        scale = np.exp(z[0] + z[1])
        shape = w * np.exp(-z[1])
        return rng.gamma(shape=shape, scale=scale)

    def moment(
        self, z: np.ndarray, k: int, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        if k == 1:
            return w * np.exp(z[0])
        elif k == 2:
            return w * np.exp(2 * z[0] + z[1])


@inherit_docstrings
class BetaPrimeDistribution(Distribution):
    def __init__(
        self,
    ):
        """
        Initialize a beta prime distribution object.

        Parameterization: z[0] = log(mu), z[1] = log(v), where E[X] = w*mu, Var(X) =w*mu*(1+mu)/v
        """
        self.d = 2

    def loss(
        self, y: np.ndarray, z: np.ndarray, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        return (
            (w * np.exp(z[0]) + w * np.exp(z[1]) + w**2 * np.exp(z[0] + z[1]))
            * np.log(y + 1)
            - w * np.exp(z[0]) * (w * np.exp(z[1]) + 1) * np.log(y)
            + loggamma(w * np.exp(z[0]) * (w * np.exp(z[1]) + 1))
            + loggamma(w * np.exp(z[1]) + 2)
            - loggamma(
                w * np.exp(z[0]) + w * np.exp(z[1]) + w**2 * np.exp(z[0] + z[1]) + 2
            )
        )

    def grad(
        self, y: np.ndarray, z: np.ndarray, j: int, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        if j == 0:
            return (
                w
                * np.exp(z[0])
                * (1 + w * np.exp(z[1]))
                * (
                    polygamma(0, w * np.exp(z[0]) * (1 + w * np.exp(z[1])))
                    - polygamma(
                        0,
                        w * np.exp(z[0])
                        + w * np.exp(z[1])
                        + w**2 * np.exp(z[0] + z[1])
                        + 2,
                    )
                    + np.log((1 + y) / y)
                )
            )
        elif j == 1:
            return (
                w
                * np.exp(z[1])
                * (
                    w * np.exp(z[0]) * np.log((y + 1) / y)
                    + np.log(y + 1)
                    + w
                    * np.exp(z[0])
                    * polygamma(0, w * np.exp(z[0]) * (w * np.exp(z[1]) + 1))
                    + polygamma(0, w * np.exp(z[1]) + 2)
                    - (w * np.exp(z[0]) + 1)
                    * polygamma(
                        0,
                        w * np.exp(z[0]) * (w * np.exp(z[1]) + 1)
                        + w * np.exp(z[1])
                        + 2,
                    )
                )
            )

    def mme(self, y: np.ndarray, w: Union[np.ndarray, float] = 1.0) -> np.ndarray:
        if isinstance(w, float):
            w = np.array([w] * len(y))
        mean = y.sum() / w.sum()
        var = sum((y - mean) ** 2) / w.sum()
        z0 = np.log(mean)
        z1 = np.log(mean) + np.log(1 + mean) - np.log(var)

        return np.array([z0, z1])

    def simulate(
        self,
        z: np.ndarray,
        w: Union[np.ndarray, float] = 1.0,
        random_state: Union[int, None] = None,
        rng: Union[np.random.Generator, None] = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(seed=random_state)
        mu = np.exp(z[0])
        v = np.exp(z[1])
        alpha = w * mu * (1 + w * v)
        beta = w * v + 2
        y0 = rng.beta(alpha, beta)
        return y0 / (1 - y0)

    def moment(
        self, z: np.ndarray, k: int, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        if k == 1:
            return w * np.exp(z[0])
        elif k == 2:
            return w * np.exp(z[0] - z[1]) * (1 + np.exp(z[0]))


def initiate_distribution(
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
    if dist == "neg_bin":
        return NegativeBinomialDistribution()
    if dist == "multivariate_normal":
        return MultivariateNormalDistribution()
    raise UnknownDistributionError("Unknown distribution")


if __name__ == "__main__":
    normal_dist = initiate_distribution(dist="normal")
    print(normal_dist.loss.__doc__)
