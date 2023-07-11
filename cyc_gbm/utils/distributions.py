from typing import Callable, Type, Union, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import loggamma, polygamma


def sigm(x: np.ndarray) -> np.ndarray:
    """Sigmoid function

    :param x: Input array.
    :return: Sigmoid of input array.
    """
    return 1 / (1 + np.exp(-x))


def sigm_inv(x: np.ndarray) -> np.ndarray:
    """Inverse sigmoid function

    :param x: Input array.
    :return: Inverse sigmoid of input array.
    """
    return np.log(x / (1 - x))


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
        n_dim: Optional[int] = None,
    ):
        """Initialize a distribution object."""
        self.n_dim = n_dim

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
        :param j: The parameter dimension to compute the gradient for.
        :param w: The weights of the observations. Default is 1.0.
        :return: The gradient(s) of the loss function for the given `z`, `y`, and `j`.
        """
        pass

    def mme(self, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Calculates the method of moments estimator for the parameter vector

        :param y: The target values.
        :param w: The weights of the observations.
        :return: the method of moments estimator
        """
        pass

    def simulate(
        self,
        z: np.ndarray,
        w: Union[np.ndarray, float] = 1.0,
        random_state: Union[int, None] = None,
        rng: Union[np.random.Generator, None] = None,
    ) -> np.ndarray:
        """Simulate values given parameter values in z

        :param z: Parameter values of shape (n_parameters, n_samples).
        :param w: Weights of the observations. Default is 1.0.
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


@inherit_docstrings
class MultivariateNormalDistribution(Distribution):
    def __init__(
        self,
    ):
        """Initialize a multivariate normal distribution object with equal means and variances and
        correlation

        Parameterization: z[0] = mu, z[1] = 2*log(sigma), z[2] = inv_sigm(rho)
        where
        E[Y] = [w*mu,w*mu]
        Cov(Y) = w*[sigma^2, rho*sigma^2; rho*sigma^2, sigma^2]
        """
        super().__init__(n_dim=3)

    def loss(
        self, y: np.ndarray, z: np.ndarray, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        if np.any(w != 1):
            raise NotImplementedError(
                "Weighted loss not implemented for multivariate normal distribution"
            )
        mu = z[0]
        s2 = np.exp(z[1])
        rho = sigm(z[2])

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
        if np.any(w != 1):
            raise NotImplementedError(
                "Weighted loss not implemented for multivariate normal distribution"
            )
        if j == 0:
            mu = z[0]
            s2 = np.exp(z[1])
            rho = sigm(z[2])

            return (1 / (s2 * (1 + rho))) * (2 * mu - y[:, 0] - y[:, 1])
        elif j == 1:
            mu = z[0]
            s2 = np.exp(z[1])
            rho = sigm(z[2])

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
            mu = z[0]
            s2 = np.exp(z[1])
            rho = sigm(z[2])

            m_1 = (y[:, 0] - mu) ** 2 + (y[:, 1] - mu) ** 2
            m_2 = (y[:, 0] - mu) * (y[:, 1] - mu)
            return 1 * (
                (rho / (1 + rho))
                * (
                    (1 / (s2 * (1 - rho**2))) * (-m_2 * rho**2 + m_1 * rho - m_2)
                    - rho
                )
            )

    def mme(self, y: np.ndarray, w: Union[np.ndarray, float] = 1.0) -> np.ndarray:
        # Return the mean, variance and correlation of a 2d normal distribution
        if np.any(w != 1):
            raise NotImplementedError(
                "Weighted loss not implemented for multivariate normal distribution"
            )
        mu = y.mean()
        s2 = np.mean((y[0] - mu) ** 2 + (y[1] - mu) ** 2)
        rho = np.corrcoef(y.T)[0, 1]
        return np.array([mu, np.log(s2), sigm_inv(rho)])

    def simulate(
        self,
        z: np.ndarray,
        w: Union[np.ndarray, float] = 1.0,
        random_state: Union[int, None] = None,
        rng: Union[np.random.Generator, None] = None,
    ) -> np.ndarray:
        if np.any(w != 1):
            raise NotImplementedError(
                "Weighted loss not implemented for multivariate normal distribution"
            )
        if rng is None:
            rng = np.random.default_rng(seed=random_state)
        mu = np.stack([z[0]] * 2)
        s2 = np.exp(z[1])
        rho = sigm(z[2])
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
        if np.any(w != 1):
            raise NotImplementedError(
                "Weighted loss not implemented for multivariate normal distribution"
            )
        if k == 1:
            return np.array([z[0], z[0]])
        elif k == 2:
            rho = sigm(z[2])
            s2 = np.exp(z[1])
            return np.stack([np.stack([s2, s2 * rho]), np.stack([s2 * rho, s2])])


@inherit_docstrings
class NormalDistribution(Distribution):
    def __init__(
        self,
    ):
        """Initialize a normal distribution object.

        Parameterization: z[0] = mu, z[1] = log(sigma), where E[Y] = w*mu, Var(Y) = w*sigma^2
        """
        super().__init__(n_dim=2)

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

        Parameterization: z[0] = np.log(mu), z[1] = log(theta), where E[Y] = w*mu, Var(Y) = w*mu*(1+mu/theta)
        """
        super().__init__(n_dim=2)

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
        p = np.exp(z[1]) / (np.exp(z[0]) + np.exp(z[1]))
        r = w * np.exp(z[1])
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

        Parameterization: z[0] = log(mu), z[1] = log(lambda), where E[Y] = w*mu, Var(Y) =w*mu^3 / lambda
        """
        super().__init__(n_dim=2)

    def loss(
        self, y: np.ndarray, z: np.ndarray, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        if np.any(w != 1):
            raise NotImplementedError(
                "Weighted loss not implemented for InverseGaussianDistribution"
            )
        return (
            np.exp(z[1]) * (y * np.exp(-2 * z[0]) - 2 * np.exp(-z[0]) + y ** (-1))
            - z[1]
        )

    def grad(
        self, y: np.ndarray, z: np.ndarray, j: int, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        if np.any(w != 1):
            raise NotImplementedError(
                "Weighted loss not implemented for InverseGaussianDistribution"
            )
        if j == 0:
            return 2 * np.exp(z[1] - z[0]) * (1 - y * np.exp(-z[0]))
        elif j == 1:
            return (
                np.exp(z[1]) * (y * np.exp(-2 * z[0]) - 2 * np.exp(-z[0]) + y ** (-1))
                - 1
            )

    def mme(self, y: np.ndarray, w: Union[np.ndarray, float] = 1.0) -> np.ndarray:
        if np.any(w != 1):
            raise NotImplementedError(
                "Weighted loss not implemented for InverseGaussianDistribution"
            )
        mean = y.mean()
        var = np.mean((y - mean) ** 2)
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
        if np.any(w != 1):
            raise NotImplementedError(
                "Weighted loss not implemented for InverseGaussianDistribution"
            )
        if rng is None:
            rng = np.random.default_rng(seed=random_state)
        return rng.wald(np.exp(z[0]), np.exp(z[1]))

    def moment(
        self, z: np.ndarray, k: int, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        if np.any(w != 1):
            raise NotImplementedError(
                "Weighted loss not implemented for InverseGaussianDistribution"
            )
        if k == 1:
            return np.exp(z[0])
        elif k == 2:
            return np.exp(3 * z[0] - z[1])


@inherit_docstrings
class GammaDistribution(Distribution):
    def __init__(
        self,
    ):
        """
        Initialize a gamma distribution object.

        Parameterization: z[0] = log(mu), z[1] = log(phi), where E[Y] = w*mu, Var(Y) =w*phi * mu^2
        """
        super().__init__(n_dim=2)

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
        var = sum((y - y.mean()) ** 2) / w.sum()
        z0 = np.log(mean)
        z1 = 2 * z0 - np.log(var - np.exp(z0))

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

        Parameterization: z[0] = log(mu), z[1] = log(v), where E[Y] = w*mu, Var(Y) =w*mu*(1+mu)/v
        """
        super().__init__(n_dim=2)

    def loss(
        self, y: np.ndarray, z: np.ndarray, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        if np.any(np.any(w != 1)):
            raise NotImplementedError(
                "Weighted loss not implemented for BetaPrimeDistribution"
            )
        return (
            (np.exp(z[0]) + np.exp(z[1]) + np.exp(z[0] + z[1])) * np.log(y + 1)
            - np.exp(z[0]) * (np.exp(z[1]) + 1) * np.log(y)
            + loggamma(np.exp(z[0]) * (np.exp(z[1]) + 1))
            + loggamma(np.exp(z[1]) + 2)
            - loggamma(np.exp(z[0]) + np.exp(z[1]) + np.exp(z[0] + z[1]) + 2)
        )

    def grad(
        self, y: np.ndarray, z: np.ndarray, j: int, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        if np.any(w != 1):
            raise NotImplementedError(
                "Weighted loss not implemented for BetaPrimeDistribution"
            )
        if j == 0:
            return (
                np.exp(z[0])
                * (1 + np.exp(z[1]))
                * (
                    polygamma(0, np.exp(z[0]) * (1 + np.exp(z[1])))
                    - polygamma(
                        0,
                        np.exp(z[0]) + np.exp(z[1]) + np.exp(z[0] + z[1]) + 2,
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
                * polygamma(
                    0,
                    np.exp(z[0]) * (np.exp(z[1]) + 1) + np.exp(z[1]) + 2,
                )
            )

    def mme(self, y: np.ndarray, w: Union[np.ndarray, float] = 1.0) -> np.ndarray:
        if np.any(w != 1):
            raise NotImplementedError(
                "Weighted loss not implemented for BetaPrimeDistribution"
            )
        mean = y.mean()
        var = np.mean((y - mean) ** 2)
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
        if np.any(w != 1):
            raise NotImplementedError(
                "Weighted loss not implemented for BetaPrimeDistribution"
            )
        if rng is None:
            rng = np.random.default_rng(seed=random_state)
        mu = np.exp(z[0])
        v = np.exp(z[1])
        alpha = mu * (1 + v)
        beta = v + 2
        y0 = rng.beta(alpha, beta)
        return y0 / (1 - y0)

    def moment(
        self, z: np.ndarray, k: int, w: Union[np.ndarray, float] = 1.0
    ) -> np.ndarray:
        if np.any(w != 1):
            raise NotImplementedError(
                "Weighted loss not implemented for BetaPrimeDistribution"
            )
        if k == 1:
            return np.exp(z[0])
        elif k == 2:
            return np.exp(z[0] - z[1]) * (1 + np.exp(z[0]))


@inherit_docstrings
class CustomDistribution(Distribution):
    def __init__(
        self,
        loss: Callable[[np.ndarray, np.ndarray, Union[np.ndarray, float]], np.ndarray],
        grad: Callable[
            [np.ndarray, np.ndarray, int, Union[np.ndarray, float]], np.ndarray
        ],
        mme: Callable[[np.ndarray, Union[np.ndarray, float]], np.ndarray],
        simulate: Callable[
            [
                np.ndarray,
                Union[np.ndarray, float],
                Union[int, None],
                Union[np.random.Generator, None],
            ],
            np.ndarray,
        ],
        moment: Callable[[np.ndarray, int, Union[np.ndarray, float]], np.ndarray],
        d: int,
    ):
        """
        Initialize a custom distribution object.

        :param loss: A function that computes the loss.
        :param grad: A function that computes the gradient of the loss.
        :param mme: A function that computes the method of moments estimator.
        :param simulate: A function that simulates data from the distribution.
        :param moment: A function that computes the kth moment of the distribution.
        :param d: The dimension of the distribution.
        """
        super().__init__(n_dim=d)
        self.loss = loss
        self.grad = grad
        self.mme = mme
        self.simulate = simulate
        self.moment = moment


def initiate_distribution(
    distribution: str = "custom",
    loss: Union[
        None, Callable[[np.ndarray, np.ndarray, Union[np.ndarray, float]], np.ndarray]
    ] = None,
    grad: Union[
        None,
        Callable[[np.ndarray, np.ndarray, int, Union[np.ndarray, float]], np.ndarray],
    ] = None,
    mme: Union[
        None, Callable[[np.ndarray, Union[np.ndarray, float]], np.ndarray]
    ] = None,
    simulate: Union[
        None,
        Callable[
            [
                np.ndarray,
                Union[np.ndarray, float],
                Union[int, None],
                Union[np.random.Generator, None],
            ],
            np.ndarray,
        ],
    ] = None,
    moment: Union[
        None, Callable[[np.ndarray, int, Union[np.ndarray, float]], np.ndarray]
    ] = None,
    n_dim: int = 2,
) -> Distribution:
    """
    Returns a probability distribution object based on the distribution name.

    :param distribution: A string representing the name of the distribution to create.
    :param loss: A function that computes the loss. Only used if dist == "custom".
    :param grad: A function that computes the gradient of the loss. Only used if dist == "custom".
    :param mme: A function that computes the method of moments estimator. Only used if dist == "custom".
    :param simulate: A function that simulates data from the distribution. Only used if dist == "custom".
    :param moment: A function that computes the kth moment of the distribution. Only used if dist == "custom".
    :param d: The dimension of the distribution.
    :return: A probability distribution object based on the distribution name.
    :raises UnknownDistribution: If the input distribution name is not recognized.
    """
    if distribution == "normal":
        if n_dim == 2:
            return NormalDistribution()
        if n_dim == 3:
            return MultivariateNormalDistribution()
        raise ValueError(f"{n_dim}-dimensional normal distribution not implemented")
    if distribution == "gamma":
        return GammaDistribution()
    if distribution == "beta_prime":
        return BetaPrimeDistribution()
    if distribution == "inv_gauss":
        return InverseGaussianDistribution()
    if distribution == "neg_bin":
        return NegativeBinomialDistribution()
    if distribution == "custom":
        return CustomDistribution(
            loss=loss, grad=grad, mme=mme, simulate=simulate, moment=moment, d=d
        )
    raise ValueError(f"Unknown distribution: {distribution}")


if __name__ == "__main__":
    normal_dist = initiate_distribution(distribution="normal")
    print(normal_dist.loss.__doc__)
