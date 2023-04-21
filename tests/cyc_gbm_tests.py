import unittest
import numpy as np

from src.cyc_gbm import CycGBM
from src.cyc_gbm.utils import tune_kappa
from src.cyc_gbm.distributions import initiate_distribution


class GBMTests(unittest.TestCase):
    """
    A class that defines unit tests for the `GBM` classes.
    """

    def test_normal_distribution_uni(self):
        """
        Test method for the CycGBM` class on a dataset where the target variable
        follows a univariate normal distribution with constant variance.

        :raises AssertionError: If the calculated loss does not match the expected loss
            to within a tolerance.
        """
        n = 100
        rng = np.random.default_rng(seed=10)
        X = rng.normal(0, 1, (n, 2))
        z_0 = 10 * (X[:, 0] > 0.3 * n) + 5 * (X[:, 1] > 0.5 * n)
        z_1 = np.ones(n) * np.log(1.5)
        z = np.stack([z_0, z_1])
        dist = initiate_distribution(dist="normal")
        y = dist.simulate(z, random_state=10)

        kappa = [100, 0]
        gbm = CycGBM(dist="normal", kappa=kappa)
        gbm.fit(X, y)
        loss = gbm.dist.loss(y=y, z=gbm.predict(X)).sum()

        self.assertAlmostEqual(
            first=54.96955230902564,
            second=loss,
            places=5,
            msg="UniGBM Normal distribution loss not as expected",
        )

    def test_gamma_distribution_uni(self):
        """
        Test method for the `CycGBM` class on a dataset where the target variable
        follows a gamma distribution with constant overdispersion

        :raises AssertionError: If the calculated loss does not match the expected loss
            to within a tolerance.
        """
        n = 100
        rng = np.random.default_rng(seed=10)
        X = rng.normal(0, 1, (n, 2))
        z_0 = 0.1 * (1 + 10 * (X[:, 0] > 0) + 5 * (X[:, 1] > 0))
        z_1 = np.ones(n) * np.log(1)
        z = np.stack([z_0, z_1])
        dist = initiate_distribution(dist="gamma")
        y = dist.simulate(z, random_state=10)

        kappa = [100, 0]
        eps = 0.1
        gbm = CycGBM(dist="gamma", kappa=kappa, eps=eps)
        gbm.fit(X, y)
        loss = gbm.dist.loss(y=y, z=gbm.predict(X)).sum()

        self.assertAlmostEqual(
            first=134.98811187268964,
            second=loss,
            places=5,
            msg="UniGBM Gamma distribution sse not as expected",
        )

    def test_kappa_tuning_uni(self):
        """Tests the `tune_kappa` function to ensure it returns the correct value of the kappa parameter for uniparametric distributions.

        :raises AssertionError: If the estimated number of boosting steps does not match the expecter number.
        """
        expected_kappa = 36
        n = 1000
        rng = np.random.default_rng(seed=10)
        X0 = np.arange(0, n)
        X1 = np.arange(0, n)
        rng.shuffle(X1)
        mu = 10 * (X0 > 0.3 * n) + 5 * (X1 > 0.5 * n)

        X = np.stack([X0, X1]).T
        y = rng.normal(mu, 1.5)

        tuning_results = tune_kappa(X=X, y=y, random_state=5, kappa_max=[1000, 0])

        self.assertEqual(
            first=expected_kappa,
            second=tuning_results["kappa"][0],
            msg="Optimal number of boosting steps not correct for CycGBM in normal distribution with constant variance",
        )

    def test_normal_distribution_cyc(self):
        """
        Test method for the `CycGBM` class on a dataset where the target variable
        follows a normal distribution.

        :raises AssertionError: If the calculated loss does not match the expected loss to within a tolerance.
        """
        n = 100
        expected_loss = 186.8538898178347
        rng = np.random.default_rng(seed=10)
        X0 = np.arange(0, n)
        X1 = np.arange(0, n)
        rng.shuffle(X1)
        mu = 10 * (X0 > 0.3 * n) + 5 * (X1 > 0.5 * n)
        sigma = np.exp(1 + 1 * (X0 < 0.4 * n))

        X = np.stack([X0, X1]).T
        y = rng.normal(mu, sigma)

        kappas = [100, 10]
        eps = 0.1
        max_depth = 2
        min_samples_leaf = 20
        gbm = CycGBM(
            kappa=kappas,
            eps=eps,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
        )
        gbm.fit(X, y)
        z_hat = gbm.predict(X)

        loss = gbm.dist.loss(y=y, z=z_hat).sum()

        self.assertAlmostEqual(
            first=expected_loss,
            second=loss,
            places=5,
            msg="CycGBM Normal distribution loss not as expected",
        )

    def test_gamma_distribution_cyc(self):
        """
        Test method for the `CycGBM` class on a dataset where the target variable
        follows a gamma distribution.

        :raises AssertionError: If the calculated loss does not match the expected loss
            to within a tolerance.
        """
        n = 1000
        expected_loss = 2594.5555073093524
        rng = np.random.default_rng(seed=10)
        X0 = np.arange(0, n)
        X1 = np.arange(0, n)
        rng.shuffle(X1)
        mu = np.exp(1 * (X0 > 0.3 * n) + 0.5 * (X1 > 0.5 * n))
        phi = np.exp(1 + 1 * (X0 < 0.4 * n))

        X = np.stack([X0, X1]).T
        y = rng.gamma(1 / phi, mu * phi)

        kappas = [15, 30]
        eps = 0.1
        gbm = CycGBM(kappa=kappas, eps=eps, dist="gamma")
        gbm.fit(X, y)
        z_hat = gbm.predict(X)

        loss = gbm.dist.loss(y=y, z=z_hat).sum()

        self.assertAlmostEqual(
            first=expected_loss,
            second=loss,
            places=5,
            msg="CycGBM Gamma distribution loss not as expected",
        )

    def test_kappa_tuning_cyc(self):
        """Tests the `tune_kappa` function to ensure it returns the correct value of the kappa parameter for multiparametric distributions.

        :raises AssertionError: If the estimated number of boosting steps does not match the expecter number.
        """
        n = 100
        expected_kappa = [12, 16]
        rng = np.random.default_rng(seed=10)
        X0 = np.arange(0, n)
        X1 = np.arange(0, n)
        rng.shuffle(X1)
        mu = 10 * (X0 > 0.3 * n) + 5 * (X1 > 0.5 * n)
        sigma = np.exp(1 + 1 * (X0 < 0.4 * n))

        X = np.stack([X0, X1]).T
        y = rng.normal(mu, sigma)

        kappa_max = [1000, 100]
        eps = 0.1
        max_depth = 2
        min_samples_leaf = 20
        random_state = 5
        tuning_results = tune_kappa(
            X=X,
            y=y,
            kappa_max=kappa_max,
            eps=eps,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            dist="normal",
            n_splits=4,
            random_state=random_state,
        )
        for j in [0, 1]:
            self.assertEqual(
                first=expected_kappa[j],
                second=tuning_results["kappa"][j],
                msg=f"CycGBM Tuning method not giving expected result for dimension {j}",
            )

    def test_beta_prime(self):
        """
        Test method for the `CycGBM` class on a dataset where the target variable
        follows a beta prime distribution.

        :raises AssertionError: If the calculated loss does not match the expected loss
            to within a tolerance.
        """
        expected_loss = 121.22775641886105
        n = 1000
        rng = np.random.default_rng(seed=10)
        X0 = np.arange(0, n) / n
        X1 = np.arange(0, n) / n
        rng.shuffle(X1)
        mu = np.exp(1 * (X0 > 0.3 * n) + 0.5 * (X1 > 0.5 * n))
        v = np.exp(1 + 1 * X0 - 3 * np.abs(X1))

        X = np.stack([X0, X1]).T
        alpha = mu * (1 + v)
        beta = v + 2
        y0 = rng.beta(alpha, beta)
        y = y0 / (1 - y0)

        max_depth = 2
        min_samples_leaf = 20
        eps = [0.1, 0.1]
        kappa = [20, 100]

        gbm = CycGBM(
            kappa=kappa,
            eps=eps,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            dist="beta_prime",
        )
        gbm.fit(X, y)
        z_hat = gbm.predict(X)
        loss = gbm.dist.loss(y=y, z=z_hat).sum()

        self.assertAlmostEqual(
            first=expected_loss,
            second=loss,
            places=3,
            msg="CycGBM BetaPrime distribution loss not as expected",
        )

    def test_inv_gaussian(self):
        """
        Test method for the `CycGBM` class on a dataset where the target variable
        follows an Inverse Gaussian distribution

        :raises AssertionError: If the calculated loss does not match the expected loss
            to within a tolerance.
        """
        n = 100
        expected_loss = 502.33229761058215
        rng = np.random.default_rng(seed=10)
        X0 = np.arange(0, n)
        X1 = np.arange(0, n)
        rng.shuffle(X1)
        mu = np.exp(1 * (X0 > 0.3 * n) + 0.5 * (X1 > 0.5 * n))
        l = np.exp(-1 + 0.1 * X0 - 0.002 * X1**2)

        X = np.stack([X0, X1]).T
        y = rng.wald(mu, l)

        kappa = 100
        eps = 0.001
        gbm = CycGBM(dist="inv_gauss", kappa=kappa)
        gbm.fit(X, y)
        z_hat = gbm.predict(X)
        loss = gbm.dist.loss(y=y, z=z_hat).sum()

        self.assertAlmostEqual(
            first=expected_loss,
            second=loss,
            places=5,
            msg="CycGBM Inverse Gaussian distribution loss not as expected",
        )


if __name__ == "__main__":
    unittest.main()
