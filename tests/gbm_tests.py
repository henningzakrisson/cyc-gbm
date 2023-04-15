import unittest
import numpy as np
from src.uni_gbm import UniGBM, tune_kappa
from src.cyc_gbm import CycGBM


class GBMTests(unittest.TestCase):
    """
    A class that defines unit tests for the `GBM` classes.
    """

    def test_normal_distribution_uni(self):
        """
        Test method for the `UniGBM` class on a dataset where the target variable
        follows a normal distribution.

        :raises AssertionError: If the calculated loss does not match the expected loss
            to within a tolerance.
        """
        n = 100
        expected_loss = 181.94200480862173
        rng = np.random.default_rng(seed=10)
        X0 = np.arange(0, n)
        X1 = np.arange(0, n)
        rng.shuffle(X1)
        mu = 10 * (X0 > 0.3 * n) + 5 * (X1 > 0.5 * n)

        X = np.stack([X0, X1]).T
        y = rng.normal(mu, 1.5)

        gbm = UniGBM(dist="normal")
        gbm.train(X, y)
        z_hat = gbm.predict(X)

        loss = gbm.dist.loss(z_hat, y).sum()

        self.assertAlmostEqual(
            first=loss,
            second=expected_loss,
            places=5,
            msg="UniGBM Normal distribution loss not as expected",
        )

    def test_gamma_distribution_uni(self):
        """
        Test method for the `UniGBM` class on a dataset where the target variable
        follows a gamma distribution.

        :raises AssertionError: If the calculated loss does not match the expected loss
            to within a tolerance.
        """

        n = 1000
        expected_sse = 893.2061449825943
        rng = np.random.default_rng(seed=10)
        X0 = np.arange(0, n)
        X1 = np.arange(0, n)
        rng.shuffle(X1)
        mu = 0.1 * (1 + 10 * (X0 > X0.mean()) + 5 * (X1 > X1.mean()))

        X = np.stack([X0, X1]).T
        y = rng.gamma(1, mu)

        gbm = UniGBM(dist="gamma", kappa=100, eps=0.1)
        gbm.train(X, y)
        mu_hat = np.exp(gbm.predict(X))

        sse = sum((y - mu_hat) ** 2)

        self.assertAlmostEqual(
            first=sse,
            second=expected_sse,
            places=5,
            msg="UniGBM Gamma distribution sse not as expected",
        )

    def test_kappa_tuning_uni(self):
        """Tests the `tune_kappa` function to ensure it returns the correct value of the kappa parameter."""
        expected_kappa = 35
        n = 1000
        rng = np.random.default_rng(seed=10)
        X0 = np.arange(0, n)
        X1 = np.arange(0, n)
        rng.shuffle(X1)
        mu = 10 * (X0 > 0.3 * n) + 5 * (X1 > 0.5 * n)

        X = np.stack([X0, X1]).T
        y = rng.normal(mu, 1.5)

        kappa = tune_kappa(X=X, y=y, random_state=5)

        self.assertEqual(
            first=kappa,
            second=expected_kappa,
            msg="Optimal number of boosting steps not correct for UniGBM",
        )

    def test_normal_distribution_cyc(self):
        """
        Test method for the `CycGBM` class on a dataset where the target variable
        follows a normal distribution.

        :raises AssertionError: If the calculated loss does not match the expected loss
            to within a tolerance.
        """
        n = 100
        expected_loss = 641.9173857564037
        rng = np.random.default_rng(seed=10)
        X0 = np.arange(0, n)
        X1 = np.arange(0, n)
        rng.shuffle(X1)
        mu = 10 * (X0 > 0.3 * n) + 5 * (X1 > 0.5 * n)
        sigma = np.exp(1 + 1 * (X0 < 0.4 * n))

        X = np.stack([X0, X1]).T
        y = rng.normal(mu, sigma)

        kappas = [100, 10]
        eps = [0.1, 0.01]
        gbm = CycGBM(kappa=kappas, eps=eps)
        gbm.train(X, y)
        z_hat = gbm.predict(X)

        loss = gbm.dist.loss(z_hat, y).sum()

        self.assertAlmostEqual(
            first=loss,
            second=expected_loss,
            places=5,
            msg="CycGBM Normal distribution loss not as expected",
        )
