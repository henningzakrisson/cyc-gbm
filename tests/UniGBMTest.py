import unittest
import numpy as np
from src.UniGBM import UniGBM


class UniGBMTest(unittest.TestCase):
    """
    A class that defines unit tests for the `UniGBM` class.
    """

    def test_normal_distribution(self):
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
        mu_hat = gbm.predict(X)

        loss = sum((y - mu_hat) ** 2)

        self.assertAlmostEqual(
            first=loss,
            second=expected_loss,
            places=5,
            msg="Normal distribution loss not as expected",
        )

    def test_gamma_distribution(self):
        """
        Test method for the `UniGBM` class on a dataset where the target variable
        follows a gamma distribution.

        :raises AssertionError: If the calculated loss does not match the expected loss
            to within a tolerance.
        """

        n = 1000
        expected_loss = 893.2061449825943
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

        loss = sum((y - mu_hat) ** 2)

        self.assertAlmostEqual(
            first=loss,
            second=expected_loss,
            places=5,
            msg="Gamma distribution loss not as expected",
        )
