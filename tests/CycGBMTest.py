import unittest
import numpy as np
from src.CycGBM import CycGBM


class CycGBMTest(unittest.TestCase):
    """
    A class that defines unit tests for the `CycGBM` class.
    """

    def test_normal_distribution(self):
        """
        Test method for the `CycGBM` class on a dataset where the target variable
        follows a normal distribution.

        :raises AssertionError: If the calculated loss does not match the expected loss
            to within a tolerance.
        """
        n = 100
        expected_loss = 641.6928805767581
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
        gbm = CycGBM(kappas=kappas, eps=eps)
        gbm.train(X, y)
        z_hat = gbm.predict(X)

        loss = gbm.loss(z_hat, y).sum()

        self.assertAlmostEqual(
            first=loss,
            second=expected_loss,
            places=5,
            msg="Normal distribution loss not as expected",
        )
