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
        expected_loss = 2.0702865596620234e-26
        rng = np.random.default_rng(seed=10)
        X0 = np.arange(0, 100)
        X1 = np.arange(0, 100)
        rng.shuffle(X1)
        mu = 10 * (X0 > 30) + 5 * (X1 > 50)

        X = np.stack([X0, X1]).T
        y = rng.normal(mu, 1.5)

        gbm = UniGBM()
        gbm.train(X, y)
        y_hat = gbm.predict(X)

        loss = sum(y - y_hat) ** 2

        self.assertEqual(
            loss, expected_loss, msg="Normal distribution loss not as expected"
        )
