import unittest
import numpy as np
from src.uni_gbm import UniGBM
from src.uni_gbm import tune_kappa as tune_kappa_uni
from src.cyc_gbm import CycGBM
from src.cyc_gbm import tune_kappa as tune_kappa_cyc


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
            first=expected_loss,
            second=loss,
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
            first=expected_sse,
            second=sse,
            places=5,
            msg="UniGBM Gamma distribution sse not as expected",
        )

    def test_kappa_tuning_uni(self):
        """Tests the `tune_kappa` function to ensure it returns the correct value of the kappa parameter."""
        expected_kappa = 36
        n = 1000
        rng = np.random.default_rng(seed=10)
        X0 = np.arange(0, n)
        X1 = np.arange(0, n)
        rng.shuffle(X1)
        mu = 10 * (X0 > 0.3 * n) + 5 * (X1 > 0.5 * n)

        X = np.stack([X0, X1]).T
        y = rng.normal(mu, 1.5)

        kappa = tune_kappa_uni(X=X, y=y, random_state=5)

        self.assertEqual(
            first=expected_kappa,
            second=kappa,
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
        expected_loss = 187.46122289939993
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
        gbm.train(X, y)
        z_hat = gbm.predict(X)

        loss = gbm.dist.loss(z_hat, y).sum()

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
        expected_loss = 2600.6806681794524
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
        gbm.train(X, y)
        z_hat = gbm.predict(X)

        loss = gbm.dist.loss(z_hat, y).sum()

        self.assertAlmostEqual(
            first=expected_loss,
            second=loss,
            places=5,
            msg="CycGBM Gamma distribution loss not as expected",
        )

    def test_kappa_tuning_cyc(self):
        n = 100
        expected_kappa = [12, 35]
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
        kappa, _ = tune_kappa_cyc(
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
        # Assume two dimensions
        for j in [0, 1]:
            self.assertEqual(
                first=expected_kappa[0],
                second=kappa[0],
                msg=f"CycGBM Tuning method not giving expected result for dimension {j}",
            )
