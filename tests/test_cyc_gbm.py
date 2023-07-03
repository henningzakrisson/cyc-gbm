import unittest

import numpy as np

from cyc_gbm import CyclicalGradientBooster
from cyc_gbm.tune_kappa import tune_kappa
from cyc_gbm.distributions import initiate_distribution


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
        dist = initiate_distribution(distribution="normal")
        y = dist.simulate(z, random_state=10)

        kappa = [100, 0]
        gbm = CyclicalGradientBooster(distribution="normal", kappa=kappa)
        gbm.fit(X, y)
        loss = gbm.dist.loss(y=y, z=gbm.predict(X)).sum()

        self.assertAlmostEqual(
            first=51.017764411696184,
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
        dist = initiate_distribution(distribution="gamma")
        y = dist.simulate(z, random_state=10)

        kappa = [100, 0]
        eps = 0.1
        gbm = CyclicalGradientBooster(distribution="gamma", kappa=kappa, eps=eps)
        gbm.fit(X, y)
        loss = gbm.dist.loss(y=y, z=gbm.predict(X)).sum()

        self.assertAlmostEqual(
            first=130.9327996047943,
            second=loss,
            places=5,
            msg="UniGBM Gamma distribution sse not as expected",
        )

    def test_kappa_tuning_uni(self):
        """Tests the `tune_kappa` function to ensure it returns the correct value of the kappa parameter for uniparametric distributions.

        :raises AssertionError: If the estimated number of boosting steps does not match the expecter number.
        """
        expected_kappa = 35
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
        gbm = CyclicalGradientBooster(
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
        gbm = CyclicalGradientBooster(kappa=kappas, eps=eps, distribution="gamma")
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
        expected_kappa = [16, 15]
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
            distribution="normal",
            n_splits=4,
            random_state=random_state,
        )
        for j in [0, 1]:
            self.assertEqual(
                first=expected_kappa[j],
                second=tuning_results["kappa"][j],
                msg=f"CycGBM Tuning method not giving expected result for dimension {j} in normal distribution",
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

        gbm = CyclicalGradientBooster(
            kappa=kappa,
            eps=eps,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            distribution="beta_prime",
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
        gbm = CyclicalGradientBooster(distribution="inv_gauss", kappa=kappa)
        gbm.fit(X, y)
        z_hat = gbm.predict(X)
        loss = gbm.dist.loss(y=y, z=z_hat).sum()

        self.assertAlmostEqual(
            first=expected_loss,
            second=loss,
            places=5,
            msg="CycGBM Inverse Gaussian distribution loss not as expected",
        )

    def test_negative_binomial(self):
        """
        Test method for the `CycGBM` class on a dataset where the target variable
        follows a negative binomial distribution.

        :raises AssertionError: If the calculated loss does not match the expected loss
            to within a tolerance.
        """
        expected_loss = -105.64814333065765
        n = 100
        rng = np.random.default_rng(seed=10)
        X = rng.normal(0, 1, (n, 2))
        z0 = -1 + 0.004 * np.minimum(2, X[:, 0]) ** 2 + 2.2 * np.minimum(0.5, X[:, 1])
        z1 = -2 + 0.3 * (X[:, 1] > 0) + 0.2 * np.abs(X[:, 1]) * (X[:, 0] > 0)
        z = np.stack([z0, z1])
        distribution = initiate_distribution(distribution="neg_bin")
        y = distribution.simulate(z=z, random_state=5)

        kappa = 100
        eps = 0.01
        gbm = CyclicalGradientBooster(distribution="neg_bin", kappa=kappa)
        gbm.fit(X, y)
        z_hat = gbm.predict(X)
        loss = gbm.dist.loss(y=y, z=z_hat).sum()

        # The tolerance is set to 1 decimal place because the negative binomial
        # loss is not convex in two dimensions
        self.assertAlmostEqual(
            first=expected_loss,
            second=loss,
            places=1,
            msg="CycGBM Negative Binomial distribution loss not as expected",
        )

    def test_multivariate_normal(self):
        """
        Test method for the `CycGBM` class on a dataset where the target variable
        follows a multivariate normal distribution.
        :raises AssertionError: If the calculated loss does not match the expected loss
            to within a tolerance.
        """
        expected_loss = 2707.27301145109
        rng = np.random.default_rng(seed=10)
        n = 1000
        p = 9
        X = np.concatenate([np.ones((1, n)), rng.normal(0, 1, (p - 1, n))]).T
        z0 = (
            1.5 * X[:, 1]
            + 2 * X[:, 3]
            - 0.65 * X[:, 2] ** 2
            + 0.5 * np.abs(X[:, 3]) * np.sin(0.5 * X[:, 2])
            + 0.45 * X[:, 4] * X[:, 5] ** 2
        )
        z1 = 1 + 0.02 * X[:, 2] + 0.5 * X[:, 1] * (X[:, 1] < 2) + 1.8 * (X[:, 5] > 0)
        z2 = 0.2 * X[:, 3] + 0.03 * X[:, 2] ** 2
        z = np.stack([z0, z1, z2])
        distribution = initiate_distribution(distribution="multivariate_normal")
        y = distribution.simulate(z=z, random_state=5)

        kappa = [23, 17, 79]
        eps = [0.5, 0.25, 0.1]
        gbm = CyclicalGradientBooster(distribution="multivariate_normal", kappa=kappa, eps=eps)
        gbm.fit(X, y)
        z_hat = gbm.predict(X)
        loss = gbm.dist.loss(y=y, z=z_hat).sum()

        self.assertAlmostEqual(
            first=expected_loss,
            second=loss,
            places=4,
            msg="CycGBM Multivariate Normal distribution loss not as expected",
        )

    def test_feature_importance(self):
        """Test method for the 'CycGBM' class to test the feature importance calculation."""
        rng = np.random.default_rng(seed=10)
        n = 10000
        p = 5
        X = np.concatenate([np.ones((1, n)), rng.normal(0, 1, (p - 1, n))]).T
        z0 = 1.5 * X[:, 1] + 2 * X[:, 2]
        z1 = 1 + 1.2 * X[:, 1]
        z = np.stack([z0, z1])
        distribution = initiate_distribution(distribution="normal")
        y = distribution.simulate(z=z, random_state=5)

        kappa = 100
        eps = 0.1
        max_depth = 2
        gbm = CyclicalGradientBooster(distribution="normal", kappa=kappa, eps=eps, max_depth=max_depth)
        gbm.fit(X, y)

        feature_importances = {j: gbm.feature_importances(j=j) for j in [0, 1, "all"]}
        expected_feature_importances = {
            0: [0, 0.27203, 0.72798, 0, 0],
            1: [0, 0.94076, 0.05484, 0.00224, 0.00217],
            "all": [0, 0.64087, 0.3567, 0.00123, 0.0012],
        }
        for j in [0, 1, "all"]:
            for feature in range(p):
                self.assertAlmostEqual(
                    first=expected_feature_importances[j][feature],
                    second=feature_importances[j][feature],
                    places=5,
                    msg=f"CycGBM feature importance not as expected for feature {feature}, parameter {j}",
                )

    def test_gamma_with_weights(self):
        """
        Test method for the `CycGBM` class on a dataset where the target variable
        follows a gamma distribution with weights.
        :raises AssertionError: If the calculated loss does not match the expected loss
            to within a tolerance.
        """
        rng = np.random.default_rng(seed=10)
        n = 1000
        expected_loss = 1208.247290263608
        X0 = np.arange(0, n)
        X1 = np.arange(0, n)
        rng.shuffle(X1)
        z0 = 1 * (X0 > 0.3 * n) + 0.5 * (X1 > 0.5 * n)
        z1 = 1 + 1 * (X0 < 0.4 * n)
        rng = np.random.default_rng(seed=10)
        w = rng.poisson(10, n)

        X = np.stack([X0, X1]).T
        z = np.stack([z0, z1])
        distribution = initiate_distribution(distribution="gamma")
        y = distribution.simulate(z=z, w=w, random_state=5)

        kappas = [15, 30]
        eps = 0.1
        gbm = CyclicalGradientBooster(kappa=kappas, eps=eps, distribution="gamma")
        gbm.fit(X, y)
        z_hat = gbm.predict(X)

        loss = gbm.dist.loss(y=y, z=z_hat).sum()

        self.assertAlmostEqual(
            first=expected_loss,
            second=loss,
            places=5,
            msg="CycGBM Gamma distribution with weights loss not as expected",
        )

    def test_normal_with_weights(self):
        """
        Test method for the `CycGBM` class on a dataset where the target variable
        follows a normal distribution with weights.
        :raises AssertionError: If the calculated loss does not match the expected loss
            to within a tolerance.
        """
        rng = np.random.default_rng(seed=10)
        n = 1000
        expected_loss = 3069.316477311563
        X0 = np.arange(0, n)
        X1 = np.arange(0, n)
        rng.shuffle(X1)
        z0 = 1 * (X0 > 0.3 * n) + 0.5 * (X1 > 0.5 * n)
        z1 = 1 + 1 * (X0 < 0.4 * n)
        rng = np.random.default_rng(seed=10)
        w = rng.poisson(10, n)

        X = np.stack([X0, X1]).T
        z = np.stack([z0, z1])
        distribution = initiate_distribution(distribution="normal")
        y = distribution.simulate(z=z, w=w, random_state=5)

        kappas = [15, 30]
        eps = 0.1
        gbm = CyclicalGradientBooster(kappa=kappas, eps=eps, distribution="normal")
        gbm.fit(X, y)
        z_hat = gbm.predict(X)

        loss = gbm.dist.loss(y=y, z=z_hat).sum()

        self.assertAlmostEqual(
            first=expected_loss,
            second=loss,
            places=5,
            msg="CycGBM Normal distribution with weights loss not as expected",
        )


if __name__ == "__main__":
    unittest.main()
