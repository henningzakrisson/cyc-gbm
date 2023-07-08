import unittest

import numpy as np
import pandas as pd

from cyc_gbm import CyclicalGradientBooster
from cyc_gbm.utils.tuning import tune_n_estimators
from cyc_gbm.utils.distributions import initiate_distribution


class CyclicalGradientBoosterTestCase(unittest.TestCase):
    """
    A class that defines unit tests for the CyclicalGradientBooster class.
    """

    def setUp(self):
        """
        Set up for the unit tests.
        """
        self.rng = np.random.default_rng(seed=11)
        n = 1000
        p = 4
        self.X = self.rng.normal(0, 1, (n, p))
        z_0 = self.X[:, 0] ** 2 + np.sin(2 * self.X[:, 1])
        z_1 = 0.5 + 0.5 * (self.X[:, 2] > 0) - 0.75 * (self.X[:, 3] > 0)
        z_2 = self.X[:, 1] * self.X[:, 2]
        self.z = np.stack([z_0, z_1, z_2])
        self.n_estimators = 25
        self.w = self.rng.choice([0.5, 1.0, 2.0], size=n, replace=True)

    def test_normal_distribution_uni(self):
        """
        Test method for the CycGBM` class on a dataset where the target variable
        follows a univariate normal distribution with constant variance.

        :raises AssertionError: If the calculated loss does not match the expected loss
            to within a tolerance.
        """
        distribution = initiate_distribution(distribution="normal")
        z = np.stack([self.z[0], np.zeros(self.z[0].shape)])
        y = distribution.simulate(z=z, rng=self.rng)

        gbm = CyclicalGradientBooster(
            distribution=distribution,
            n_estimators=[self.n_estimators, 0],
        )
        gbm.fit(X=self.X, y=y)

        self.assertAlmostEqual(
            first=0.5521796794285049,
            second=distribution.loss(y=y, z=gbm.predict(self.X)).mean(),
            places=5,
            msg="UniGBM Normal distribution loss not as expected",
        )

    def test_n_estimators_tuning_uni(self):
        """Tests the `tune_n_estimators` function to ensure it returns the correct value of the n_estimators parameter for uniparametric distributions.

        :raises AssertionError: If the estimated number of boosting steps does not match the expecter number.
        """
        distribution = initiate_distribution(distribution="normal")
        z = np.stack([self.z[0], np.zeros(self.z[0].shape)])
        y = distribution.simulate(z=z, rng=self.rng)

        tuning_results = tune_n_estimators(
            X=self.X,
            y=y,
            rng=self.rng,
            n_estimators_max=[100, 0],
        )

        self.assertEqual(
            first=52,
            second=tuning_results["n_estimators"][0],
            msg="Optimal number of boosting steps not correct for CycGBM in normal distribution with constant variance",
        )

    def test_normal_distribution(self):
        """
        Test method for the `CycGBM` class on a dataset where the target variable follows a normal distribution.

        :raises AssertionError: If the calculated loss does not match the expected loss to within a tolerance.
        """
        distribution = initiate_distribution(distribution="normal")
        y = distribution.simulate(z=self.z[:2], rng=self.rng)

        gbm = CyclicalGradientBooster(
            n_estimators=self.n_estimators,
        )
        gbm.fit(self.X, y)

        self.assertAlmostEqual(
            first=0.8849930601326066,
            second=distribution.loss(y=y, z=gbm.predict(self.X)).mean(),
            places=5,
            msg="CycGBM Normal distribution loss not as expected",
        )

    def test_gamma_distribution(self):
        """
        Test method for the `CycGBM` class on a dataset where the target variable
        follows a gamma distribution.

        :raises AssertionError: If the calculated loss does not match the expected loss
            to within a tolerance.
        """
        distribution = initiate_distribution(distribution="gamma")
        y = distribution.simulate(z=self.z[:2], rng=self.rng)

        gbm = CyclicalGradientBooster(
            distribution=distribution,
            n_estimators=self.n_estimators,
        )
        gbm.fit(self.X, y)

        self.assertAlmostEqual(
            first=2.0453458895192274,
            second=distribution.loss(y=y, z=gbm.predict(self.X)).mean(),
            places=5,
            msg="CycGBM Gamma distribution loss not as expected",
        )

    def test_n_estimators_tuning_cyc(self):
        """Tests the `tune_n_estimators` function to ensure it returns the correct value of the n_estimators parameter for multiparametric distributions.

        :raises AssertionError: If the estimated number of boosting steps does not match the expecter number.
        """
        distribution = initiate_distribution(distribution="normal")
        y = distribution.simulate(z=self.z[:2], rng=self.rng)

        tuning_results = tune_n_estimators(
            X=self.X,
            y=y,
            rng=self.rng,
            n_estimators_max=100,
            distribution=distribution,
            n_splits=2,
        )
        expected_n_estimators = [32, 24]
        for j in range(distribution.n_dim):
            self.assertEqual(
                first=expected_n_estimators[j],
                second=tuning_results["n_estimators"][j],
                msg=f"Optimal number of boosting steps not correct for parameter dimension {j} in CycGBM in normal distribution with constant variance",
            )

    def test_beta_prime(self):
        """
        Test method for the `CycGBM` class on a dataset where the target variable
        follows a beta prime distribution.

        :raises AssertionError: If the calculated loss does not match the expected loss
            to within a tolerance.
        """
        distribution = initiate_distribution(distribution="beta_prime")
        y = distribution.simulate(z=self.z[:2], rng=self.rng)

        gbm = CyclicalGradientBooster(
            distribution=distribution,
            n_estimators=self.n_estimators,
        )
        gbm.fit(self.X, y)

        self.assertAlmostEqual(
            first=-1.3969559499821258,
            second=distribution.loss(y=y, z=gbm.predict(self.X)).mean(),
            places=5,
            msg="CycGBM beta_prime distribution loss not as expected",
        )

    def test_inv_gaussian(self):
        """
        Test method for the `CycGBM` class on a dataset where the target variable
        follows an Inverse Gaussian distribution

        :raises AssertionError: If the calculated loss does not match the expected loss
            to within a tolerance.
        """
        distribution = initiate_distribution(distribution="inv_gauss")
        y = distribution.simulate(z=self.z[:2], rng=self.rng)

        gbm = CyclicalGradientBooster(
            distribution=distribution,
            n_estimators=self.n_estimators,
        )
        gbm.fit(self.X, y)

        self.assertAlmostEqual(
            first=0.6519232343689577,
            second=distribution.loss(y=y, z=gbm.predict(self.X)).mean(),
            places=5,
            msg="CycGBM inv_gaussian distribution loss not as expected",
        )

    def test_negative_binomial(self):
        """
        Test method for the `CycGBM` class on a dataset where the target variable
        follows a negative binomial distribution.

        :raises AssertionError: If the calculated loss does not match the expected loss
            to within a tolerance.
        """
        distribution = initiate_distribution(distribution="neg_bin")
        y = distribution.simulate(z=self.z[:2], rng=self.rng)

        gbm = CyclicalGradientBooster(
            distribution=distribution,
            n_estimators=self.n_estimators,
        )
        gbm.fit(self.X, y)

        self.assertAlmostEqual(
            first=-12944.715688945882,
            second=distribution.loss(y=y, z=gbm.predict(self.X)).mean(),
            places=5,
            msg="CycGBM neg_bin distribution loss not as expected",
        )

    def test_multivariate_normal(self):
        """
        Test method for the `CycGBM` class on a dataset where the target variable
        follows a multivariate normal distribution.
        :raises AssertionError: If the calculated loss does not match the expected loss
            to within a tolerance.
        """
        distribution = initiate_distribution(distribution="normal", n_dim=3)
        y = distribution.simulate(z=self.z, rng=self.rng)

        gbm = CyclicalGradientBooster(
            distribution=distribution,
            n_estimators=self.n_estimators,
        )
        gbm.fit(self.X, y)

        self.assertAlmostEqual(
            first=1.4502626196695108,
            second=distribution.loss(y=y, z=gbm.predict(self.X)).mean(),
            places=5,
            msg="CycGBM multivariate normal distribution loss not as expected",
        )

    def test_feature_importance(self):
        """Test method for the 'CycGBM' class to test the feature importance calculation."""
        distribution = initiate_distribution(distribution="normal")
        y = distribution.simulate(z=self.z[:2], rng=self.rng)

        gbm = CyclicalGradientBooster(
            n_estimators=self.n_estimators,
        )
        gbm.fit(self.X, y)

        feature_importances = {
            j: gbm.calculate_feature_importances(j=j) for j in [0, 1, "all"]
        }
        expected_feature_importances = {
            0: [0.53851, 0.45073, 0.00092, 0.00984],
            1: [0.03491, 0.02740, 0.27443, 0.66326],
            "all": [0.28105, 0.23431, 0.14075, 0.34389],
        }
        for j in [0, 1, "all"]:
            for feature in range(self.X.shape[1]):
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
        distribution = initiate_distribution(distribution="gamma")
        y = distribution.simulate(z=self.z[:2], w=self.w, rng=self.rng)

        gbm = CyclicalGradientBooster(
            distribution=distribution,
            n_estimators=self.n_estimators,
        )
        gbm.fit(X=self.X, y=y, w=self.w)

        self.assertAlmostEqual(
            first=2.083812889187767,
            second=distribution.loss(y=y, z=gbm.predict(self.X), w=self.w).mean(),
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
        distribution = initiate_distribution(distribution="normal")
        y = distribution.simulate(z=self.z[:2], w=self.w, rng=self.rng)

        gbm = CyclicalGradientBooster(
            distribution=distribution,
            n_estimators=self.n_estimators,
        )
        gbm.fit(X=self.X, y=y, w=self.w)

        self.assertAlmostEqual(
            first=0.9091152884599392,
            second=distribution.loss(y=y, z=gbm.predict(self.X), w=self.w).mean(),
            places=5,
            msg="CycGBM Normal distribution with weights loss not as expected",
        )

    def test_pandas_support(self):
        """
        Test method for the `CycGBM` class support for pandas dataframes, to make sure that
        the model can handle both pandas and numpy dataframes and that the column names are
        used instead of the column indices.
        :raises AssertionError: If the calculated loss does not match the expected loss
        """
        X = pd.DataFrame(self.X, columns=["a", "b", "c", "d"])
        w = pd.Series(self.w, name="w")

        distribution = initiate_distribution(distribution="normal")
        y = pd.Series(
            distribution.simulate(z=self.z[:2], w=w.values, rng=self.rng), name="y"
        )

        gbm = CyclicalGradientBooster(
            distribution=distribution,
            n_estimators=self.n_estimators,
        )
        gbm.fit(X=X, y=y, w=w)

        for i in range(4):
            X_shuffled = X.sample(frac=1, axis=1, random_state=10)
            self.assertAlmostEqual(
                first=0.9091152884599392,
                second=distribution.loss(
                    y=y, z=gbm.predict(X_shuffled), w=self.w
                ).mean(),
                places=5,
                msg="CycGBM Normal distribution not invariant to column order",
            )

    def test_selected_features(self):
        """
        Test method for the `CycGBM` class support for pandas dataframes, to make sure that
        the model can handle both pandas and numpy dataframes and that the column names are
        used instead of the column indices.
        :raises AssertionError: If the calculated loss does not match the expected loss
        """
        X = pd.DataFrame(self.X, columns=["a", "b", "c", "d"])
        w = pd.Series(self.w, name="w")

        distribution = initiate_distribution(distribution="normal")
        y = pd.Series(
            distribution.simulate(z=self.z[:2], w=w.values, rng=self.rng), name="y"
        )

        gbm = CyclicalGradientBooster(
            distribution=distribution,
            n_estimators=self.n_estimators,
        )
        gbm.fit(X=X, y=y, w=w, features={0: ["a", "b"], 1: ["c", "d"]})

        expected_feature_importance = {
            0: {
                "a": 0.55302,
                "b": 0.44698,
                "c": 0,
                "d": 0,
            },
            1: {"a": 0, "b": 0, "c": 0.35005, "d": 0.64995},
            "all": {"a": 0.25253, "b": 0.20411, "c": 0.19020, "d": 0.35316},
        }

        feature_importance = {
            0: gbm.calculate_feature_importances(j=0),
            1: gbm.calculate_feature_importances(j=1),
            "all": gbm.calculate_feature_importances(j="all"),
        }

        for j in [0, 1, "all"]:
            for feature in ["a", "b", "c", "d"]:
                self.assertAlmostEqual(
                    first=expected_feature_importance[j][feature],
                    second=feature_importance[j][feature],
                    places=5,
                    msg=f"Feature importance for feature {feature} for parameter j = {j} not as expected",
                )


if __name__ == "__main__":
    unittest.main()
