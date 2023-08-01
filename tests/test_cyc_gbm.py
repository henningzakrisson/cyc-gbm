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

    def _test_distribution(
        self,
        distribution_name: str,
        expected_loss: float,
        n_dim: int,
        use_weights: bool = False,
    ):
        """
        Test method for the CycGBM` class on a dataset where the target variable
        is generated from a given distribution.

        :param distribution_name: The name of the distribution to use.
        :param expected_loss: The expected loss of the CycGBM model.
        :param n_dim: The number of dimensions to use
        :param use_weights: Whether to use weights in the fit method.
        """
        distribution = initiate_distribution(
            distribution=distribution_name, n_dim=n_dim
        )
        z = self.z[:n_dim]
        w = self.w if use_weights else np.ones(self.X.shape[0])
        y = distribution.simulate(z=z, w=w, rng=self.rng)

        gbm = CyclicalGradientBooster(
            distribution=distribution,
            n_estimators=self.n_estimators,
        )
        gbm.fit(X=self.X, y=y, w=w)

        self.assertAlmostEqual(
            first=expected_loss,
            second=distribution.loss(y=y, z=gbm.predict(self.X), w=w).mean(),
            places=5,
            msg=f"CycGBM {distribution_name} distribution loss not as expected",
        )

    def test_normal_distribution(self):
        self._test_distribution(
            distribution_name="normal", expected_loss=0.8777181600299709, n_dim=2
        )

    def test_normal_distribution_weighted(self):
        self._test_distribution(
            distribution_name="normal",
            expected_loss=0.900653358604354,
            n_dim=2,
            use_weights=True,
        )

    def test_gamma_distribution(self):
        self._test_distribution(
            distribution_name="gamma", expected_loss=1.9655223678827858, n_dim=2
        )

    def test_gamma_distribution_weighted(self):
        self._test_distribution(
            distribution_name="gamma",
            expected_loss=1.9967752095726388,
            n_dim=2,
            use_weights=True,
        )

    def test_beta_prime(self):
        self._test_distribution(
            distribution_name="beta_prime", expected_loss=-1.4001803080633544, n_dim=2
        )

    def test_inv_gaussian(self):
        self._test_distribution(
            distribution_name="inv_gauss", expected_loss=0.6066039743569882, n_dim=2
        )

    def test_neg_bin(self):
        self._test_distribution(
            distribution_name="neg_bin", expected_loss=-12944.799372247004, n_dim=2
        )

    def test_multi_normal(self):
        self._test_distribution(
            distribution_name="normal", expected_loss=1.3351243486545377, n_dim=3
        )

    def test_n_estimators_tuning(self):
        """Tests the `tune_n_estimators` function to ensure it returns the correct value of the n_estimators parameter for multiparametric distributions.

        :raises AssertionError: If the estimated number of boosting steps does not match the expecter number.
        """
        distribution = initiate_distribution(distribution="normal")
        y = distribution.simulate(z=self.z[:2], rng=self.rng)

        model = CyclicalGradientBooster(
            distribution=distribution,
            max_depth=2,
            min_samples_leaf=20,
        )

        tuning_results = tune_n_estimators(
            X=self.X,
            y=y,
            rng=self.rng,
            n_estimators_max=[30, 100],
            model=model,
            n_splits=2,
        )
        expected_n_estimators = [30, 24]
        for j in range(distribution.n_dim):
            self.assertEqual(
                first=expected_n_estimators[j],
                second=tuning_results["n_estimators"][j],
                msg=f"Optimal number of boosting steps not correct for parameter dimension {j} in CycGBM in normal distribution with constant variance",
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
            j: gbm.compute_feature_importances(j=j) for j in [0, 1, "all"]
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

        expected_loss = distribution.loss(y=y, z=gbm.predict(X), w=w).mean()

        for i in range(4):
            X_shuffled = X.sample(frac=1, axis=1, random_state=10)
            self.assertAlmostEqual(
                first=expected_loss,
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
            feature_selection=[["a", "b"], ["c", "d"]],
        )
        gbm.fit(X=X, y=y, w=w)

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
            0: gbm.compute_feature_importances(j=0),
            1: gbm.compute_feature_importances(j=1),
            "all": gbm.compute_feature_importances(j="all"),
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
