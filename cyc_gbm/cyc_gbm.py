from typing import List, Union, Optional, Dict

import numpy as np
import pandas as pd

from cyc_gbm.utils.distributions import Distribution, initiate_distribution
from cyc_gbm.utils.logger import CycGBMLogger
from cyc_gbm.utils.convert_data import convert_data
from cyc_gbm.boosting_tree import BoostingTree


class CyclicalGradientBooster:
    """
    Class for cyclical gradient boosting regressors.

    The model estimates a d-dimensional parameter function theta that depends on X to responses y.
    y is assumed to be observations of Y ~ F(theta(X)), where F is a distribution.
    """

    def __init__(
        self,
        distribution: Union[str, Distribution] = "normal",
        learning_rate: Union[float, List[float]] = 0.1,
        n_estimators: Union[int, List[int]] = 100,
        min_samples_split: Union[int, List[int]] = 2,
        min_samples_leaf: Union[int, List[int]] = 1,
        max_depth: Union[int, List[int]] = 3,
    ):
        """
        Initialize a CyclicalGradientBooster object.

        :param distribution: distribution for losses and gradients. String or Distribution object.
        :param learning_rate: Shrinkage factors, which scales the contribution of each tree. Dimension-wise or global for all parameter dimensions.
        :param n_estimators: Number of boosting steps. Dimension-wise or global for all parameter dimensions.
        :param min_samples_split: Minimum number of samples required to split an internal node. Dimension-wise or global for all parameter dimensions.
        :param min_samples_leaf: Minimum number of samples required at a leaf node. Dimension-wise or global for all parameter dimensions.
        :param max_depth: Maximum depths of each decision tree. Dimension-wise or global for all parameter dimensions.
        """
        if isinstance(distribution, str):
            self.distribution = initiate_distribution(distribution=distribution)
        else:
            self.distribution = distribution
        self.n_dim = self.distribution.n_dim

        self.n_estimators = self._setup_hyper_parameter(hyper_parameter=n_estimators)
        self.learning_rate = self._setup_hyper_parameter(hyper_parameter=learning_rate)
        self.min_samples_split = self._setup_hyper_parameter(
            hyper_parameter=min_samples_split
        )
        self.min_samples_leaf = self._setup_hyper_parameter(
            hyper_parameter=min_samples_leaf
        )
        self.max_depth = self._setup_hyper_parameter(hyper_parameter=max_depth)

        self.z0 = 0
        self.trees = [
            [
                BoostingTree(
                    distribution=self.distribution,
                    max_depth=self.max_depth[j],
                    min_samples_split=self.min_samples_split[j],
                    min_samples_leaf=self.min_samples_leaf[j],
                )
                for _ in range(self.n_estimators[j])
            ]
            for j in range(self.n_dim)
        ]
        self.feature_names = None
        self.n_features = None

    def _setup_hyper_parameter(self, hyper_parameter) -> List:
        """
        Initialize parameter to default_value if parameter is not a list or numpy array.

        :param hyper_parameter: parameter to initialize
        """
        return (
            hyper_parameter
            if isinstance(hyper_parameter, (list, np.ndarray))
            else [hyper_parameter] * self.n_dim
        )

    def _adjust_mle(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
        w: Union[np.ndarray, pd.Series, float] = 1,
    ) -> None:
        """
        Adjust the initial values of the model for parameters not modeled by the GBM

        :param X: Input data matrix of shape (n_samples, n_features).
        :param y: True response values for the input data.
        :param w: Weights for the training data, of shape (n_samples,). Default is 1 for all samples.
        """
        z = self.predict(X=X)
        for j in range(self.n_dim):
            if self.n_estimators[j] == 0:
                self.z0[j] += self.distribution.opt_step(y=y, z=z, w=w, j=j)

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
        w: Union[np.ndarray, pd.Series, float] = None,
        features: Optional[Dict[int, List[Union[str, int]]]] = None,
        logger: Optional[CycGBMLogger] = None,
    ) -> None:
        """
        Train the model on the given training data.

        :param X: Input data matrix of shape (n_samples, n_features).
        :param y: True response values for the input data.
        :param w: Weights for the training data, of shape (n_samples,). Default is 1 for all samples.
        :param features: Dictionary of features to use for each parameter dimension. Default is all for all.
        :param logger: Logger object to log progress. Default is no logger.
        """
        if logger is None:
            logger = CycGBMLogger(verbose=0)
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
            if features is not None:
                self.features = {
                    j: [X.columns.get_loc(f) for f in features[j]]
                    for j in range(self.n_dim)
                }
        X, y, w = convert_data(X=X, y=y, w=w)
        if features is None:
            self.features = {j: list(range(X.shape[1])) for j in range(self.n_dim)}
        self.n_features = X.shape[1]

        self.z0 = self.distribution.mle(y=y, w=w)[:, None]
        z = np.tile(self.z0, (1, len(y)))

        for k in range(0, max(self.n_estimators)):
            for j in range(self.n_dim):
                if k < self.n_estimators[j]:
                    self.trees[j][k].fit_gradients(
                        X=X[:, self.features[j]], y=y, z=z, w=w, j=j
                    )
                    z[j] += self.learning_rate[j] * self.trees[j][k].predict(
                        X[:, self.features[j]]
                    )

                    logger.log_progress(
                        step=(k + 1) * (j + 1),
                        total_steps=(max(self.n_estimators) * self.n_dim),
                        verbose=2,
                    )

        self._adjust_mle(X=X, y=y, w=w)

    def add_tree(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
        j: int,
        z: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        w: Union[np.ndarray, pd.Series, float] = 1,
    ) -> None:
        """
        Updates the current boosting model with one additional tree at dimension j.

        :param X: The training input data, shape (n_samples, n_features).
        :param y: The target values for the training data.
        :param j: Parameter dimension to update
        :param z: Current predictions of the model. If None, the current predictions are calculated.
        :param w: Weights for the training data, of shape (n_samples,). Default is 1 for all samples.
        """
        X, y, w = convert_data(X=X, y=y, w=w, feature_names=self.feature_names)
        if z is None:
            z = self.predict(X)
        self.trees[j].append(
            BoostingTree(
                distribution=self.distribution,
                max_depth=self.max_depth[j],
                min_samples_split=self.min_samples_split[j],
                min_samples_leaf=self.min_samples_leaf[j],
            )
        )
        self.trees[j][-1].fit_gradients(X=X[:, self.features[j]], y=y, z=z, w=w, j=j)
        self.n_estimators[j] += 1

    def predict(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Predict response values for the input data using the trained model.

        :param X: Input data matrix of shape (n_samples, n_features).
        :return: Predicted response values of shape (d,n_samples).
        """
        X, _, _ = convert_data(X=X, feature_names=self.feature_names)
        return self.z0 + np.array(
            [
                self.learning_rate[j]
                * np.sum(
                    [tree.predict(X[:, self.features[j]]) for tree in self.trees[j]],
                    axis=0,
                )
                if self.trees[j]
                else np.zeros(len(X))
                for j in range(self.n_dim)
            ]
        )

    def calculate_feature_importances(
        self, j: Union[str, int] = "all", normalize: bool = True
    ) -> Union[np.ndarray, pd.Series]:
        """
        Computes the feature importances for parameter dimension j

        :param j: Parameter dimension. If 'all', calculate importance over all parameter dimensions.
        :return: Feature importance of shape (n_features,)
        """
        if j == "all":
            feature_importances = np.zeros(self.n_features)
            for j in range(self.n_dim):
                feature_importances_from_trees = np.array(
                    [tree.feature_importances() for tree in self.trees[j]]
                ).sum(axis=0)
                feature_importances[self.features[j]] += feature_importances_from_trees
        else:
            feature_importances = np.zeros(self.n_features)
            feature_importances_from_trees = np.array(
                [tree.feature_importances() for tree in self.trees[j]]
            ).sum(axis=0)

            feature_importances[self.features[j]] = feature_importances_from_trees
        if normalize:
            feature_importances /= feature_importances.sum()

        if self.feature_names is not None:
            feature_importances = pd.Series(
                feature_importances, index=self.feature_names
            )
        return feature_importances
