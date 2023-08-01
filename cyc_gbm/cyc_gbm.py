from typing import List, Union, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from cyc_gbm.utils.distributions import Distribution, initiate_distribution
from cyc_gbm.utils.logger import CycGBMLogger
from cyc_gbm.utils.fix_datatype import fix_datatype
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
        feature_selection: Optional[List[List[Union[str, int]]]] = None,
    ):
        """
        Initialize a CyclicalGradientBooster object.

        :param distribution: distribution for losses and gradients. String or Distribution object.
        :param learning_rate: Shrinkage factors, which scales the contribution of each tree. Dimension-wise or global for all parameter dimensions.
        :param n_estimators: Number of boosting steps. Dimension-wise or global for all parameter dimensions.
        :param min_samples_split: Minimum number of samples required to split an internal node. Dimension-wise or global for all parameter dimensions.
        :param min_samples_leaf: Minimum number of samples required at a leaf node. Dimension-wise or global for all parameter dimensions.
        :param max_depth: Maximum depths of each decision tree. Dimension-wise or global for all parameter dimensions.
        :param feature_selection: Features to use for each parameter dimension. If None, all feature_selection are used for all parameter dimensions.
        """
        if isinstance(distribution, str):
            self.distribution = initiate_distribution(distribution=distribution)
        else:
            self.distribution = distribution
        self.n_dim = self.distribution.n_dim

        self.n_estimators = self._setup_hyper_parameter(n_estimators)
        self.learning_rate = self._setup_hyper_parameter(learning_rate)
        self.min_samples_split = self._setup_hyper_parameter(min_samples_split)
        self.min_samples_leaf = self._setup_hyper_parameter(min_samples_leaf)
        self.max_depth = self._setup_hyper_parameter(max_depth)
        self.feature_selection = feature_selection

        self.z0 = 0
        self.trees = self._initialize_trees()
        self.feature_names = None
        self.n_features = None

    def _setup_hyper_parameter(self, hyper_parameter) -> List:
        """
        Initialize parameter to default_value if parameter is not a list or numpy array.

        :param hyperparameter: parameter to initialize
        """
        hyper_parameter_list = (
            hyper_parameter
            if isinstance(hyper_parameter, (list, np.ndarray))
            else [hyper_parameter] * self.n_dim
        )
        if len(hyper_parameter_list) != self.n_dim:
            raise ValueError(
                f"All hyperparameters must be a list as long as the number of parameter dimensions {self.n_dim} or a single value."
            )
        return hyper_parameter_list

    def _initialize_trees(self):
        """
        Initialize trees for each parameter dimension.
        """
        return [
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

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
        w: Union[np.ndarray, pd.Series, float] = None,
        logger: Optional[CycGBMLogger] = None,
    ) -> None:
        """
        Train the model on the given training data.

        :param X: Input data matrix of shape (n_samples, n_features).
        :param y: True response values for the input data.
        :param w: Weights for the training data, of shape (n_samples,). Default is 1 for all samples.
        :param feature_selection: Dictionary of feature_selection to use for each parameter dimension. Default is all for all.
        :param logger: Logger object to log progress. Default is no logger.
        """
        if logger is None:
            logger = CycGBMLogger(verbose=0)
        self._initialize_feature_metadata(X=X)
        X, y, w = fix_datatype(X=X, y=y, w=w)

        self.z0 = self.initialize_estimate(y=y, w=w)
        z = np.tile(self.z0, (len(y), 1)).T
        for k in range(0, max(self.n_estimators)):
            for j in range(self.n_dim):
                if k < self.n_estimators[j]:
                    self.trees[j][k].fit_gradients(
                        X=X[:, self.feature_selection[j]], y=y, z=z, w=w, j=j
                    )
                    z[j] += self.learning_rate[j] * self.trees[j][k].predict(
                        X[:, self.feature_selection[j]]
                    )

                    logger.log_progress(
                        step=(k + 1) * (j + 1),
                        total_steps=(max(self.n_estimators) * self.n_dim),
                        verbose=2,
                    )

        # Adjust initial estimate given current tree estimates
        self.z0 += self.initialize_estimate(y=y, w=w, z=z)

    def _initialize_feature_metadata(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        """Get the feature names from the input data.
        If the input data is a DataFrame, the column names are returned.
        Otherwise, the features are named 0, 1, ..., p-1.
        The feature selection is fixed to comply with the numpy array format.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            if self.feature_selection is not None:
                self.feature_selection = [
                    [X.columns.get_loc(f) for f in self.feature_selection[j]]
                    for j in range(self.n_dim)
                ]
        else:
            self.feature_names = list(np.arange(X.shape[1]))
        if self.feature_selection is None:
            self.feature_selection = [
                list(range(X.shape[1])) for j in range(self.n_dim)
            ]
        self.n_features = X.shape[1]

    def initialize_estimate(
        self,
        y: np.ndarray,
        w: np.ndarray,
        z: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Initialize the estimate of the parameter vector z0.
        If z is not None, the estimate is initialized to fit to the residuals of z.

        :param y: The target values for the training data.
        :param w: The weights for the training data.
        :param z: The current estimate of the parameter vector. If None, the estimate is initialized from zero.
        """
        if z is None:
            z = np.zeros((self.n_dim, len(y)))
        return minimize(
            fun=lambda z0: self.distribution.loss(y=y, z=z0[:, None] + z, w=w).sum(),
            x0=self.distribution.mme(y=y, w=w),
        )["x"]

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
        X, y, w = fix_datatype(X=X, y=y, w=w, feature_names=self.feature_names)
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
        self.trees[j][-1].fit_gradients(
            X=X[:, self.feature_selection[j]], y=y, z=z, w=w, j=j
        )
        self.n_estimators[j] += 1

    def predict(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Predict response values for the input data using the trained model.

        :param X: Input data matrix of shape (n_samples, n_features).
        :return: Predicted response values of shape (d,n_samples).
        """
        X = fix_datatype(X=X, feature_names=self.feature_names)
        return self.z0[:, None] + np.array(
            [
                self.learning_rate[j]
                * np.sum(
                    [
                        tree.predict(X[:, self.feature_selection[j]])
                        for tree in self.trees[j]
                    ],
                    axis=0,
                )
                if self.trees[j]
                else np.zeros(len(X))
                for j in range(self.n_dim)
            ]
        )

    def compute_feature_importances(
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
                    [tree.compute_feature_importances() for tree in self.trees[j]]
                ).sum(axis=0)
                feature_importances[
                    self.feature_selection[j]
                ] += feature_importances_from_trees
        else:
            feature_importances = np.zeros(self.n_features)
            feature_importances_from_trees = np.array(
                [tree.compute_feature_importances() for tree in self.trees[j]]
            ).sum(axis=0)

            feature_importances[
                self.feature_selection[j]
            ] = feature_importances_from_trees
        if normalize:
            feature_importances /= feature_importances.sum()

        if self.feature_names is not None:
            feature_importances = pd.Series(
                feature_importances, index=self.feature_names
            )
        return feature_importances

    def reset(self, n_estimators: Optional[Union[int, List[int]]] = None) -> None:
        """
        Resets the model to its initial state.

        :param n_estimators: Number of estimators to reset to. If None, the model is reset to its initial state.
        """
        if n_estimators is not None:
            self.n_estimators = (
                [n_estimators] * self.n_dim
                if isinstance(n_estimators, int)
                else n_estimators
            )

        self.trees = self._initialize_trees()

        self.z0 = None
        self.n_features = None
        self.feature_names = None
