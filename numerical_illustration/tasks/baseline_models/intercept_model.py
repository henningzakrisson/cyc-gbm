from typing import Union

import numpy as np

from cyc_gbm.utils.distributions import Distribution


class InterceptModel:
    """
    Class for intercept models, i.e., models that only predict a constant value for all parameters.
    """

    def __init__(self, distribution: Union[str, Distribution] = "normal"):
        """
        Initialize the model.

        :param distribution: distribution for losses and gradients
        """
        self.distribution = distribution
        self.z0 = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: Union[np.ndarray, float] = 1,
    ) -> None:
        """
        Fit the model.

        :param X: Input data matrix of shape (n_samples, n_features).
        :param y: True response values for the input data.
        :param w: Weights for the training data, of shape (n_samples,). Default is 1 for all samples.
        """
        # TODO: Implement this method
        self.z0 = np.zeros(self.distribution.n_dim)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the response for the given input data.

        :param X: Input data matrix of shape (n_samples, n_features).
        :return: Predicted response of shape (n_samples,).
        """
        return np.tile(self.z0, (X.shape[0], 1)).T
