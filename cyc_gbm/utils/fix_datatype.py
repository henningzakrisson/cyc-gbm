from typing import Tuple, Union, Optional, List

import numpy as np
import pandas as pd


def fix_datatype(
    X: Union[np.ndarray, pd.DataFrame],
    y: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
    w: Optional[Union[np.ndarray, pd.Series, pd.DataFrame, float]] = None,
    feature_names: Optional[List[str]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Convert data to numpy arrays if they are pandas dataframes or series.

    :param X: Input data matrix of shape (n_samples, n_features).
    :param y: True response values for the input data.
    :param w: Weights for the data, of shape (n_samples,). Default is 1 for all samples.
    :param feature_names: Names of the features in X.
    """
    if isinstance(X, pd.DataFrame):
        if feature_names is not None:
            X = X[feature_names]
        X = X.to_numpy()
    if y is None:
        return X
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.to_numpy()
    if isinstance(w, (pd.Series, pd.DataFrame)):
        w = w.to_numpy()
    if w is None:
        w = np.ones(X.shape[0])
    if isinstance(w, float):
        w = np.ones(X.shape[0]) * w
    return X, y, w
