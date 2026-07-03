import numpy as np
import pandas as pd


def get_targets_features(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Extract features, target and weights from a pipeline DataFrame.

    Returns ``X`` as a :class:`~pandas.DataFrame` so that
    ``pd.CategoricalDtype`` columns are preserved for tree-based models.
    ``y`` and ``w`` are returned as numpy arrays for compatibility with
    the distribution and model APIs.

    Args:
        data: pipeline DataFrame containing feature columns, ``y``, ``w``,
            and optionally ``theta_*`` columns.

    Returns:
        Tuple of ``(X, y, w)`` where ``X`` is a DataFrame and ``y``, ``w``
        are numpy arrays.
    """
    features = [
        col
        for col in data.columns
        if col not in ["y", "w"] and not col.startswith("theta")
    ]
    X = data[features]
    y = data["y"].to_numpy()
    w = data["w"].to_numpy()
    return X, y, w
