from typing import Dict, Union

import numpy as np
import pandas as pd

from cyc_gbm import CyclicalGradientBooster

from .baseline_models import CyclicGeneralizedLinearModel, InterceptModel
from .fit_models import _get_targets_features


def predict(
    models: Dict[
        str,
        Union[InterceptModel, CyclicGeneralizedLinearModel, CyclicalGradientBooster],
    ],
    test_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict the response for the test data using the fitted models.

    Args:
        models: dictionary of fitted models
        test_data: test data
    """
    X_test, y_test, w_test = _get_targets_features(test_data)

    predictions = {
        model_name: models[model_name].predict(X_test) for model_name in models
    }

    features = [
        col
        for col in test_data.columns
        if (col not in ["y", "w"] or col.startswith("theta"))
    ]
    n_dim = len([col for col in test_data.columns if col.startswith("theta")])
    result_data = test_data.drop(columns=features)
    for model_name in models:
        for j in range(n_dim):
            result_data[f"{model_name}_theta_{j}"] = predictions[model_name][j]

    return result_data
