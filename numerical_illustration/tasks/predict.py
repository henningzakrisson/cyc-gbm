import numpy as np
import pandas as pd

from cyc_gbm import CyclicalGradientBooster

from .baseline_models import CyclicGeneralizedLinearModel, InterceptModel
from .utils.utils import get_targets_features


def predict(
    models: dict[
        str,
        InterceptModel | CyclicGeneralizedLinearModel | CyclicalGradientBooster,
    ],
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict the response for the test data using the fitted models.

    Args:
        models: dictionary of fitted models
        data: test data
    """
    X_test, _, _ = get_targets_features(data)

    predictions = {
        model_name: models[model_name].predict(X_test) for model_name in models
    }

    # Infer n_dim from the first model's prediction shape rather than
    # relying on theta_* columns which only exist for simulated data.
    first_pred = next(iter(predictions.values()))
    n_dim = first_pred.shape[0]

    for model_name in models:
        for j in range(n_dim):
            data[f"{model_name}_theta_{j}"] = predictions[model_name][j]

    return data
