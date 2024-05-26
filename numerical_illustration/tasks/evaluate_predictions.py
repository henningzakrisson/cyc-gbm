import pandas as pd

from cyc_gbm.utils.distributions import initiate_distribution

from .utils.constants import DISTRIBUTION


def evaluate_predictions(predictions: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Evaluate the predictions.

    Args:
        predictions: predictions from the models
        config: configuration dictionary
    """
    # Evaluate the predictions
    distribution = initiate_distribution(config[DISTRIBUTION])
