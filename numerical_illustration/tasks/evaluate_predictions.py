from collections import deque

import numpy as np
import pandas as pd

from cyc_gbm.utils.distributions import Distribution, initiate_distribution

from ..schema.data import DataConfig


def _convert_gamma_theta(
    theta: np.ndarray,
    from_parameterization: str,
    to_parameterization: str,
) -> np.ndarray:
    """Convert oracle gamma parameters between parametrizations.

    Supported conversions:
        shape-rate  → mean-dispersion:
            log(mu)  = log(alpha) - log(beta)  = theta[0] - theta[1]
            log(phi) = -log(alpha)             = -theta[0]
        mean-dispersion → shape-rate:
            log(alpha) = -log(phi)             = -theta[1]
            log(beta)  = -log(phi) - log(mu)   = -theta[1] - theta[0]
    """
    if from_parameterization == to_parameterization:
        return theta
    if from_parameterization == "shape-rate" and to_parameterization == "mean-dispersion":
        return np.stack([theta[0] - theta[1], -theta[0]])
    if from_parameterization == "mean-dispersion" and to_parameterization == "shape-rate":
        return np.stack([-theta[1], -theta[1] - theta[0]])
    raise ValueError(
        f"Unsupported gamma theta conversion: {from_parameterization!r} → {to_parameterization!r}"
    )


def _get_unique_parameterizations(model_distributions: dict[str, Distribution]) -> list[str]:
    """Return the unique parametrization names present among all model distributions."""
    seen: list[str] = []
    for dist in model_distributions.values():
        name = _distribution_parameterization(dist)
        if name not in seen:
            seen.append(name)
    return seen


def _distribution_parameterization(dist: Distribution) -> str:
    """Infer the parametrization name from a Distribution instance."""
    from cyc_gbm.utils.distributions import GammaShapeRateDistribution
    if isinstance(dist, GammaShapeRateDistribution):
        return "shape-rate"
    return "mean-dispersion"


def evaluate_predictions(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model_distributions: dict[str, Distribution],
    model_names: list[str],
    data_config: DataConfig,
    is_simulation: bool,
) -> pd.DataFrame:
    """Evaluate the predictions.

    Args:
        train_data: training data with prediction columns
        test_data: test data with prediction columns
        model_distributions: mapping from model name to its fitted Distribution
        model_names: ordered list of model names
        data_config: data configuration (provides simulation parametrization and
            distribution name for constructing oracle distributions)
        is_simulation: whether the data was generated via simulation
            (if so, include oracle parameter losses in the metrics)
    """
    model_rows = list(model_names)

    # Build "true" rows — one per unique parametrization among the fitted models.
    true_rows: list[tuple[str, Distribution]] = []
    if is_simulation:
        unique_params = _get_unique_parameterizations(model_distributions)
        sim_param = data_config.parameterization
        for param in unique_params:
            suffix = "sr" if param == "shape-rate" else "md"
            row_name = f"true_{suffix}" if len(unique_params) > 1 else "true"
            dist = initiate_distribution(data_config.distribution, parameterization=param)
            true_rows.append((row_name, dist, param))

    all_rows = deque([r[0] for r in true_rows] + model_rows)
    metrics = pd.DataFrame(columns=["train", "test"], index=all_rows)

    sim_param = data_config.parameterization

    for data_set, data_name in zip([train_data, test_data], metrics.columns):
        y = data_set["y"].values
        w = data_set["w"].values

        # Evaluate oracle rows
        for row_name, dist, param in true_rows:
            n_dim = dist.n_dim
            theta_cols = ["theta_" + str(i) for i in range(n_dim)]
            z = data_set[theta_cols].values.T

            # Convert oracle thetas from simulation space to this parametrization
            if data_config.distribution == "gamma" and param != sim_param:
                z = _convert_gamma_theta(z, from_parameterization=sim_param, to_parameterization=param)

            metrics.at[row_name, data_name] = dist.loss(y=y, z=z, w=w).mean()

        # Evaluate model rows
        for model_name in model_rows:
            dist = model_distributions[model_name]
            theta_cols = [
                col
                for col in data_set.columns
                if col.startswith(model_name + "_theta_")
            ]
            z = data_set[theta_cols].values.T
            metrics.at[model_name, data_name] = dist.loss(y=y, z=z, w=w).mean()

    return metrics
