"""Numerical illustration pipeline for CycGBM.

Runs the full pipeline: data loading, preprocessing, tuning, fitting,
prediction, evaluation, and saving results. Supports repeated bootstrap
runs to estimate mean and standard deviation of losses.

Usage:
    python numerical_illustration
    python -m numerical_illustration
    python numerical_illustration/numerical_illustration.py --config path/to/config.yaml
"""

import argparse
import logging

import numpy as np
import pandas as pd

from .tasks import (
    evaluate_predictions,
    fit_models,
    load_input_data,
    predict,
    preprocess_input_data,
    save_results,
    setup_pipeline_run,
    tune_models,
)
from .tasks.utils.constants import N_BOOTSTRAPS

DEFAULT_CONFIG_DIR = "numerical_illustration/config/demo_config.yaml"

# Set up a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _format_metrics(mean: pd.DataFrame, std: pd.DataFrame) -> pd.DataFrame:
    """Format mean ± std into a single DataFrame of strings."""
    formatted = mean.copy().astype(object)
    for col in mean.columns:
        for idx in mean.index:
            m = float(mean.at[idx, col])
            s = float(std.at[idx, col])
            formatted.at[idx, col] = f"{m:.4f} ± {s:.4f}"
    return formatted


def main():
    parser = argparse.ArgumentParser(description="Run the numerical illustration pipeline.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_DIR,
        help="Path to config YAML (default: %(default)s)",
    )
    args = parser.parse_args()
    logger.info(f"Using config: {args.config}")

    # Setup the numerical illustration
    config, rng, output_path = setup_pipeline_run(config_path=args.config)
    logger.info("Setup complete")

    n_bootstraps = config.get(N_BOOTSTRAPS, 1)
    child_rngs = rng.spawn(n_bootstraps)

    all_metrics: list[pd.DataFrame] = []

    for b in range(n_bootstraps):
        logger.info(f"Bootstrap iteration {b + 1}/{n_bootstraps}")
        b_rng = child_rngs[b]

        # Load data
        raw_input_data = load_input_data(config=config, rng=b_rng)

        # Preprocess data
        train_data, test_data = preprocess_input_data(
            config=config, data=raw_input_data, rng=b_rng
        )

        # Tune models
        tuning_results, n_estimators = tune_models(
            config=config,
            train_data=train_data,
            rng=b_rng,
        )

        # Fit models
        models = fit_models(
            config=config, train_data=train_data, rng=b_rng, n_estimators=n_estimators
        )

        # Predict
        train_data = predict(models=models, data=train_data)
        test_data = predict(models=models, data=test_data)

        # Evaluate
        metrics = evaluate_predictions(
            train_data=train_data, test_data=test_data, config=config
        )
        all_metrics.append(metrics.astype(float))

    logger.info("All bootstrap iterations complete")

    # Aggregate metrics across bootstrap iterations
    stacked = np.stack([m.values for m in all_metrics], axis=0)
    mean_values = np.mean(stacked, axis=0)
    std_values = np.std(stacked, axis=0)

    ref = all_metrics[0]
    metrics_mean = pd.DataFrame(mean_values, index=ref.index, columns=ref.columns)
    metrics_std = pd.DataFrame(std_values, index=ref.index, columns=ref.columns)

    # Save results (last iteration's data + aggregated metrics)
    save_results(
        train_data=train_data,
        test_data=test_data,
        loss_results=tuning_results,
        metrics_mean=metrics_mean,
        metrics_std=metrics_std,
        models=models,
        output_path=output_path,
    )
    logger.info("Results saved")

    formatted = _format_metrics(metrics_mean, metrics_std)
    logger.info(f"\n{formatted}")


if __name__ == "__main__":
    main()
