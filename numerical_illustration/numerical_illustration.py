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

from joblib import Parallel, delayed
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
from .tasks.utils.constants import N_BOOTSTRAPS, N_JOBS, PARALLEL

DEFAULT_CONFIG_DIR = "numerical_illustration/config/demo_config.yaml"

# Set up a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _run_single_iteration(
    config: dict, rng: np.random.Generator, iteration: int = 1, n_bootstraps: int = 1
):
    """Run one full simulation iteration: load, preprocess, tune, fit, predict, evaluate."""
    logger.info(f"Bootstrap iteration {iteration + 1}/{n_bootstraps}")
    raw_input_data = load_input_data(config=config, rng=rng)
    train_data, test_data = preprocess_input_data(
        config=config, data=raw_input_data, rng=rng
    )
    tuning_results, n_estimators = tune_models(
        config=config, train_data=train_data, rng=rng
    )
    models = fit_models(
        config=config, train_data=train_data, rng=rng, n_estimators=n_estimators
    )
    train_data = predict(models=models, data=train_data)
    test_data = predict(models=models, data=test_data)
    metrics = evaluate_predictions(
        train_data=train_data, test_data=test_data, config=config
    )
    return train_data, test_data, tuning_results, models, metrics.astype(float)


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

    parallel = config.get(PARALLEL, False)
    if parallel and n_bootstraps > 1:
        n_jobs = config.get(N_JOBS, -1)
        logger.info(
            f"Running {n_bootstraps} bootstrap iterations in parallel (n_jobs={n_jobs})"
        )
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_iteration)(
                config=config,
                rng=child_rngs[b],
                iteration=b,
                n_bootstraps=n_bootstraps,
            )
            for b in range(n_bootstraps)
        )
    else:
        results = [
            _run_single_iteration(
                config=config,
                rng=child_rngs[b],
                iteration=b,
                n_bootstraps=n_bootstraps,
            )
            for b in range(n_bootstraps)
        ]
    train_data, test_data, tuning_results, models, _ = results[-1]
    all_metrics = [r[4] for r in results]

    logger.info("All bootstrap iterations complete")

    # Aggregate metrics across bootstrap iterations
    stacked = np.stack([m.values for m in all_metrics], axis=0)
    ref = all_metrics[0]
    metrics_mean = pd.DataFrame(
        np.mean(stacked, axis=0), index=ref.index, columns=ref.columns
    )
    metrics_std = pd.DataFrame(
        np.std(stacked, axis=0), index=ref.index, columns=ref.columns
    )

    # Compute mean rank across bootstrap iterations (excluding "true")
    model_names = [idx for idx in ref.index if idx != "true"]
    mean_rank = pd.DataFrame(
        np.stack(
            [m.loc[model_names].rank().values for m in all_metrics], axis=0
        ).mean(axis=0),
        index=model_names,
        columns=ref.columns,
    )

    # Save results (last iteration's data + aggregated metrics)
    save_results(
        train_data=train_data,
        test_data=test_data,
        loss_results=tuning_results,
        metrics_mean=metrics_mean,
        metrics_std=metrics_std,
        mean_rank=mean_rank,
        models=models,
        output_path=output_path,
    )
    logger.info("Results saved")

    formatted = _format_metrics(metrics_mean, metrics_std)
    logger.info(f"\n{formatted}")
    logger.info(f"Mean rank (models only):\n{mean_rank}")


if __name__ == "__main__":
    main()
