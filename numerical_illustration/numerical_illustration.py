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
from typing import Any

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from .schema import NumericalIllustrationConfig, SimulationConfig
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

DEFAULT_CONFIG_PATH = "numerical_illustration/configs/demo_config.yaml"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _run_single_iteration(
    config: NumericalIllustrationConfig,
    rng: np.random.Generator,
    iteration: int = 1,
    n_bootstraps: int = 1,
    log_prefix: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Run one full simulation iteration: load, preprocess, tune, fit, predict, evaluate."""
    if n_bootstraps > 1:
        log_prefix = f"[bootstrap {iteration + 1}/{n_bootstraps}] "
    logger.info(f"{log_prefix}Starting iteration")
    raw_input_data = load_input_data(data_config=config.data, rng=rng)
    train_data, test_data = preprocess_input_data(
        data_config=config.data, data=raw_input_data, rng=rng
    )
    tuning_results, n_estimators = tune_models(
        config=config, train_data=train_data, rng=rng, log_prefix=log_prefix
    )
    models = fit_models(
        model_configs=config.models,
        data_config=config.data,
        train_data=train_data,
        rng=rng,
        n_estimators=n_estimators,
        log_prefix=log_prefix,
    )
    train_data = predict(models=models, data=train_data)
    test_data = predict(models=models, data=test_data)
    # Build per-model distribution mapping for evaluation
    from cyc_gbm.utils.distributions import initiate_distribution
    model_distributions = {
        mc.name: initiate_distribution(
            config.data.distribution,
            parameterization=getattr(mc, "parameterization", None) or config.data.parameterization,
        )
        for mc in config.models
    }
    metrics = evaluate_predictions(
        train_data=train_data,
        test_data=test_data,
        model_distributions=model_distributions,
        model_names=config.model_names,
        data_config=config.data,
        is_simulation=isinstance(config.data, SimulationConfig),
    )
    return train_data, test_data, tuning_results, models, metrics.astype(float)


def _format_metrics(mean: pd.DataFrame, std: pd.DataFrame) -> pd.DataFrame:
    """Format mean +/- std into a single DataFrame of strings."""
    formatted = mean.copy().astype(object)
    for col in mean.columns:
        for idx in mean.index:
            m = float(mean.at[idx, col])
            s = float(std.at[idx, col])
            formatted.at[idx, col] = f"{m:.4f} ± {s:.4f}"
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the numerical illustration pipeline.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to config YAML (default: %(default)s)",
    )
    args = parser.parse_args()
    logger.info(f"Using config: {args.config}")

    config, rng, output_path = setup_pipeline_run(config_path=args.config)
    logger.info("Setup complete")

    n_bootstraps = config.bootstrap.n_bootstraps
    child_rngs = rng.spawn(n_bootstraps)

    parallel = config.bootstrap.parallel
    if parallel and n_bootstraps > 1:
        n_jobs = config.bootstrap.n_jobs
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

    stacked = np.stack([m.values for m in all_metrics], axis=0)
    ref = all_metrics[0]
    metrics_mean = pd.DataFrame(
        np.mean(stacked, axis=0), index=ref.index, columns=ref.columns
    )
    metrics_std = pd.DataFrame(
        np.std(stacked, axis=0), index=ref.index, columns=ref.columns
    )

    model_names = [idx for idx in ref.index if not idx.startswith("true")]
    mean_rank = pd.DataFrame(
        np.stack(
            [m.loc[model_names].rank().values for m in all_metrics], axis=0
        ).mean(axis=0),
        index=model_names,
        columns=ref.columns,
    )

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
