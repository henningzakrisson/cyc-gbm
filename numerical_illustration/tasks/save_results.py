import os
from typing import Dict, List

import numpy as np
import pandas as pd

from cyc_gbm import CyclicalGradientBooster


def save_results(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    loss_results: Dict[str, Dict[str, List[float]]],
    metrics_mean: pd.DataFrame,
    metrics_std: pd.DataFrame,
    mean_rank: pd.DataFrame,
    models: Dict[str, CyclicalGradientBooster],
    output_path: str,
) -> None:
    train_data.to_csv(os.path.join(output_path, "train_data.csv"), index=False)
    test_data.to_csv(os.path.join(output_path, "test_data.csv"), index=False)
    _save_tuning_results(loss_results, output_path)
    _save_metrics(metrics_mean, metrics_std, mean_rank, output_path)
    gbm_models = {
        k: models[k]
        for k in ["cgbm", "gbm"]
        if k in models
    }
    _save_feature_importances(models=gbm_models, output_path=output_path)


def _save_metrics(
    metrics_mean: pd.DataFrame,
    metrics_std: pd.DataFrame,
    mean_rank: pd.DataFrame,
    output_path: str,
) -> None:
    """Save aggregated metrics with separate mean and std columns."""
    combined = pd.DataFrame(index=metrics_mean.index)
    for col in metrics_mean.columns:
        combined[f"{col}_mean"] = metrics_mean[col]
        combined[f"{col}_std"] = metrics_std[col]
    for col in mean_rank.columns:
        combined[f"{col}_mean_rank"] = mean_rank[col]
    combined.to_csv(os.path.join(output_path, "metrics.csv"), index=True)

def _save_tuning_results(
    loss_results: Dict[str, Dict[str, List[float]]], output_path: str
) -> None:
    if not loss_results:
        return
    loss_folder = os.path.join(output_path, "loss")
    os.makedirs(loss_folder, exist_ok=True)
    for dataset in ["train", "valid"]:
        dataset_folder = os.path.join(loss_folder, dataset)
        os.makedirs(dataset_folder, exist_ok=True)
        for model_name, losses in loss_results.items():
            df_loss = pd.DataFrame()
            avg_loss = np.mean(losses[dataset], axis=0)
            # Save the loss after both parameter updates as 0 and 1 respectively
            df_loss[f"{dataset}_0"] = avg_loss[:, 0]
            df_loss[f"{dataset}_1"] = avg_loss[:, 1]
            df_loss.to_csv(os.path.join(dataset_folder, f"{model_name}_loss.csv"), index=True)

def _save_feature_importances(models: Dict[str, CyclicalGradientBooster], output_path: str) -> None:
    feature_importances_folder = os.path.join(output_path, "feature_importances")
    os.makedirs(feature_importances_folder)
    for model in models:
        pd.DataFrame(
            {
            j: models[model].compute_feature_importances(j = j)
            for j in [0,1,"all"]
            }
        ).to_csv(f"{feature_importances_folder}/{model}_feature_importances.csv", index=True)
