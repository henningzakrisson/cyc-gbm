import os
from typing import Dict, List

import numpy as np
import pandas as pd


def save_results(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    loss_results: Dict[str, Dict[str, List[float]]],
    metrics: pd.DataFrame,
    output_path: str,
) -> None:
    train_data.to_csv(os.path.join(output_path, "train_data.csv"), index=False)
    test_data.to_csv(os.path.join(output_path, "test_data.csv"), index=False)
    _save_tuning_results(loss_results, output_path)
    metrics.to_csv(os.path.join(output_path, "metrics.csv"), index=True)


def _save_tuning_results(
    loss_results: Dict[str, Dict[str, List[float]]], output_path: str
) -> None:
    # Create a loss folder
    loss_folder = os.path.join(output_path, "loss")
    os.makedirs(loss_folder, exist_ok=True)
    for model_name, losses in loss_results.items():
        df_loss = pd.DataFrame()
        avg_loss = np.mean(losses["train"], axis=0)
        # Save the loss after both parameter updates as 0 and 1 respectively
        df_loss["train_0"] = avg_loss[:, 0]
        df_loss["train_1"] = avg_loss[:, 1]
        df_loss.to_csv(os.path.join(loss_folder, f"{model_name}_loss.csv"), index=False)
