import os

import matplotlib.pyplot as plt
import pandas as pd


def save_results(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    metrics: pd.DataFrame,
    output_path: str,
) -> None:
    train_data.to_csv(os.path.join(output_path, "train_data.csv"), index=False)
    test_data.to_csv(os.path.join(output_path, "test_data.csv"), index=False)
    metrics.to_csv(os.path.join(output_path, "metrics.csv"), index=True)
