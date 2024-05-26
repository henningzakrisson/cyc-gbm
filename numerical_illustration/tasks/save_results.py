import os
import pandas as pd
import matplotlib.pyplot as plt

from .utils.constants import OUTPUT_DIR
def save_results(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        metrics: pd.DataFrame,
        figure: plt.Figure,
        output_path: str,
) -> None:
    train_data.to_csv(os.path.join(output_path, "train_data.csv"), index=False)
    test_data.to_csv(os.path.join(output_path, "test_data.csv"), index=False)
    metrics.to_csv(os.path.join(output_path, "metrics.csv"), index=True)
    
    figure.savefig(os.path.join(output_path, "predictions.png"))    