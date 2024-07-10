import numpy as np
import pandas as pd


def get_targets_features(train_data: pd.DataFrame) -> np.ndarray:
    features = [
        col
        for col in train_data.columns
        if col not in ["y", "w"] and not col.startswith("theta")
    ]
    X_train = train_data[features].values
    y_train = train_data["y"].values
    w_train = train_data["w"].values
    return X_train, y_train, w_train
