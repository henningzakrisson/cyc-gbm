import numpy as np
import pandas as pd

NORMALIZE = "normalize"
TEST_SIZE = "test_size"
OUTPUT_DIR = "output_dir"


def preprocess_input_data(
    config: dict, data: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Preprocess the input data.

    Args:
        config: configuration dictionary
        data: input data
        rng: random number generator
    """
    features = [
        col for col in data.columns if col not in ["y", "w"] or col.startswith("theta")
    ]

    if config[NORMALIZE]:
        data[features] = _normalize_data(data[features])

    train_data, test_data = _split_data(data, config[TEST_SIZE], rng)

    _save_train_data(train_data, config[OUTPUT_DIR])

    return train_data, test_data


def _normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the data.

    Args:
        data: input data
    """
    return (data - data.mean()) / data.std()


def _split_data(
    data: pd.DataFrame, test_size: float, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Split the data into training and test data.

    Args:
        data: input data
        test_size: size of the test data
        rng: random number generator
    """
    n_test = int(test_size * data.shape[0])
    test_indices = rng.choice(data.index, size=n_test, replace=False)
    test_data = data.loc[test_indices]
    train_data = data.drop(test_indices)
    return train_data, test_data


def _save_train_data(train_data: pd.DataFrame, output_folder: str) -> None:
    """
    Save the training data to a CSV file.

    Args:
        train_data: training data
        output_folder: output folder
    """
    train_data.to_csv(f"{output_folder}/train_data.csv", index=False)
