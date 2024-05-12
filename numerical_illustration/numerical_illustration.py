import os
import shutil
import time

import numpy as np
import pandas as pd
import yaml
from tasks.load_input_data import load_input_data
from tasks.preprocess_input_data import preprocess_input_data

OUTPUT_DIR = "data/results"
CONFIG_DIR = "numerical_illustration/config/demo_config.yaml"
RANDOM_SEED = "random_seed"


def _get_run_id() -> str:
    """
    Get a unique run ID.

    Assumes that there is no more than one run per second."""
    return time.strftime("%Y%m%d%H%M%S")


def _create_output_folder(output_dir: str, run_id: str) -> str:
    """
    Create a folder for the current run.

    Args:
        output_dir: output directory
        run_id: run ID

    Returns:
        folder name of existing or created folder
    """
    folder = f"{output_dir}/{run_id}"
    os.makedirs(folder, exist_ok=True)
    return folder


def main():
    # Setup the run and output
    run_id = _get_run_id()
    output_folder = _create_output_folder(OUTPUT_DIR, run_id)

    # Load the configuration
    with open(CONFIG_DIR, "r") as f:
        config = yaml.safe_load(f)
        # Save it to the output folder

    # Copy configuration to output folder
    shutil.copy(CONFIG_DIR, f"{output_folder}/config.yaml")

    # Setup the random number generator
    random_seed = config[RANDOM_SEED] if RANDOM_SEED in config else 1
    rng = np.random.default_rng(random_seed)
    # Load data
    raw_input_data = load_input_data(config=config, rng=rng)

    # Preprocess the data
    train_data, test_data = preprocess_input_data(
        config=config, data=raw_input_data, rng=rng
    )

    # Save the train data
    train_data.to_csv(f"{output_folder}/train_data.csv", index=False)


if __name__ == "__main__":
    main()
