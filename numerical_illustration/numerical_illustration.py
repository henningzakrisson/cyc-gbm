import os
import shutil
import time

import numpy as np
import pandas as pd
import yaml
from tasks import (
    fit_models,
    load_input_data,
    predict,
    preprocess_input_data,
    setup_pipeline_run,
)

CONFIG_DIR = "numerical_illustration/config/demo_config.yaml"
RANDOM_SEED = "random_seed"


def main():
    # Setup the numerical illustration
    config, rng = setup_pipeline_run(config_path=CONFIG_DIR)

    # Load data
    raw_input_data = load_input_data(config=config, rng=rng)

    # Preprocess data
    train_data, test_data = preprocess_input_data(
        config=config, data=raw_input_data, rng=rng
    )

    # Fit models
    models = fit_models(config=config, train_data=train_data, rng=rng)

    # Predict
    predictions = predict(models=models, test_data=test_data)


if __name__ == "__main__":
    main()
