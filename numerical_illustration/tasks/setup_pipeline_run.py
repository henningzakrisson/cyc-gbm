import os
import shutil
import time
from typing import Any, Dict, Tuple

import numpy as np
import yaml

from .utils.constants import OUTPUT_DIR, RANDOM_SEED


def setup_pipeline_run(config_path: str) -> Tuple[Dict[str, Any], np.random.Generator]:
    """
    Setup the pipeline run.

    Args:
        config_path: path to the configuration file

    Returns:
        configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    run_id = _get_run_id()
    output_folder = _create_output_folder(config[OUTPUT_DIR], run_id)
    shutil.copy(config_path, f"{output_folder}/config.yaml")

    random_seed = config[RANDOM_SEED] if RANDOM_SEED in config else 1
    rng = np.random.default_rng(random_seed)

    return config, rng


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
