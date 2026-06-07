import os
import time

import numpy as np
import yaml

from ..schema import NumericalIllustrationConfig


def setup_pipeline_run(
    config_path: str,
) -> tuple[NumericalIllustrationConfig, np.random.Generator, str]:
    """
    Setup the pipeline run.

    Args:
        config_path: path to the configuration file

    Returns:
        Tuple of (validated config, seeded RNG, output path)
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    config = NumericalIllustrationConfig(**raw)

    run_id = _get_run_id()
    output_path = _create_output_path(str(config.output.output_dir), run_id)

    with open(os.path.join(output_path, "config.yaml"), "w") as f:
        yaml.dump(
            config.model_dump(mode="json"),
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    rng = np.random.default_rng(config.data.random_seed)

    return config, rng, output_path


def _get_run_id() -> str:
    """Get a unique run ID.

    Assumes that there is no more than one run per second."""
    return time.strftime("%Y%m%d%H%M%S")


def _create_output_path(output_dir: str, run_id: str) -> str:
    """Create a folder for the current run.

    Args:
        output_dir: output directory
        run_id: run ID

    Returns:
        folder name of existing or created folder
    """
    folder = f"{output_dir}/{run_id}"
    os.makedirs(folder, exist_ok=True)
    return folder
