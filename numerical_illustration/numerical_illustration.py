import time
import yaml
import os

OUTPUT_DIR = "data/results"
CONFIG_DIR = "config/demo_config.yaml"

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

    # Load data 


if __name__ == "__main__":
    main()