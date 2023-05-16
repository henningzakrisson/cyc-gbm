import logging
from typing import Union
import os


class SimulationLogger(logging.Logger):
    """Logger for the simulation study."""

    def __init__(
        self,
        run_id: int = 0,
        verbose: int = 0,
        output_path: Union[str, None] = None,
    ):
        """Initialize the logger.

        :param run_id: The id of the run.
        :param verbose: The verbosity level.
        :param output_path: The path to the output directory.
        """
        super().__init__("simulation_logger")
        self.verbose = verbose
        self.setLevel(logging.INFO)
        self.addHandler(logging.StreamHandler())
        formatter = logging.Formatter(
            f"[%(asctime)s][run_{run_id}][%(message)s]", datefmt="%Y-%m-%d %H:%M"
        )
        self.handlers[0].setFormatter(formatter)

        if output_path is not None:
            log_file = os.path.join(f"{output_path}/run_{run_id}", "log.txt")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.addHandler(file_handler)
            self.handlers[1].setFormatter(formatter)

    def log_info(self, msg: str, verbose: int = 0):
        if verbose <= self.verbose:
            self.info(msg)
