import logging
from typing import Union
import os


class SimulationLogger:
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
        self.verbose = verbose
        self.logger = logging.Logger("simulation_logger")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        formatter = logging.Formatter(
            f"[%(asctime)s][run_{run_id}][%(message)s]", datefmt="%Y-%m-%d %H:%M"
        )
        self.logger.handlers[0].setFormatter(formatter)

        if output_path is not None:
            log_file = os.path.join(f"{output_path}/run_{run_id}", "log.txt")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.handlers[1].setFormatter(formatter)

    def log(self, msg: str, verbose: int = 0):
        if verbose <= self.verbose:
            self.logger.info(msg)

    def append_format_level(self, level_msg):
        """Append the level to the message.

        :param level_msg: The level to append to the message.
        """
        formatter = self.logger.handlers[0].formatter
        format_msg = formatter._fmt.split("[%(message)s]")[0]
        format_msg += f"[{level_msg}][%(message)s]"
        formatter = logging.Formatter(format_msg, datefmt="%Y-%m-%d %H:%M")
        self.logger.handlers[0].setFormatter(formatter)
        if len(self.logger.handlers) > 1:
            self.logger.handlers[1].setFormatter(formatter)

    def remove_format_level(self):
        """Remove one level from the message."""
        formatter = self.logger.handlers[0].formatter
        format_msg = formatter._fmt.rsplit("[", 2)[0]
        format_msg += "[%(message)s]"
        formatter = logging.Formatter(format_msg, datefmt="%Y-%m-%d %H:%M")
        self.logger.handlers[0].setFormatter(formatter)
        if len(self.logger.handlers) > 1:
            self.logger.handlers[1].setFormatter(formatter)
