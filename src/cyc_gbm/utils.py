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

    def append_format_level(self, level_msg):
        """Append the level to the message.

        :param level_msg: The level to append to the message.
        """
        formatter = self.handlers[0].formatter
        format_msg = formatter._fmt.split("[%(message)s]")[0]
        format_msg += f"[{level_msg}][%(message)s]"
        formatter = logging.Formatter(format_msg, datefmt="%Y-%m-%d %H:%M")
        self.handlers[0].setFormatter(formatter)
        if len(self.handlers) > 1:
            self.handlers[1].setFormatter(formatter)

    def remove_format_level(self):
        """Remove one level from the message."""
        formatter = self.handlers[0].formatter
        # Split on the second to last occurence of a bracket
        format_msg = formatter._fmt.rsplit("[", 2)[0]
        format_msg += "[%(message)s]"
        formatter = logging.Formatter(format_msg, datefmt="%Y-%m-%d %H:%M")
        self.handlers[0].setFormatter(formatter)
        if len(self.handlers) > 1:
            self.handlers[1].setFormatter(formatter)
