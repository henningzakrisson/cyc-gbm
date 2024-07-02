import numpy as np

def calculate_progress(
step: int, total_steps: int,
) -> float:
    """Calculate the progress of the simulation.

    :param step: The current step.
    :param total_steps: The total number of steps.
    """
    # Check if progress has been made
    new_progress = np.floor(10 * step / total_steps) / 10
    return new_progress
