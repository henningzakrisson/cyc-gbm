"""Output / result storage configuration."""

from pathlib import Path

from pydantic import BaseModel


class DumpingConfig(BaseModel):
    """Configuration for where pipeline results are stored.

    Attributes:
        output_dir: Root directory for result output.  A timestamped
            subdirectory is created under this path for each run.
    """

    output_dir: Path = Path("data/results")
