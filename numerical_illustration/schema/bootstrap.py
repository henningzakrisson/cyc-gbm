"""Bootstrap configuration."""

from pydantic import BaseModel, field_validator


class BootstrapConfig(BaseModel):
    """Configuration for repeated bootstrap evaluation runs.

    Attributes:
        n_bootstraps: Number of bootstrap iterations to run.
        parallel: Whether to run bootstrap iterations in parallel via joblib.
        n_jobs: Number of parallel workers.  Must be a strictly positive
            integer or ``-1`` (use all available cores).
    """

    n_bootstraps: int = 1
    parallel: bool = False
    n_jobs: int = -1

    @field_validator("n_jobs")
    @classmethod
    def n_jobs_must_be_positive_or_minus_one(cls, v: int) -> int:
        if v != -1 and v < 1:
            raise ValueError(
                f"n_jobs must be a positive integer or -1, got {v}"
            )
        return v
