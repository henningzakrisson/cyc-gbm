"""Enumerations and constants for the numerical illustration pipeline."""

from enum import StrEnum


class ModelClass(StrEnum):
    """Identifier for each model type in the pipeline."""

    GBM = "gbm"
    CGBM = "cgbm"
    NGBOOST = "ngboost"
    CGLM = "cglm"
    INTERCEPT = "intercept"


class DataSource(StrEnum):
    """Identifier for each data source type."""

    SIMULATION = "simulation"
    FILE = "file"
