"""Tuning configuration."""

from pydantic import BaseModel


class TuningConfig(BaseModel):
    """Configuration for cross-validated hyperparameter tuning.

    When ``perform_tuning`` is ``True``, the pipeline runs k-fold CV to
    select optimal ``n_estimators`` for tree-based models.  Other
    hyperparameters are taken as-is from each model config.

    Attributes:
        perform_tuning: Whether to run CV-based tuning of ``n_estimators``.
        n_folds: Number of CV folds.
        stratify: Whether to use sort-and-deal stratification when splitting folds.
        stratify_by_weights: If True, stratify on y/w instead of y.
    """

    perform_tuning: bool = False
    n_folds: int = 4
    stratify: bool = False
    stratify_by_weights: bool = False
