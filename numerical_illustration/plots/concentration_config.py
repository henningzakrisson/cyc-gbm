"""Configuration models for the concentration curve plot."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

from numerical_illustration.schema.constants import ModelClass

HexColor = Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")]
RGBTriple = tuple[
    Annotated[int, Field(ge=0, le=255)],
    Annotated[int, Field(ge=0, le=255)],
    Annotated[int, Field(ge=0, le=255)],
]

DEFAULT_MODEL_COLORS: dict[ModelClass, HexColor] = {
    ModelClass.GBM: "#B0B0B0",
    ModelClass.CGBM: "#333333",
    ModelClass.NGBOOST: "#1f77b4",
    ModelClass.CGLM: "#ff7f0e",
    ModelClass.INTERCEPT: "#2ca02c",
}

DEFAULT_TIKZ_COLORS: dict[ModelClass, RGBTriple] = {
    ModelClass.GBM: (176, 176, 176),
    ModelClass.CGBM: (51, 51, 51),
    ModelClass.NGBOOST: (31, 119, 180),
    ModelClass.CGLM: (255, 127, 14),
    ModelClass.INTERCEPT: (44, 160, 44),
}


class DatasetConfig(BaseModel):
    """Description of a single dataset (one row in the concentration curve grid).

    Attributes:
        results_dir: Path to a numerical illustration results directory
            (must contain ``test_data.csv``).
        distribution: Name of the parametric distribution used in the run,
            e.g. ``"normal"``, ``"gamma"``, ``"neg_bin"``.
        label: Human-readable label for this dataset, used in subplot titles
            (e.g. ``"Number of claims"``, ``"Payment size"``).
    """

    results_dir: Path
    distribution: str
    label: str


class ConcentrationCurvePlotConfig(BaseModel):
    """Configuration for generating concentration curve plots.

    The resulting figure has ``len(datasets)`` rows and 2 columns (expected
    value / variance).

    Attributes:
        datasets: One entry per dataset row in the figure.
        models: Subset of model names to include.  Defaults to all models
            detected from the ``*_theta_0`` columns in the first dataset's
            ``test_data.csv``.
        mean_model_for_residuals: Which model's mean predictions to use when
            computing the squared-residual response for the variance
            concentration curve.  Defaults to ``ModelClass.GBM``.
        n_points: Number of evenly spaced α-points on the [0, 1] grid.
        model_colors: Matplotlib hex colour per model.  Missing entries
            fall back to ``DEFAULT_MODEL_COLORS`` or a built-in palette.
        tikz_colors: TikZ RGB triple per model.  Missing entries fall
            back to ``DEFAULT_TIKZ_COLORS``.
        output_dir: Directory where output files are written.  Defaults to
            the first dataset's ``results_dir``.
        figsize: Width and height of the matplotlib figure in inches.
    """

    datasets: list[DatasetConfig]
    models: list[ModelClass] | None = None
    mean_model_for_residuals: ModelClass = ModelClass.GBM
    n_points: int = 100
    model_colors: dict[ModelClass, HexColor] | None = None
    tikz_colors: dict[ModelClass, RGBTriple] | None = None
    output_dir: Path | None = None
    figsize: tuple[float, float] = (12.0, 5.0)
