"""Configuration models for the concentration curve plot."""

from pathlib import Path

from pydantic import BaseModel

from numerical_illustration.schema.constants import ModelClass

# Default model colours matching the paper's concentration_curves.tex.
DEFAULT_MODEL_COLORS: dict[str, str] = {
    ModelClass.GBM: "#B0B0B0",       # lightgray (RGB 176,176,176)
    ModelClass.CGBM: "#333333",      # darkslategray (RGB 51,51,51)
    ModelClass.NGBOOST: "#1f77b4",   # tab:blue
    ModelClass.CGLM: "#ff7f0e",      # tab:orange
    ModelClass.INTERCEPT: "#2ca02c", # tab:green
}

# Corresponding TikZ colour definitions (name → RGB triple).
DEFAULT_TIKZ_COLORS: dict[str, tuple[int, int, int]] = {
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
            concentration curve.  Defaults to ``"gbm"``.
        n_points: Number of evenly spaced α-points on the [0, 1] grid.
        model_colors: Matplotlib colour per model name.  Missing entries
            fall back to ``DEFAULT_MODEL_COLORS`` or a built-in palette.
        tikz_colors: TikZ RGB triple per model name.  Missing entries fall
            back to ``DEFAULT_TIKZ_COLORS``.
        output_dir: Directory where output files are written.  Defaults to
            the first dataset's ``results_dir``.
        figsize: Width and height of the matplotlib figure in inches.
    """

    datasets: list[DatasetConfig]
    models: list[str] | None = None
    mean_model_for_residuals: str = ModelClass.GBM
    n_points: int = 100
    model_colors: dict[str, str] | None = None
    tikz_colors: dict[str, tuple[int, int, int]] | None = None
    output_dir: Path | None = None
    figsize: tuple[float, float] = (12.0, 5.0)
