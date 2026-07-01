"""Configuration model for the binned response plot."""

from pathlib import Path

from pydantic import BaseModel


class BinnedResponsePlotConfig(BaseModel):
    """Configuration for generating a binned response plot.

    Attributes:
        results_dir: Path to a numerical illustration results directory
            (must contain ``test_data.csv``).
        distribution: Name of the parametric distribution used in the run,
            e.g. ``"beta_prime"``, ``"normal"``, ``"gamma"``.  Must match a
            key accepted by ``initiate_distribution``.
        n_bins: Number of equal-sized bins.  Defaults to
            ``floor(sqrt(m))`` where *m* is the test-set size.
        models: Subset of model names to include in the plot, e.g.
            ``["gbm", "cgbm"]``.  Defaults to all models detected from the
            ``*_theta_0`` columns in ``test_data.csv``.  Use this to exclude
            baseline models (``intercept``, ``cglm``) or limit the figure to
            the models of interest.
        bias_adjustment: Whether to add a second row of panels with globally
            bias-adjusted mean and variance predictions.
        output_dir: Directory where output files are written.  Defaults to
            ``results_dir``.
        figsize: Width and height of the matplotlib figure in inches.
    """

    results_dir: Path
    distribution: str
    n_bins: int | None = None
    models: list[str] | None = None
    bias_adjustment: bool = True
    output_dir: Path | None = None
    figsize: tuple[float, float] = (12.0, 8.0)
