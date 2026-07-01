"""Binned response plot for the numerical illustration pipeline.

Produces:
- A PNG figure (matplotlib)
- Per-subplot ``.dat`` data files (space-separated ``bin_index value``)
- A ``.tex`` file that assembles those data files into a pgfplots groupplot,
  matching the style of the existing ``src/figures/beta_prime_sim.tex`` in the
  paper repository.

Usage example::

    from numerical_illustration.plots import BinnedResponsePlotConfig, create_binned_response_plot

    config = BinnedResponsePlotConfig(
        results_dir="data/results/20260531133858",
        distribution="normal",
    )
    create_binned_response_plot(config)
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cyc_gbm.utils.distributions import initiate_distribution

from .config import BinnedResponsePlotConfig
from .tikz_writer import write_tikz


class _BinData:
    """Computed bin statistics for one model × adjustment variant."""

    def __init__(
        self,
        observed: np.ndarray,
        pred_mean: np.ndarray,
        pred_upper: np.ndarray,
        pred_lower: np.ndarray,
    ) -> None:
        self.observed = observed
        self.pred_mean = pred_mean
        self.pred_upper = pred_upper
        self.pred_lower = pred_lower


def _detect_models(test_data: pd.DataFrame) -> list[str]:
    """Return model names from columns named ``<model>_theta_0``."""
    return [
        col.removesuffix("_theta_0")
        for col in test_data.columns
        if col.endswith("_theta_0") and col != "theta_0"
    ]


def _compute_bins(
    y: np.ndarray,
    pred_mean: np.ndarray,
    pred_var: np.ndarray,
    n_bins: int,
) -> _BinData:
    """Compute per-bin observed average, predicted mean, and ±1 std band.

    Follows the paper's equations (5) and (6) under w=1:

    * Observed bin average:   ``ȳ_j = mean(y_i)``
    * Predicted bin mean:     ``μ̂_j = mean(μ̂_i)``
    * Predicted bin std:      ``σ̂_j = sqrt(mean(σ̂²_i) / n_j)``

    Observations are pre-sorted by predicted mean before calling this function.
    """
    n = len(y)
    observed = np.empty(n_bins)
    p_mean = np.empty(n_bins)
    p_upper = np.empty(n_bins)
    p_lower = np.empty(n_bins)

    for j in range(n_bins):
        lo = math.floor(j * n / n_bins)
        hi = math.floor((j + 1) * n / n_bins)
        idx = slice(lo, hi)

        n_j = hi - lo
        observed[j] = y[idx].mean()
        p_mean[j] = pred_mean[idx].mean()
        sigma_j = math.sqrt(pred_var[idx].mean() / n_j)
        p_upper[j] = p_mean[j] + sigma_j
        p_lower[j] = p_mean[j] - sigma_j

    return _BinData(
        observed=observed,
        pred_mean=p_mean,
        pred_upper=p_upper,
        pred_lower=p_lower,
    )


def _bias_correction_factors(
    pred_mean_train: np.ndarray,
    pred_var_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[float, float]:
    """Compute multiplicative bias correction factors from training data.

    Returns ``(c_mu, c_v)`` where:

    * ``c_mu = mean(y_train) / mean(μ̂_train)``
    * ``c_v  = var(y_train) / mean(σ̂²_train)``

    Corrections are applied directly to the predicted moments (not to z), so
    they are distribution-agnostic.
    """
    c_mu = y_train.mean() / pred_mean_train.mean()
    pred_var_mean = pred_var_train.mean()
    obs_var = y_train.var()
    c_v = obs_var / pred_var_mean if (pred_var_mean > 0 and obs_var > 0) else 1.0
    return c_mu, c_v


def _model_bin_data(
    test_data: pd.DataFrame,
    model: str,
    dist,
    n_bins: int,
    adjust: bool,
    train_data: pd.DataFrame | None = None,
) -> _BinData:
    """Compute _BinData for *model*, optionally with bias adjustment.

    When *adjust* is True, *train_data* must be provided — correction factors
    are estimated from the training set moments and applied to the test set
    moments directly, making the approach distribution-agnostic.
    """
    y = test_data["y"].to_numpy()

    n_dim = dist.n_dim if dist.n_dim is not None else sum(
        1 for col in test_data.columns if col.startswith(f"{model}_theta_")
    )

    theta_cols = [f"{model}_theta_{j}" for j in range(n_dim)]
    z = test_data[theta_cols].to_numpy().T

    pred_mean = dist.moment(z, k=1)
    pred_var = dist.moment(z, k=2)

    if adjust:
        if train_data is None:
            raise ValueError("train_data is required when adjust=True")
        z_train = train_data[theta_cols].to_numpy().T
        y_train = train_data["y"].to_numpy()
        pred_mean_train = dist.moment(z_train, k=1)
        pred_var_train = dist.moment(z_train, k=2)
        c_mu, c_v = _bias_correction_factors(pred_mean_train, pred_var_train, y_train)
        pred_mean = pred_mean * c_mu
        pred_var = pred_var * c_v

    order = np.argsort(pred_mean)
    return _compute_bins(y[order], pred_mean[order], pred_var[order], n_bins)


def _write_dat(path: Path, values: np.ndarray) -> None:
    """Write ``bin_index<space>value`` rows to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{i} {v}" for i, v in enumerate(values)]
    path.write_text("\n".join(lines) + "\n")


def _export_dat_files(
    output_dir: Path,
    model: str,
    bin_data: _BinData,
    suffix: str = "",
) -> dict[str, Path]:
    """Write the four series for one subplot and return their paths."""
    stem = f"{model}{suffix}"
    paths = {
        "mean": output_dir / f"{stem}_mean.dat",
        "observed": output_dir / f"{stem}_observed.dat",
        "band_upper": output_dir / f"{stem}_band_upper.dat",
        "band_lower": output_dir / f"{stem}_band_lower.dat",
    }
    _write_dat(paths["mean"], bin_data.pred_mean)
    _write_dat(paths["observed"], bin_data.observed)
    _write_dat(paths["band_upper"], bin_data.pred_upper)
    _write_dat(paths["band_lower"], bin_data.pred_lower)
    return paths


def _plot_subplot(
    ax: plt.Axes,
    bin_data: _BinData,
    title: str,
    ymin: float,
    ymax: float,
) -> None:
    n_bins = len(bin_data.pred_mean)
    x = np.arange(n_bins)

    ax.fill_between(
        x,
        bin_data.pred_lower,
        bin_data.pred_upper,
        color="gray",
        alpha=0.4,
        linewidth=0,
    )
    ax.plot(x, bin_data.pred_mean, color="black", linewidth=2)
    ax.scatter(x, bin_data.observed, color="black", s=4, zorder=3)
    ax.set_title(title)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, n_bins - 1)


def create_binned_response_plot(config: BinnedResponsePlotConfig) -> None:
    """Generate the binned response plot from a completed pipeline run.

    Writes to *output_dir* (defaults to *results_dir*):

    - ``binned_response_plot.png``  — matplotlib figure
    - ``dat/``                      — one ``.dat`` file per data series
    - ``binned_response_plot.tex``  — pgfplots groupplot ready for ``\\input``

    Args:
        config: Plot configuration.
    """
    results_dir = Path(config.results_dir)
    output_dir = Path(config.output_dir) if config.output_dir else results_dir
    dat_dir = output_dir / "dat"

    test_data = pd.read_csv(results_dir / "test_data.csv")
    train_data = pd.read_csv(results_dir / "train_data.csv")

    for name, data in [("test_data", test_data), ("train_data", train_data)]:
        if not (data["w"] == 1).all():
            raise NotImplementedError(
                f"Binned response plot is only implemented for w=1 but {name} "
                "contains non-unit weights."
            )

    m = len(test_data)

    n_bins = config.n_bins if config.n_bins is not None else math.floor(math.sqrt(m))
    models = config.models if config.models is not None else _detect_models(test_data)
    dist = initiate_distribution(config.distribution)

    unadjusted: dict[str, _BinData] = {
        model: _model_bin_data(test_data, model, dist, n_bins, adjust=False)
        for model in models
    }
    adjusted: dict[str, _BinData] = {}
    if config.bias_adjustment:
        adjusted = {
            model: _model_bin_data(
                test_data, model, dist, n_bins, adjust=True, train_data=train_data
            )
            for model in models
        }

    dat_paths: dict[str, dict[str, dict[str, Path]]] = {}
    for model in models:
        dat_paths[model] = {}
        dat_paths[model]["unadjusted"] = _export_dat_files(dat_dir, model, unadjusted[model])
        if config.bias_adjustment:
            dat_paths[model]["adjusted"] = _export_dat_files(
                dat_dir, model, adjusted[model], suffix="_bias_adjusted"
            )

    all_values = np.concatenate(
        [
            np.concatenate([bd.observed, bd.pred_mean, bd.pred_upper, bd.pred_lower])
            for bd in {**unadjusted, **adjusted}.values()
        ]
    )
    ymin = float(np.nanmin(all_values))
    ymax = float(np.nanmax(all_values))
    margin = 0.05 * (ymax - ymin)
    ymin -= margin
    ymax += margin

    n_rows = 2 if config.bias_adjustment else 1
    n_cols = len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=config.figsize, squeeze=False)

    dist_label = config.distribution.replace("_", " ").title()

    for col, model in enumerate(models):
        model_label = model.upper()
        _plot_subplot(
            axes[0, col],
            unadjusted[model],
            title=f"{model_label}, {dist_label}",
            ymin=ymin,
            ymax=ymax,
        )
        if config.bias_adjustment:
            _plot_subplot(
                axes[1, col],
                adjusted[model],
                title=f"{model_label}, {dist_label}, bias adjusted",
                ymin=ymin,
                ymax=ymax,
            )

    fig.tight_layout()
    png_path = output_dir / "binned_response_plot.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    tex_path = output_dir / "binned_response_plot.tex"
    write_tikz(
        path=tex_path,
        models=models,
        distribution_label=dist_label,
        dat_paths=dat_paths,
        n_bins=n_bins,
        ymin=ymin,
        ymax=ymax,
        bias_adjustment=config.bias_adjustment,
    )


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser(
        description="Generate a binned response plot from a numerical illustration run."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    with open(args.config) as fh:
        raw = yaml.safe_load(fh)

    plot_config = BinnedResponsePlotConfig(**raw)
    create_binned_response_plot(plot_config)
