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

# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helper: detect model names from CSV columns
# ---------------------------------------------------------------------------


def _detect_models(test_data: pd.DataFrame) -> list[str]:
    """Return model names from columns named ``<model>_theta_0``."""
    return [
        col.removesuffix("_theta_0")
        for col in test_data.columns
        if col.endswith("_theta_0") and col != "theta_0"
    ]


# ---------------------------------------------------------------------------
# Helper: compute bin statistics
# ---------------------------------------------------------------------------


def _compute_bins(
    y: np.ndarray,
    w: np.ndarray,
    pred_mean: np.ndarray,
    pred_var: np.ndarray,
    n_bins: int,
) -> _BinData:
    """Compute per-bin observed average, predicted mean, and ±1 std band.

    Follows the paper's equations (5) and (6):

    * Observed bin average:
      ``ȳ_j = sum(y_i) / sum(w_i)``  for ``i`` in bin ``j``.

    * Predicted bin mean:
      ``μ̂_j = sum(w_i * μ̂_i) / sum(w_i)``

    * Predicted bin variance:
      ``σ̂²_j = sum(w_i * σ̂²_i) / sum(w_i)²``

    Observations are pre-sorted by ``pred_mean / w`` before calling this
    function (standardised-exposure ordering).
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

        w_j = w[idx]
        sum_w = w_j.sum()

        observed[j] = y[idx].sum() / sum_w
        p_mean[j] = (w_j * pred_mean[idx]).sum() / sum_w
        sigma_j = math.sqrt((w_j * pred_var[idx]).sum() / sum_w**2)
        p_upper[j] = p_mean[j] + sigma_j
        p_lower[j] = p_mean[j] - sigma_j

    return _BinData(
        observed=observed,
        pred_mean=p_mean,
        pred_upper=p_upper,
        pred_lower=p_lower,
    )


# ---------------------------------------------------------------------------
# Helper: bias-adjust z parameters and recompute moments
# ---------------------------------------------------------------------------


def _bias_adjust(
    z: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    dist,
) -> np.ndarray:
    """Return a copy of *z* with globally bias-adjusted parameters.

    Two-step procedure:

    1. **Mean adjustment**: scale ``exp(z[0])`` so that the population mean of
       predicted means matches the exposure-weighted observed mean.
       ``c_μ = mean(y/w) / mean(μ̂)``  →  ``z[0] += log(c_μ)``

    2. **Variance adjustment**: after the mean adjustment, scale ``exp(z[1])``
       so that the mean predicted variance matches the observed sample variance.
       For distributions where variance decreases with *z[1]* (e.g. Beta Prime,
       Gamma) this means ``z[1] += log(c_v)`` where
       ``c_v = mean(var_after_mean_adj) / sample_var(y/w)``.

       Note: for distributions where higher *z[1]* means higher variance (e.g.
       Normal), the sign is reversed automatically because the moment formula
       handles it.
    """
    z_adj = z.copy()

    # --- step 1: mean ---
    mu_hat = dist.moment(z_adj, k=1)
    obs_mean = (y / w).mean()
    pred_mean = mu_hat.mean()
    c_mu = obs_mean / pred_mean
    z_adj[0] = z_adj[0] + math.log(c_mu)

    # --- step 2: variance ---
    var_adj = dist.moment(z_adj, k=2)
    obs_var = np.var(y / w)
    pred_var_mean = var_adj.mean()
    if pred_var_mean > 0 and obs_var > 0:
        c_v = pred_var_mean / obs_var
        z_adj[1] = z_adj[1] + math.log(c_v)

    return z_adj


# ---------------------------------------------------------------------------
# Helper: sort + bin for one model
# ---------------------------------------------------------------------------


def _model_bin_data(
    test_data: pd.DataFrame,
    model: str,
    dist,
    n_bins: int,
    adjust: bool,
) -> _BinData:
    """Compute _BinData for *model*, optionally with bias adjustment."""
    y = test_data["y"].to_numpy()
    w = test_data["w"].to_numpy()

    # Infer n_dim from the distribution object (authoritative) or fall back to
    # counting the model's own theta columns.
    n_dim = dist.n_dim if dist.n_dim is not None else sum(
        1 for col in test_data.columns if col.startswith(f"{model}_theta_")
    )

    theta_cols = [f"{model}_theta_{j}" for j in range(n_dim)]
    z = test_data[theta_cols].to_numpy().T  # shape (n_dim, n_samples)

    if adjust:
        z = _bias_adjust(z, y, w, dist)

    pred_mean = dist.moment(z, k=1)
    pred_var = dist.moment(z, k=2)

    # Sort by standardised predicted mean (pred_mean / w)
    order = np.argsort(pred_mean / w)
    y_s = y[order]
    w_s = w[order]
    pm_s = pred_mean[order]
    pv_s = pred_var[order]

    return _compute_bins(y_s, w_s, pm_s, pv_s, n_bins)


# ---------------------------------------------------------------------------
# Helper: export .dat files
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helper: matplotlib subplot
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


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

    # --- load data ---
    test_data = pd.read_csv(results_dir / "test_data.csv")
    m = len(test_data)

    # --- resolve options ---
    n_bins = config.n_bins if config.n_bins is not None else math.floor(math.sqrt(m))
    models = config.models if config.models is not None else _detect_models(test_data)
    dist = initiate_distribution(config.distribution)

    # --- compute all bin data ---
    # unadjusted[model] and adjusted[model]
    unadjusted: dict[str, _BinData] = {
        m_: _model_bin_data(test_data, m_, dist, n_bins, adjust=False)
        for m_ in models
    }
    adjusted: dict[str, _BinData] = {}
    if config.bias_adjustment:
        adjusted = {
            m_: _model_bin_data(test_data, m_, dist, n_bins, adjust=True)
            for m_ in models
        }

    # --- export .dat files ---
    dat_paths: dict[str, dict[str, dict[str, Path]]] = {}
    for model in models:
        dat_paths[model] = {}
        dat_paths[model]["unadjusted"] = _export_dat_files(
            dat_dir, model, unadjusted[model]
        )
        if config.bias_adjustment:
            dat_paths[model]["adjusted"] = _export_dat_files(
                dat_dir, model, adjusted[model], suffix="_bias_adjusted"
            )

    # --- global y-axis limits (shared across all panels) ---
    all_values = np.concatenate(
        [
            np.concatenate([bd.observed, bd.pred_mean, bd.pred_upper, bd.pred_lower])
            for bd in {**unadjusted, **adjusted}.values()
        ]
    )
    ymin = float(np.nanmin(all_values))
    ymax = float(np.nanmax(all_values))
    # Add small margin
    margin = 0.05 * (ymax - ymin)
    ymin -= margin
    ymax += margin

    # --- matplotlib figure ---
    n_rows = 2 if config.bias_adjustment else 1
    n_cols = len(models)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=config.figsize, squeeze=False
    )

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

    # --- tikz ---
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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

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

    create_binned_response_plot(BinnedResponsePlotConfig(**raw))
