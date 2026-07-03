"""Concentration curve plot for the numerical illustration pipeline.

Produces:
- A PNG figure (matplotlib)
- Per-curve ``.dat`` data files (space-separated ``alpha value``)
- A ``.tex`` file that assembles the data into a pgfplots groupplot,
  matching the style of the paper's ``src/figures/concentration_curves.tex``.

The concentration curve is defined following Eq. (3.1) in Denuit et al.:

    CC(Y, μ̂(X); α) = E[Y · 1{μ̂(X) ≤ F⁻¹_μ̂(α)}] / E[Y],   α ∈ [0, 1].

For variance assessment, Y is replaced with squared residuals centred on a
configurable reference model's mean predictions, and the ordering is by the
evaluated model's predicted variance.

Usage example::

    from numerical_illustration.plots import (
        ConcentrationCurvePlotConfig,
        create_concentration_curve_plot,
    )
    from numerical_illustration.plots.concentration_config import DatasetConfig

    config = ConcentrationCurvePlotConfig(
        datasets=[
            DatasetConfig(
                results_dir="data/results/20260531133858",
                distribution="gamma",
                label="Simulated Gamma",
            ),
        ],
    )
    create_concentration_curve_plot(config)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cyc_gbm.utils.distributions import initiate_distribution

from .concentration_config import (
    DEFAULT_MODEL_COLORS,
    ConcentrationCurvePlotConfig,
)
from .concentration_tikz_writer import write_concentration_tikz

_NON_NEGATIVE_DISTRIBUTIONS: set[str] = {
    "gamma",
    "neg_bin",
    "beta_prime",
    "inverse_gaussian",
}


def _check_distribution_support(distribution: str) -> None:
    """Raise if *distribution* can produce negative values."""
    if distribution not in _NON_NEGATIVE_DISTRIBUTIONS:
        raise NotImplementedError(
            f"Concentration curves are only implemented for distributions "
            f"with non-negative support ({', '.join(sorted(_NON_NEGATIVE_DISTRIBUTIONS))}). "
            f"Got distribution={distribution!r}, which can produce negative "
            f"values and would lead to meaningless concentration curves."
        )


def _detect_models(data: pd.DataFrame) -> list[str]:
    """Return model names from columns named ``<model>_theta_0``."""
    return [
        col.removesuffix("_theta_0")
        for col in data.columns
        if col.endswith("_theta_0") and col != "theta_0"
    ]


def _get_model_color(model: str, overrides: dict[str, str] | None) -> str:
    """Resolve a matplotlib colour string for *model*."""
    if overrides and model in overrides:
        return overrides[model]
    if model in DEFAULT_MODEL_COLORS:
        return DEFAULT_MODEL_COLORS[model]
    fallback_cycle = [
        "#d62728", "#9467bd", "#8c564b", "#e377c2",
        "#7f7f7f", "#bcbd22", "#17becf",
    ]
    idx = hash(model) % len(fallback_cycle)
    return fallback_cycle[idx]


def _compute_concentration_curve(
    response: np.ndarray,
    sort_values: np.ndarray,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an empirical concentration curve.

    Args:
        response: The variable being accumulated (y for mean CC,
            squared residuals for variance CC).
        sort_values: Values used to order observations (predicted mean
            for mean CC, predicted variance for variance CC).
        n_points: Number of evenly spaced α-points on [0, 1].

    Returns:
        ``(alphas, cc)`` where ``alphas`` has shape ``(n_points + 1,)``
        ranging from 0 to 1 and ``cc`` is the corresponding concentration
        curve values.
    """
    order = np.argsort(sort_values)
    response_sorted = response[order]
    total = response_sorted.sum()

    cumsum = np.concatenate([[0.0], np.cumsum(response_sorted)])
    n = len(response_sorted)
    fracs = np.arange(n + 1) / n

    alphas = np.linspace(0.0, 1.0, n_points + 1)
    cc = np.interp(alphas, fracs, cumsum / total if total != 0 else cumsum)

    return alphas, cc


def _model_moments(
    data: pd.DataFrame,
    model: str,
    dist,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pred_mean, pred_var)`` for *model* from *data*."""
    n_dim = dist.n_dim if dist.n_dim is not None else sum(
        1 for col in data.columns if col.startswith(f"{model}_theta_")
    )
    theta_cols = [f"{model}_theta_{j}" for j in range(n_dim)]
    z = data[theta_cols].to_numpy().T
    return dist.moment(z, k=1), dist.moment(z, k=2)


def _compute_mean_cc(
    test_data: pd.DataFrame,
    model: str,
    dist,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Concentration curve based on mean predictions sorting, accumulating y."""
    y = test_data["y"].to_numpy()
    pred_mean, _ = _model_moments(test_data, model, dist)
    return _compute_concentration_curve(y, pred_mean, n_points)


def _compute_variance_cc(
    test_data: pd.DataFrame,
    model: str,
    dist,
    mean_model: str,
    mean_dist,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Concentration curve for variance predictions.

    Squared residuals are centred on *mean_model*'s mean predictions.
    Sorting is by the current *model*'s predicted variance.

    When exposure weights ``w`` are present, the conditional mean is
    ``w · μ̂(x)`` and the conditional variance is ``w · σ̂²(x)``.  The
    squared residual for each observation is ``(y_i − w_i μ̂_ref_i)²``.
    """
    y = test_data["y"].to_numpy()
    w = test_data["w"].to_numpy()

    ref_mean, _ = _model_moments(test_data, mean_model, mean_dist)
    squared_residuals = (y - w * ref_mean) ** 2

    _, pred_var = _model_moments(test_data, model, dist)
    sort_values = w * pred_var

    return _compute_concentration_curve(squared_residuals, sort_values, n_points)


def _write_dat(path: Path, alphas: np.ndarray, values: np.ndarray) -> None:
    """Write ``alpha<space>value`` rows to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{a} {v}" for a, v in zip(alphas, values)]
    path.write_text("\n".join(lines) + "\n")


def _plot_cc_subplot(
    ax: plt.Axes,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    model_colors: dict[str, str],
    show_legend: bool,
) -> None:
    """Render one concentration-curve panel."""
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)

    for model, (alphas, cc) in curves.items():
        color = model_colors.get(model, "black")
        ax.plot(alphas, cc, color=color, linewidth=1.5, label=model.upper())

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_aspect("equal")

    if show_legend:
        ax.legend(
            loc="upper left",
            framealpha=0.8,
            edgecolor="#cccccc",
        )


def create_concentration_curve_plot(config: ConcentrationCurvePlotConfig) -> None:
    """Generate concentration curve plots from completed pipeline runs.

    Writes to *output_dir* (defaults to the first dataset's ``results_dir``):

    - ``concentration_curve_plot.png``  — matplotlib figure
    - ``dat/``                          — one ``.dat`` file per curve
    - ``concentration_curve_plot.tex``  — pgfplots groupplot ready for
      ``\\input``

    Args:
        config: Plot configuration.
    """
    datasets = config.datasets

    for ds in datasets:
        _check_distribution_support(ds.distribution)

    output_dir = (
        Path(config.output_dir)
        if config.output_dir
        else Path(datasets[0].results_dir)
    )
    dat_dir = output_dir / "dat"

    first_test = pd.read_csv(Path(datasets[0].results_dir) / "test_data.csv")
    models = (
        config.models if config.models is not None else _detect_models(first_test)
    )

    resolved_colors: dict[str, str] = {
        m: _get_model_color(m, config.model_colors) for m in models
    }

    all_curves: list[dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]] = []

    for ds in datasets:
        test_data = pd.read_csv(Path(ds.results_dir) / "test_data.csv")
        dist = initiate_distribution(ds.distribution)
        mean_dist = dist

        mean_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        var_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for model in models:
            mean_curves[model] = _compute_mean_cc(
                test_data, model, dist, config.n_points,
            )
            var_curves[model] = _compute_variance_cc(
                test_data, model, dist,
                mean_model=config.mean_model_for_residuals,
                mean_dist=mean_dist,
                n_points=config.n_points,
            )

        all_curves.append({"mean": mean_curves, "variance": var_curves})

    dat_paths: list[dict[str, dict[str, Path]]] = []

    for ds_idx, ds in enumerate(datasets):
        ds_dat: dict[str, dict[str, Path]] = {"mean": {}, "variance": {}}
        label_slug = ds.label.lower().replace(" ", "_")

        for cc_type in ("mean", "variance"):
            for model in models:
                alphas, cc = all_curves[ds_idx][cc_type][model]
                fname = f"{label_slug}_{cc_type}_{model}.dat"
                fpath = dat_dir / fname
                _write_dat(fpath, alphas, cc)
                ds_dat[cc_type][model] = fpath

        diag_alphas = np.linspace(0.0, 1.0, config.n_points + 1)
        for cc_type in ("mean", "variance"):
            diag_path = dat_dir / f"{label_slug}_{cc_type}_diagonal.dat"
            _write_dat(diag_path, diag_alphas, diag_alphas)
            ds_dat[cc_type]["__diagonal__"] = diag_path

        dat_paths.append(ds_dat)

    n_rows = len(datasets)
    n_cols = 2
    figw, figh = config.figsize
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figw, figh * n_rows / max(n_rows, 1)),
        squeeze=False,
    )

    for ds_idx, ds in enumerate(datasets):
        for col, cc_type in enumerate(("mean", "variance")):
            cc_label = "expected value" if cc_type == "mean" else "variance"
            title = f"{ds.label}, {cc_label}"
            _plot_cc_subplot(
                axes[ds_idx, col],
                all_curves[ds_idx][cc_type],
                title=title,
                model_colors=resolved_colors,
                show_legend=(ds_idx == 0 and col == 0),
            )

    fig.tight_layout()
    png_path = output_dir / "concentration_curve_plot.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    tex_path = output_dir / "concentration_curve_plot.tex"
    write_concentration_tikz(
        path=tex_path,
        datasets=datasets,
        models=models,
        dat_paths=dat_paths,
        config=config,
    )


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser(
        description="Generate concentration curve plots from numerical illustration runs."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    with open(args.config) as fh:
        raw = yaml.safe_load(fh)

    plot_config = ConcentrationCurvePlotConfig(**raw)
    create_concentration_curve_plot(plot_config)
