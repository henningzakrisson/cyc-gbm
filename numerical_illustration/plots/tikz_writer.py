"""Tikz groupplot writer for the binned response plot.

Produces a ``.tex`` snippet that can be ``\\input``-ted directly into a LaTeX
document, matching the style of the existing ``src/figures/beta_prime_sim.tex``
in the paper repository.

Each subplot is rendered with:
- A filled grey ``\\path`` polygon for the ±1 std band (upper edge forward,
  lower edge reversed, closed cycle) — identical to the tikzplotlib approach.
- A thick black ``\\addplot`` line with inline data for the predicted bin mean.
- Black dots (``only marks``) with inline data for the observed bin averages.

This avoids the ``fill between`` pgfplots library and external ``.dat`` files,
producing self-contained ``.tex`` output that matches the original paper figure.
"""

from __future__ import annotations

from pathlib import Path


def _read_dat(path: Path) -> list[tuple[int, float]]:
    """Read a ``.dat`` file and return ``[(index, value), ...]``."""
    pairs: list[tuple[int, float]] = []
    for line in path.read_text().strip().splitlines():
        idx_s, val_s = line.split()
        pairs.append((int(idx_s), float(val_s)))
    return pairs


def _band_path(upper_path: Path, lower_path: Path) -> str:
    """Return a ``\\path`` polygon for the grey ±1 std band.

    Draws the upper edge left-to-right, then the lower edge right-to-left,
    and closes the cycle — exactly as tikzplotlib v0.10.1 generates.
    """
    upper = _read_dat(upper_path)
    lower = _read_dat(lower_path)

    coords: list[str] = []
    # Upper edge: forward
    for idx, val in upper:
        coords.append(f"(axis cs:{idx},{val})")
    # Lower edge: reversed
    for idx, val in reversed(lower):
        coords.append(f"(axis cs:{idx},{val})")

    joined = "\n--".join(coords)
    return f"\\path [draw=gray, fill=gray, opacity=0.4]\n{joined}\n--cycle;"


def _inline_table(path: Path) -> str:
    """Return an inline ``table {%...}`` block from a ``.dat`` file."""
    pairs = _read_dat(path)
    rows = "\n".join(f"{idx} {val}" for idx, val in pairs)
    return f"table {{%\n{rows}\n}}"


def _inline_table_with_error(obs_path: Path, err_path: Path) -> str:
    """Return an inline 3-column ``table`` block with y-error data."""
    obs = _read_dat(obs_path)
    err = _read_dat(err_path)
    rows = "\n".join(f"{idx} {val} {e}" for (idx, val), (_, e) in zip(obs, err))
    return f"table [y error index=2] {{%\n{rows}\n}}"


def _subplot_body(
    model: str,
    distribution_label: str,
    series_paths: dict[str, Path],
    n_bins: int,
    ymin: float,
    ymax: float,
    bias_adjusted: bool,
) -> str:
    """Return the tikz commands for a single \\nextgroupplot panel."""
    suffix = ", bias adjusted" if bias_adjusted else ""
    title = f"{model.upper()}, {distribution_label}{suffix}"

    band = _band_path(series_paths["band_upper"], series_paths["band_lower"])
    mean_table = _inline_table(series_paths["mean"])
    obs_err_table = _inline_table_with_error(
        series_paths["observed"], series_paths["obs_std"]
    )

    lines = [
        "\\nextgroupplot[",
        "tick align=outside,",
        "tick pos=left,",
        f"title={{{title}}},",
        "x grid style={darkgray176},",
        f"xmin=0, xmax={n_bins},",
        "xtick style={color=black},",
        "y grid style={darkgray176},",
        f"ymin={ymin:.15g}, ymax={ymax:.15g},",
        "ytick style={color=black}",
        "]",
        band,
        "",
        f"\\addplot [thick, black]\n{mean_table};",
        "\\addplot [semithick, black, mark=*, mark size=1, mark options={solid}, only marks,",
        "  error bars/.cd, y dir=both, y explicit]",
        f"{obs_err_table};",
    ]
    return "\n".join(lines)


def write_tikz(
    path: Path,
    models: list[str],
    distribution_label: str,
    dat_paths: dict[str, dict[str, dict[str, Path]]],
    n_bins: int,
    ymin: float,
    ymax: float,
    bias_adjustment: bool,
) -> None:
    """Write a pgfplots groupplot ``.tex`` file to *path*.

    Args:
        path: Destination ``.tex`` file.
        models: Ordered list of model names (one column per model).
        distribution_label: Human-readable distribution name for subplot titles.
        dat_paths: Nested dict ``dat_paths[model][variant][series]`` where
            *variant* is ``"unadjusted"`` or ``"adjusted"`` and *series* is
            one of ``"mean"``, ``"observed"``, ``"band_upper"``, ``"band_lower"``.
        n_bins: Number of bins (sets the x-axis range to ``[0, n_bins]``).
        ymin: Shared y-axis minimum across all panels.
        ymax: Shared y-axis maximum across all panels.
        bias_adjustment: Whether a second row of bias-adjusted panels is included.
    """
    n_cols = len(models)
    n_rows = 2 if bias_adjustment else 1

    subplots: list[str] = []

    # Top row: unadjusted
    for model in models:
        subplots.append(
            _subplot_body(
                model=model,
                distribution_label=distribution_label,
                series_paths=dat_paths[model]["unadjusted"],
                n_bins=n_bins,
                ymin=ymin,
                ymax=ymax,
                bias_adjusted=False,
            )
        )

    # Bottom row: bias-adjusted
    if bias_adjustment:
        for model in models:
            subplots.append(
                _subplot_body(
                    model=model,
                    distribution_label=distribution_label,
                    series_paths=dat_paths[model]["adjusted"],
                    n_bins=n_bins,
                    ymin=ymin,
                    ymax=ymax,
                    bias_adjusted=True,
                )
            )

    body = "\n\n".join(subplots)

    tex = (
        "% Generated by numerical_illustration/plots/tikz_writer.py\n"
        "\\begin{tikzpicture}[scale = 0.75]\n"
        "\n"
        "\\definecolor{darkgray176}{RGB}{176,176,176}\n"
        "\\definecolor{gray}{RGB}{128,128,128}\n"
        "\n"
        f"\\begin{{groupplot}}[group style={{group size={n_cols} by {n_rows}, "
        "vertical sep=1.5cm}]\n"
        f"{body}\n"
        "\\end{groupplot}\n"
        "\n"
        "\\end{tikzpicture}\n"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex)
