"""TikZ groupplot writer for concentration curve plots.

Produces a ``.tex`` snippet that can be ``\\input``-ted directly into a LaTeX
document, matching the style of the paper's
``src/figures/concentration_curves.tex``.

Each subplot contains:
- One ``\\addplot`` line per model (with legend entry in the first panel).
- A dashed black diagonal (identity line) with ``forget plot`` in subsequent
  panels.
- Fixed axes [0, 1] × [0, 1].

Colours are resolved from the config or from ``DEFAULT_TIKZ_COLORS``.
"""

from __future__ import annotations

from pathlib import Path

from .concentration_config import (
    DEFAULT_TIKZ_COLORS,
    ConcentrationCurvePlotConfig,
    DatasetConfig,
)


def _read_dat(path: Path) -> list[tuple[float, float]]:
    """Read a ``.dat`` file and return ``[(alpha, value), ...]``."""
    pairs: list[tuple[float, float]] = []
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        pairs.append((float(parts[0]), float(parts[1])))
    return pairs


def _inline_table(path: Path) -> str:
    """Return an inline ``table {%...}`` block from a ``.dat`` file."""
    pairs = _read_dat(path)
    rows = "\n".join(f"{a} {v}" for a, v in pairs)
    return f"table {{%\n{rows}\n}}"


def _resolve_tikz_color_name(model: str) -> str:
    """Return a LaTeX-safe colour name for *model*."""
    return model.replace(" ", "").replace("_", "")


def _resolve_tikz_color_rgb(
    model: str,
    overrides: dict[str, tuple[int, int, int]] | None,
) -> tuple[int, int, int]:
    """Resolve TikZ RGB triple for *model*."""
    if overrides and model in overrides:
        return overrides[model]
    if model in DEFAULT_TIKZ_COLORS:
        return DEFAULT_TIKZ_COLORS[model]
    fallback = [
        (214, 39, 40), (148, 103, 189), (140, 86, 75),
        (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
    ]
    return fallback[hash(model) % len(fallback)]


def _subplot_body(
    ds: DatasetConfig,
    cc_type: str,
    models: list[str],
    series_paths: dict[str, Path],
    is_first_panel: bool,
    tikz_overrides: dict[str, tuple[int, int, int]] | None,
) -> str:
    """Return the tikz commands for a single ``\\nextgroupplot`` panel."""
    cc_label = "expected value" if cc_type == "mean" else "variance"
    title = f"{ds.label}, {cc_label}"

    lines: list[str] = []

    opts = [
        "tick align=outside,",
        "tick pos=left,",
        f"title={{{title}}},",
        "x grid style={darkgray176},",
        "xmin=0, xmax=1,",
        "xtick style={color=black},",
        "y grid style={darkgray176},",
        "ymin=0, ymax=1,",
        "ytick style={color=black}",
    ]

    if is_first_panel:
        legend_opts = [
            "legend cell align={left},",
            "legend style={",
            "  fill opacity=0.8,",
            "  draw opacity=1,",
            "  text opacity=1,",
            "  at={(0.03,0.97)},",
            "  anchor=north west,",
            "  draw=lightgray204",
            "},",
        ]
        opts = legend_opts + opts

    lines.append("\\nextgroupplot[")
    lines.extend(opts)
    lines.append("]")

    for model in models:
        color_name = _resolve_tikz_color_name(model)
        table = _inline_table(series_paths[model])

        if is_first_panel:
            lines.append(f"\\addplot [semithick, {color_name}]")
            lines.append(f"{table};")
            lines.append(f"\\addlegendentry{{{model.upper()}}}")
        else:
            lines.append(f"\\addplot [semithick, {color_name}]")
            lines.append(f"{table};")

    diag_table = _inline_table(series_paths["__diagonal__"])
    forget = ", forget plot" if is_first_panel else ""
    lines.append(f"\\addplot [semithick, black, dashed{forget}]")
    lines.append(f"{diag_table};")

    return "\n".join(lines)


def write_concentration_tikz(
    path: Path,
    datasets: list[DatasetConfig],
    models: list[str],
    dat_paths: list[dict[str, dict[str, Path]]],
    config: ConcentrationCurvePlotConfig,
) -> None:
    """Write a pgfplots groupplot ``.tex`` file to *path*.

    Args:
        path: Destination ``.tex`` file.
        datasets: One entry per row in the figure.
        models: Ordered list of model names.
        dat_paths: ``dat_paths[ds_idx][cc_type][model_or_diagonal]`` → Path.
        config: Full plot configuration (used for colour overrides).
    """
    n_rows = len(datasets)
    n_cols = 2

    color_defs: list[str] = []
    color_defs.append("\\definecolor{darkgray176}{RGB}{176,176,176}")
    color_defs.append("\\definecolor{lightgray204}{RGB}{204,204,204}")

    for model in models:
        cname = _resolve_tikz_color_name(model)
        r, g, b = _resolve_tikz_color_rgb(model, config.tikz_colors)
        color_defs.append(f"\\definecolor{{{cname}}}{{RGB}}{{{r},{g},{b}}}")

    subplots: list[str] = []
    is_first = True

    for ds_idx, ds in enumerate(datasets):
        for cc_type in ("mean", "variance"):
            subplots.append(
                _subplot_body(
                    ds=ds,
                    cc_type=cc_type,
                    models=models,
                    series_paths=dat_paths[ds_idx][cc_type],
                    is_first_panel=is_first,
                    tikz_overrides=config.tikz_colors,
                )
            )
            is_first = False

    body = "\n\n".join(subplots)
    defs = "\n".join(color_defs)

    tex = (
        "% Generated by numerical_illustration/plots/concentration_tikz_writer.py\n"
        "\\begin{tikzpicture}[scale = 0.75]\n"
        "\n"
        f"{defs}\n"
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
