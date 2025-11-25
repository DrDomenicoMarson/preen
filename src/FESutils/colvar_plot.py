"""
Plot COLVAR time series across multiple files.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .colvar_merge import discover_colvar_files, read_colvar_dataframe


def _resolve_columns(fields: Sequence[str], columns: Sequence[str] | None) -> list[str]:
    if columns is None:
        return list(fields[1:])  # default: all except time column
    missing = [c for c in columns if c not in fields]
    if missing:
        available = ", ".join(fields)
        raise ValueError(
            f"Columns not found in COLVAR header: {', '.join(missing)}. "
            f"Available columns: {available}"
        )
    return list(columns)


def plot_colvar_timeseries(
    base_dir: str | Path,
    basename: str = "COLVAR",
    discard_fraction: float = 0.1,
    columns: Sequence[str] | None = None,
    time_column: str | None = "time",
    output_path: str | Path | None = "colvar_timeseries.png",
    per_run: bool = False,
    include_hist: bool = True,
    marker: str = ",",
    marker_size: float = 0.4,
    verbose: bool = True,
) -> dict[str, Path]:
    """
    Plot time series from COLVAR files discovered in base_dir.

    - output_path: aggregated plot of all files (if provided).
    - per_run: also write one plot per COLVAR file next to the source file.
    - include_hist: also plot histograms for the selected columns.
    - verbose: print progress information while loading/plotting.
    Returns mapping of label -> output path.
    """
    if not (0.0 <= discard_fraction <= 1.0):
        raise ValueError("discard_fraction must be between 0.0 and 1.0")

    files = discover_colvar_files(base_dir, basename=basename)
    aggregates = []
    headers: Sequence[str] | None = None
    outputs: dict[str, Path] = {}
    if verbose:
        print(f"Found {len(files)} COLVAR file(s) matching '{basename}'")
        print(f"Loading COLVAR files: 0/{len(files)}", end="\r", flush=True)

    for idx, path in enumerate(files, start=1):
        result = read_colvar_dataframe(path, expected_fields=headers, discard_fraction=discard_fraction)
        if result is None:
            continue
        header_lines, fields, df = result
        if headers is None:
            headers = fields
        time_col = time_column if time_column and time_column in fields else fields[0]
        cols_to_plot = _resolve_columns(fields, columns)
        aggregates.append((path, time_col, cols_to_plot, df))
        if verbose:
            print(f"Loading COLVAR files: {idx}/{len(files)}", end="\r", flush=True)

    if verbose:
        print(f"Loading COLVAR files: {len(aggregates)}/{len(files)} (done)          ")

    if not aggregates:
        raise FileNotFoundError(f"No valid COLVAR files found under {base_dir}")

    if per_run and verbose:
        total_plots = len(aggregates)
        print(f"Plotting per-run: 0/{total_plots}", end="\r", flush=True)
        for i, (path, time_col, cols_to_plot, df) in enumerate(aggregates, start=1):
            out = path.with_suffix("")  # drop .gz if present
            out_path = Path(f"{out}_timeseries.png")
            _plot_single(
                path.name,
                time_col,
                cols_to_plot,
                [df],
                out_path,
                discard_fraction=discard_fraction,
                marker=marker,
                marker_size=marker_size,
            )
            outputs[path.name] = out_path
            if include_hist:
                hist_path = Path(f"{out}_hist.png")
                _plot_histograms(
                    path.name,
                    cols_to_plot,
                    [df],
                    hist_path,
                    discard_fraction=discard_fraction,
                    marker_size=marker_size,
                )
                outputs[f"{path.name}_hist"] = hist_path
            print(f"Plotting per-run: {i}/{total_plots}", end="\r", flush=True)
        print(f"Plotting per-run: {total_plots}/{total_plots} (done)          ")
    elif per_run:
        for (path, time_col, cols_to_plot, df) in aggregates:
            out = path.with_suffix("")
            out_path = Path(f"{out}_timeseries.png")
            _plot_single(
                path.name,
                time_col,
                cols_to_plot,
                [df],
                out_path,
                discard_fraction=discard_fraction,
                marker=marker,
                marker_size=marker_size,
            )
            outputs[path.name] = out_path
            if include_hist:
                hist_path = Path(f"{out}_hist.png")
                _plot_histograms(
                    path.name,
                    cols_to_plot,
                    [df],
                    hist_path,
                    discard_fraction=discard_fraction,
                    marker_size=marker_size,
                )
                outputs[f"{path.name}_hist"] = hist_path

    if output_path:
        time_col = aggregates[0][1]
        cols_to_plot = aggregates[0][2]
        dfs = [info[3] for info in aggregates]
        labels = [info[0].name for info in aggregates]
        out_path = Path(output_path)
        if verbose:
            print("Plotting aggregate...", end="\r", flush=True)
        _plot_single(
            labels,
            time_col,
            cols_to_plot,
            dfs,
            out_path,
            discard_fraction=discard_fraction,
            marker=marker,
            marker_size=marker_size,
        )
        outputs["aggregate"] = out_path
        if include_hist:
            hist_out = out_path.with_name(out_path.stem + "_hist.png")
            _plot_histograms(
                labels,
                cols_to_plot,
                dfs,
                hist_out,
                discard_fraction=discard_fraction,
                marker_size=marker_size,
            )
            outputs["aggregate_hist"] = hist_out
        if verbose:
            print(f"Wrote aggregate plot(s) to {out_path}          ")

    return outputs


def _plot_single(labels, time_col: str, cols: Sequence[str], dfs: list, out_path: Path, discard_fraction: float, marker: str, marker_size: float) -> None:
    n_plots = len(cols)
    cols_per_row = 3 if n_plots > 2 else n_plots
    rows = math.ceil(n_plots / cols_per_row)
    fig, axes = plt.subplots(rows, cols_per_row, squeeze=False)
    flat_axes = axes.flat
    for idx, col in enumerate(cols):
        ax = flat_axes[idx]
        for df, label in zip(dfs, labels if isinstance(labels, list) else [labels] * len(dfs)):
            ax.plot(
                df[time_col],
                df[col],
                linestyle="None",
                marker=marker,
                markersize=marker_size,
                markeredgewidth=0.0,
                alpha=0.8,
            )
        ax.set_title(col)
    title = f"COLVAR time series (discard {discard_fraction*100:.1f}% per file)"
    fig.suptitle(title, fontsize=10)
    # Hide unused axes
    for ax in flat_axes[n_plots:]:
        ax.set_visible(False)
    rect_right = 1
    fig.tight_layout(rect=[0, 0, rect_right, 0.96])
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def _plot_histograms(labels, cols: Sequence[str], dfs: list, out_path: Path, discard_fraction: float, marker: str | None = None, marker_size: float | None = None) -> None:
    n_plots = len(cols)
    cols_per_row = 3 if n_plots > 2 else n_plots
    rows = math.ceil(n_plots / cols_per_row)
    width_factor = 5.0 if cols_per_row > 1 else 4.5
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(width_factor * cols_per_row, 2.5 * rows), squeeze=False)
    flat_axes = axes.flat
    label_list = []
    for idx, col in enumerate(cols):
        ax = flat_axes[idx]
        for df, label in zip(dfs, labels if isinstance(labels, list) else [labels] * len(dfs)):
            ax.hist(df[col], bins=30, alpha=0.6, label=label)
            label_list.append(label)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
    unique_labels = list(dict.fromkeys(label_list))
    legend = None
    legend_space = False
    title = f"COLVAR histograms (discard {discard_fraction*100:.1f}% per file)"
    fig.suptitle(title, fontsize=10)
    for ax in flat_axes[n_plots:]:
        ax.set_visible(False)
    rect_right = 1
    fig.tight_layout(rect=[0, 0, rect_right, 0.96])
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
