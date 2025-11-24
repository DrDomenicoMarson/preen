#!/usr/bin/env python3
"""
Lightweight command-line interface for FESutils.
Currently supports: preen colvar merge
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .colvar_merge import merge_colvar_files, discover_colvar_files, _read_header_and_count
from .colvar_plot import plot_colvar_timeseries
from .api import calculate_fes
from .fes_config import FESConfig


def _add_colvar_merge(subparsers):
    parser = subparsers.add_parser(
        "merge",
        help="Merge COLVAR files from a directory tree",
        description="Merge COLVAR files (including walker-numbered files) with validation and optional time ordering.",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory to scan (default: current directory)",
    )
    parser.add_argument(
        "--basename",
        default="COLVAR",
        help="Base file name to match (default: COLVAR; also matches COLVAR.NUMBER)",
    )
    parser.add_argument(
        "--discard-fraction",
        type=float,
        default=0.1,
        help="Fraction of each file to discard from the start (0.0-1.0). Default: 0.1.",
    )
    parser.add_argument(
        "--concat",
        dest="time_ordered",
        action="store_false",
        help="Concatenate files instead of interleaving lines (default is interleaved).",
    )
    parser.set_defaults(time_ordered=True)
    parser.add_argument(
        "--output",
        type=str,
        help="Write merged COLVAR to this path (default: BASENAME_merged.dat in base-dir).",
    )
    parser.add_argument(
        "--allow-header-mismatch",
        action="store_true",
        help="Allow merging files whose header line count differs (warns and proceeds).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Threads for numba (default: min(16, available cores)).",
    )
    parser.set_defaults(func=_handle_colvar_merge)


def _parse_tuple(value: str, cast):
    parts = [p.strip() for p in value.replace(",", " ").split() if p.strip()]
    return tuple(cast(p) for p in parts)


def _parse_values(values, cast=str):
    """
    Normalize mixed comma/space inputs from argparse.
    Accepts list from nargs or a single comma-separated string.
    """
    if values is None:
        return None
    if isinstance(values, str):
        raw_items = values
    elif isinstance(values, (list, tuple)) and len(values) == 1:
        raw_items = values[0]
    else:
        raw_items = values
    items: list[str] = []
    if isinstance(raw_items, str):
        for token in raw_items.replace(",", " ").split():
            if token:
                items.append(token)
    else:
        for token in raw_items:
            if token is None:
                continue
            if isinstance(token, str) and "," in token and len(raw_items) == 1:
                for sub in token.replace(",", " ").split():
                    if sub:
                        items.append(sub)
            else:
                items.append(str(token))
    return tuple(cast(item) for item in items)


def _add_colvar_reweight(subparsers):
    parser = subparsers.add_parser(
        "reweight",
        help="Reweight COLVAR data to compute FES (no merged COLVAR written).",
    )
    parser.add_argument("--base-dir", default=".", help="Base directory to scan.")
    parser.add_argument("--basename", default="COLVAR", help="Base file name to match (default: COLVAR).")
    parser.add_argument(
        "--discard-fraction",
        type=float,
        default=0.1,
        help="Fraction of each file to discard from the start (0.0-1.0). Default: 0.1.",
    )
    parser.add_argument(
        "--concat",
        dest="time_ordered",
        action="store_false",
        help="Concatenate files instead of interleaving lines (default is interleaved).",
    )
    parser.set_defaults(time_ordered=True)
    parser.add_argument(
        "--columns",
        nargs="+",
        required=True,
        help="CV column names to use (1 or 2). Comma or space separated.",
    )
    parser.add_argument(
        "--bias-spec",
        default=".bias",
        help="Bias specification (name or column list) default '.bias'.",
    )
    parser.add_argument(
        "--sigma",
        required=True,
        help="Sigma values (1 or 2), comma or space separated.",
    )
    parser.add_argument(
        "--grid-bin",
        default=None,
        help="Grid bins (default 100 or 50 50 for 2D), comma or space separated.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=300.0,
        help="Temperature in K (default 300).",
    )
    parser.add_argument(
        "--kbt",
        type=float,
        default=None,
        help="Override kBT in kJ/mol (if provided, temp is ignored).",
    )
    parser.add_argument(
        "--output",
        default="fes-rew.dat",
        help="Output FES file (default: fes-rew.dat).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable plotting of the resulting FES.",
    )
    parser.add_argument(
        "--fmt",
        default="% 12.6f",
        help="Output format for numeric values.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Threads for numba (default: min(16, available cores)).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    parser.set_defaults(func=_handle_colvar_reweight)


def _add_colvar_plot(subparsers):
    parser = subparsers.add_parser(
        "plot",
        help="Plot COLVAR time series",
        description="Plot COLVAR time series across all matching files.",
    )
    parser.add_argument("--base-dir", default=".", help="Base directory to scan.")
    parser.add_argument("--basename", default="COLVAR", help="Base file name to match (default: COLVAR).")
    parser.add_argument(
        "--discard-fraction",
        type=float,
        default=0.0,
        help="Fraction of each file to discard from the start (0.0-1.0).",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        help="Specific columns to plot (comma or space separated; default: all except time).",
    )
    parser.add_argument(
        "--marker",
        default=",",
        help="Matplotlib marker for time-series scatter (default: ',').",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=0.4,
        help="Marker size for time-series scatter (default: 0.4).",
    )
    parser.add_argument(
        "--time-column",
        default="time",
        help="Column to use for the x-axis (default: time; falls back to first column if missing).",
    )
    parser.add_argument(
        "--output",
        default="colvar_timeseries.png",
        help="Path for aggregated plot (set empty string to skip).",
    )
    parser.add_argument(
        "--per-run",
        action="store_true",
        help="Also write one plot per COLVAR file next to the source file.",
    )
    parser.add_argument(
        "--no-hist",
        dest="include_hist",
        action="store_false",
        help="Do not generate histogram plots.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    parser.set_defaults(func=_handle_colvar_plot, include_hist=True)


def _handle_colvar_plot(args):
    columns = _parse_values(args.columns, str) if args.columns is not None else None
    output_path = args.output if args.output else None
    outputs = plot_colvar_timeseries(
        base_dir=args.base_dir,
        basename=args.basename,
        discard_fraction=args.discard_fraction,
        columns=columns,
        time_column=args.time_column,
        output_path=output_path,
        per_run=args.per_run,
        include_hist=args.include_hist,
        marker=args.marker,
        marker_size=args.marker_size,
        verbose=not args.quiet,
    )
    for label, path in outputs.items():
        try:
            display_path = Path(path).resolve().relative_to(Path.cwd())
        except ValueError:
            display_path = Path(path).resolve()
        print(f"{label}: {display_path}")
    return 0


def _handle_colvar_merge(args):
    default_output = Path(args.base_dir) / f"{args.basename}_merged.dat"
    output_path = Path(args.output) if args.output else default_output
    result = merge_colvar_files(
        base_dir=args.base_dir,
        basename=args.basename,
        discard_fraction=args.discard_fraction,
        time_ordered=args.time_ordered,
        output_path=output_path,
        verbose=not args.quiet,
        build_dataframe=False,
        allow_header_mismatch=args.allow_header_mismatch,
    )
    total_rows = result.row_count
    print(f"Merged {len(result.source_files)} file(s); total rows: {total_rows}")
    out_path = Path(output_path).resolve()
    try:
        display_path = out_path.relative_to(Path.cwd())
    except ValueError:
        display_path = out_path
    print(f"Wrote merged COLVAR to: {display_path}")
    return 0


def _handle_colvar_reweight(args):
    # Parse sigma and grid
    sigma_tuple = _parse_values(args.sigma, float)
    if len(sigma_tuple) not in (1, 2):
        raise ValueError("sigma must have 1 or 2 values")
    grid_bin_tuple = None
    if args.grid_bin:
        grid_bin_tuple = _parse_values(args.grid_bin, int)
    else:
        grid_bin_tuple = (100,) if len(sigma_tuple) == 1 else (50, 50)

    columns = _parse_values(args.columns, str) if args.columns is not None else None
    if columns is None:
        raise ValueError("At least one column must be provided via --columns")

    # Quick header validation before heavy work
    files = discover_colvar_files(args.base_dir, basename=args.basename)
    if not files:
        raise FileNotFoundError(f"No COLVAR files matching '{args.basename}' found in {args.base_dir}")
    header_lines_first, fields_first, _ = _read_header_and_count(files[0])
    if not header_lines_first:
        raise RuntimeError(f"Failed to read header from {files[0]}")
    missing = [c for c in columns if c not in fields_first]
    if missing:
        available_str = ", ".join(fields_first)
        raise ValueError(
            f"Requested columns not found: {', '.join(missing)}. Available columns: {available_str}"
        )

    verbose = not args.quiet
    merge_result = merge_colvar_files(
        base_dir=args.base_dir,
        basename=args.basename,
        discard_fraction=args.discard_fraction,
        time_ordered=args.time_ordered,
        output_path=None,
        verbose=verbose,
        build_dataframe=True,
    )
    available_fields = list(merge_result.fields)
    missing = [c for c in columns if c not in available_fields]
    if missing:
        available_str = ", ".join(available_fields)
        raise ValueError(
            f"Requested columns not found: {', '.join(missing)}. Available columns: {available_str}"
        )
    if verbose:
        cv_display = ", ".join(columns)
        bias_display = args.bias_spec
        print(f"Using CVs: {cv_display}")
        print(f"Using bias spec: {bias_display}")

    config = FESConfig(
        filename="MERGED_IN_MEMORY",
        outfile=args.output,
        kbt=args.kbt if args.kbt is not None else None,
        temp=None if args.kbt is not None else args.temp,
        grid_bin=grid_bin_tuple,
        sigma=sigma_tuple,
        cv_spec=tuple(columns),
        bias_spec=args.bias_spec,
        backup=False,
        plot=args.plot,
        fmt=args.fmt,
        num_threads=args.num_threads,
    )
    calculate_fes(config, merge_result=merge_result)
    out_path = Path(args.output).resolve()
    try:
        display_path = out_path.relative_to(Path.cwd())
    except ValueError:
        display_path = out_path
    print(f"Computed FES to: {display_path}")
    if args.plot:
        png_path = out_path.with_suffix(".png")
        try:
            png_display = png_path.relative_to(Path.cwd())
        except ValueError:
            png_display = png_path
        print(f"Plot saved to: {png_display}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="preen", description="Preen CLI utilities")
    subparsers = parser.add_subparsers(dest="command")

    colvar_parser = subparsers.add_parser("colvar", help="COLVAR utilities")
    colvar_subparsers = colvar_parser.add_subparsers(dest="colvar_command")
    _add_colvar_merge(colvar_subparsers)
    _add_colvar_plot(colvar_subparsers)
    _add_colvar_reweight(colvar_subparsers)

    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        return args.func(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
