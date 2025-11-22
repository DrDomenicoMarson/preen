#!/usr/bin/env python3
"""
Lightweight command-line interface for FESutils.
Currently supports: preen colvar merge
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .colvar_merge import merge_colvar_files
from .colvar_plot import plot_colvar_timeseries


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
        "--time-ordered",
        action="store_true",
        help="Sort merged data by time.",
    )
    parser.add_argument(
        "--no-keep-order",
        dest="keep_order",
        action="store_false",
        help="Do not preserve natural file order (only relevant when not time-ordering).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Write merged COLVAR to this path. If omitted, data stays in memory.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    parser.set_defaults(func=_handle_colvar_merge, keep_order=True)


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
        help="Specific columns to plot (default: all except time).",
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
    output_path = args.output if args.output else None
    outputs = plot_colvar_timeseries(
        base_dir=args.base_dir,
        basename=args.basename,
        discard_fraction=args.discard_fraction,
        columns=args.columns,
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
    result = merge_colvar_files(
        base_dir=args.base_dir,
        basename=args.basename,
        discard_fraction=args.discard_fraction,
        keep_order=args.keep_order,
        time_ordered=args.time_ordered,
        output_path=args.output,
        verbose=not args.quiet,
        build_dataframe=False,
    )
    total_rows = result.row_count
    print(f"Merged {len(result.source_files)} file(s); total rows: {total_rows}")
    if args.output:
        print(f"Wrote merged COLVAR to: {Path(args.output).resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="preen", description="Preen CLI utilities")
    subparsers = parser.add_subparsers(dest="command")

    colvar_parser = subparsers.add_parser("colvar", help="COLVAR utilities")
    colvar_subparsers = colvar_parser.add_subparsers(dest="colvar_command")
    _add_colvar_merge(colvar_subparsers)
    _add_colvar_plot(colvar_subparsers)

    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        return args.func(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
