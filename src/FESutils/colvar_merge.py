"""
Utilities to collect, validate, and merge PLUMED COLVAR files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .colvar_io import open_text_file


_COLVAR_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


def _natural_key(path: Path) -> tuple:
    """
    Produce a key for natural sorting of paths (numbers sorted numerically).
    """
    parts: list[str] = []
    for part in path.parts:
        for token in re.split(r"(\d+)", part):
            if token.isdigit():
                parts.append(f"{int(token):010d}")
            else:
                parts.append(token)
    return tuple(parts)


def _colvar_pattern(basename: str) -> re.Pattern[str]:
    try:
        return _COLVAR_PATTERN_CACHE[basename]
    except KeyError:
        pattern = re.compile(rf"^{re.escape(basename)}(\.\d+)?$")
        _COLVAR_PATTERN_CACHE[basename] = pattern
        return pattern


def discover_colvar_files(base_dir: str | Path, basename: str = "COLVAR") -> list[Path]:
    """
    Find COLVAR-like files in base_dir and its subdirectories.
    Matches basename and basename.NUMBER (natural-sorted).
    """
    pattern = _colvar_pattern(basename)
    base = Path(base_dir)
    candidates = [p for p in base.rglob("*") if p.is_file() and pattern.match(p.name)]
    candidates.sort(key=_natural_key)
    return candidates


def _stitch_time(values: pd.Series) -> pd.Series:
    """
    Ensure time monotonicity by adding offsets when time resets.
    """
    if values.empty:
        return values
    stitched = []
    offset = 0.0
    prev = values.iloc[0]
    prev_prev = prev
    stitched.append(prev)
    for idx in range(1, len(values)):
        current = values.iloc[idx]
        if current < prev:
            delta = prev - prev_prev if idx >= 2 else 0.0
            offset += prev + delta - current
        stitched_val = current + offset
        stitched.append(stitched_val)
        prev_prev = prev
        prev = stitched_val
    return pd.Series(stitched, index=values.index, dtype=float)


@dataclass
class MergeResult:
    dataframe: pd.DataFrame
    fields: Sequence[str]
    header_lines: list[str]
    source_files: list[Path]
    time_column: str | None


def read_colvar_dataframe(
    path: Path,
    expected_fields: Sequence[str] | None = None,
    discard_fraction: float = 0.0,
) -> tuple[list[str], list[str], pd.DataFrame] | None:
    """
    Read a single COLVAR file into a dataframe, discarding malformed rows.
    Returns (header_lines, fields, dataframe) or None if fields mismatch/empty.
    """
    with open_text_file(str(path)) as handle:
        first = handle.readline()
        if not first.startswith("#! FIELDS"):
            return None
        tokens = first.split()
        fields = tokens[2:]
        if expected_fields is not None and list(expected_fields) != fields:
            return None
        header_lines = [first]
        for line in handle:
            if line.startswith("#!"):
                header_lines.append(line)
            else:
                break
        # Collect valid rows; include the line we already read if it is data
        rows: list[list[float]] = []
        # If the previous line was data, process it
        if line and not line.startswith("#"):
            items = line.split()
            if len(items) == len(fields):
                try:
                    rows.append([float(x) for x in items])
                except ValueError:
                    pass
        for line in handle:
            if not line or line.startswith("#"):
                continue
            items = line.split()
            if len(items) != len(fields):
                continue
            try:
                rows.append([float(x) for x in items])
            except ValueError:
                continue
    if not rows:
        return None
    drop = int(len(rows) * discard_fraction) if discard_fraction > 0 else 0
    if drop >= len(rows):
        rows = []
    else:
        rows = rows[drop:]
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=fields, dtype=float)
    return header_lines, fields, df


def merge_colvar_files(
    base_dir: str | Path,
    basename: str = "COLVAR",
    discard_fraction: float = 0.0,
    keep_order: bool = True,
    time_ordered: bool = False,
    stitch_time: bool = True,
    output_path: str | Path | None = None,
) -> MergeResult:
    """
    Merge COLVAR files located under base_dir.

    - discard_fraction: drop that fraction of valid rows from the start of each file.
    - keep_order: append files in natural order (directories and suffix numbers).
    - time_ordered: sort merged data by the time column (if present).
    - stitch_time: when sorting, fix decreasing time segments by adding offsets.
    - output_path: if provided, write merged data with the first header.
    """
    files = discover_colvar_files(base_dir, basename=basename)
    if not files:
        raise FileNotFoundError(f"No COLVAR files matching '{basename}' found in {base_dir}")

    merged_frames: list[pd.DataFrame] = []
    header_lines: list[str] | None = None
    fields: Sequence[str] | None = None
    valid_sources: list[Path] = []

    for path in files:
        result = read_colvar_dataframe(path, expected_fields=fields, discard_fraction=discard_fraction)
        if result is None:
            continue
        hdr, flds, df = result
        if fields is None:
            fields = flds
        if header_lines is None:
            header_lines = hdr
        merged_frames.append(df)
        valid_sources.append(path)

    if not merged_frames or fields is None or header_lines is None:
        raise RuntimeError("No valid COLVAR data found to merge.")

    merged_df = pd.concat(merged_frames, ignore_index=True, copy=False)

    time_col = "time" if "time" in merged_df.columns else None
    if time_ordered and time_col is None:
        time_col = merged_df.columns[0]

    if time_ordered and time_col is not None:
        if stitch_time:
            merged_df[time_col] = _stitch_time(merged_df[time_col])
        merged_df = merged_df.sort_values(time_col, kind="mergesort", ignore_index=True)

    if keep_order and not time_ordered:
        merged_df = merged_df.reset_index(drop=True)

    if output_path:
        _write_colvar(Path(output_path), header_lines, merged_df)

    return MergeResult(
        dataframe=merged_df,
        fields=fields,
        header_lines=header_lines,
        source_files=valid_sources,
        time_column=time_col,
    )


def _write_colvar(path: Path, header_lines: list[str], df: pd.DataFrame) -> None:
    """
    Write merged COLVAR data using the provided header lines.
    """
    with open(path, "w", encoding="utf-8") as out:
        for line in header_lines:
            out.write(line if line.endswith("\n") else f"{line}\n")
        np.savetxt(out, df.to_numpy(), fmt="%.10g")
