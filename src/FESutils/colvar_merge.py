"""
Utilities to collect, validate, and merge PLUMED COLVAR files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

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
    # Parse header to get fields and header length
    header_lines: list[str] = []
    with open_text_file(str(path)) as handle:
        while True:
            pos = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.startswith("#"):
                handle.seek(pos)
                break
            header_lines.append(line)

    if not header_lines:
        return None
    first = header_lines[0]
    if not first.startswith("#! FIELDS"):
        return None
    fields = first.split()[2:]
    if expected_fields is not None and list(expected_fields) != fields:
        return None

    header_len = len(header_lines)
    # Fast read with pandas; skip malformed rows automatically
    df = pd.read_table(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=fields,
        skiprows=header_len,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
    )
    if not df.empty:
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna()
    if df.empty:
        return None
    drop = int(len(df) * discard_fraction) if discard_fraction > 0 else 0
    if drop >= len(df):
        return None
    if drop > 0:
        df = df.iloc[drop:]
    df = df.reset_index(drop=True)
    return header_lines, fields, df


def merge_colvar_files(
    base_dir: str | Path,
    basename: str = "COLVAR",
    discard_fraction: float = 0.1,
    keep_order: bool = True,
    time_ordered: bool = False,
    output_path: str | Path | None = None,
    verbose: bool = True,
) -> MergeResult:
    """
    Merge COLVAR files located under base_dir.

    - discard_fraction: drop that fraction of valid rows from the start of each file.
    - keep_order: append files in natural order (directories and suffix numbers).
    - time_ordered: sort merged data by the time column (if present).
    - output_path: if provided, write merged data with the first header.
    - verbose: print progress information while loading files.
    """
    if not (0.0 <= discard_fraction <= 1.0):
        raise ValueError("discard_fraction must be between 0.0 and 1.0")

    files = discover_colvar_files(base_dir, basename=basename)
    if not files:
        raise FileNotFoundError(f"No COLVAR files matching '{basename}' found in {base_dir}")
    total_files = len(files)
    if verbose:
        print(f"Found {total_files} COLVAR file(s) matching '{basename}'")
        print(f"Loading COLVAR files: 0/{total_files}", end="\r", flush=True)

    merged_frames: list[pd.DataFrame] = []
    header_lines: list[str] | None = None
    fields: Sequence[str] | None = None
    valid_sources: list[Path] = []

    for idx, path in enumerate(files, start=1):
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
        if verbose:
            print(f"Loading COLVAR files: {idx}/{total_files}", end="\r", flush=True)

    if verbose:
        print(f"Loading COLVAR files: {len(valid_sources)}/{total_files} (done)          ")

    if not merged_frames or fields is None or header_lines is None:
        raise RuntimeError("No valid COLVAR data found to merge.")

    merged_df = pd.concat(merged_frames, ignore_index=True, copy=False)

    time_col = "time" if "time" in merged_df.columns else None
    if time_ordered and time_col is None:
        time_col = merged_df.columns[0]

    if time_ordered and time_col is not None:
        merged_df = merged_df.sort_values(time_col, kind="mergesort", ignore_index=True)

    if keep_order and not time_ordered:
        merged_df = merged_df.reset_index(drop=True)

    if output_path:
        if verbose:
            print("Writing merged COLVAR...", end="\r", flush=True)
        _write_colvar(Path(output_path), header_lines, merged_df)
        if verbose:
            print(f"Wrote merged COLVAR to {output_path}          ")

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
