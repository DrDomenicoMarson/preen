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
    row_count: int


def _read_header_and_count(path: Path) -> tuple[list[str], list[str], int]:
    """
    Read header lines, extract fields, and count data lines (after header).
    """
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
        data_lines = sum(1 for _ in handle)
    if not header_lines or not header_lines[0].startswith("#! FIELDS"):
        return [], [], 0
    fields = header_lines[0].split()[2:]
    return header_lines, fields, data_lines


def read_colvar_dataframe(
    path: Path,
    expected_fields: Sequence[str] | None = None,
    discard_fraction: float = 0.0,
) -> tuple[list[str], list[str], pd.DataFrame] | None:
    """
    Read a single COLVAR file into a dataframe, discarding malformed rows.
    Returns (header_lines, fields, dataframe) or None if fields mismatch/empty.
    """
    header_lines, fields, data_lines_first = _read_header_and_count(path)
    if not header_lines:
        return None
    if expected_fields is not None and list(expected_fields) != fields:
        return None

    header_len = len(header_lines)
    raw_lines: list[str] = []
    with open_text_file(str(path)) as handle:
        for _ in range(header_len):
            next(handle, None)
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != len(fields):
                continue
            if not line.endswith("\n"):
                line = line + "\n"
            raw_lines.append(line)
    if not raw_lines:
        return None
    drop = int(len(raw_lines) * discard_fraction) if discard_fraction > 0 else 0
    if drop >= len(raw_lines):
        return None
    if drop > 0:
        raw_lines = raw_lines[drop:]
    arrays = [np.fromstring(l, sep=" ") for l in raw_lines]
    data = np.vstack(arrays)
    df = pd.DataFrame(data, columns=fields, copy=False)
    return header_lines, fields, df


def merge_colvar_files(
    base_dir: str | Path,
    basename: str = "COLVAR",
    discard_fraction: float = 0.1,
    keep_order: bool = True,
    time_ordered: bool = False,
    output_path: str | Path | None = None,
    verbose: bool = True,
    build_dataframe: bool = True,
) -> MergeResult:
    """
    Merge COLVAR files located under base_dir.

    - discard_fraction: drop that fraction of valid rows from the start of each file.
    - keep_order: append files in natural order (directories and suffix numbers).
    - time_ordered: sort merged data by the time column (if present).
    - output_path: if provided, write merged data with the first header.
    - verbose: print progress information while loading files.
    - build_dataframe: build numeric dataframe (set False to speed up merge-only CLI).
    """
    if not (0.0 <= discard_fraction <= 1.0):
        raise ValueError("discard_fraction must be between 0.0 and 1.0")
    if time_ordered:
        build_dataframe = True

    files = discover_colvar_files(base_dir, basename=basename)
    if not files:
        raise FileNotFoundError(f"No COLVAR files matching '{basename}' found in {base_dir}")
    total_files = len(files)
    if verbose:
        print(f"Found {total_files} COLVAR file(s) matching '{basename}'")
        print(f"Loading COLVAR files: 0/{total_files}", end="\r", flush=True)

    header_lines, fields, data_lines_first = _read_header_and_count(files[0])
    if not header_lines:
        raise RuntimeError("Failed to read COLVAR header")
    discard_count = int(data_lines_first * discard_fraction)

    raw_indexed: dict[int, list[str]] = {} if keep_order else {}
    raw_concat: list[str] = [] if not keep_order else []
    valid_sources: list[Path] = []

    for idx, path in enumerate(files, start=1):
        # Validate header matches first file
        hdr_curr, fields_curr, _ = _read_header_and_count(path)
        if not hdr_curr or fields_curr != fields:
            continue
        with open_text_file(str(path)) as handle:
            for _ in range(len(header_lines)):
                next(handle, None)
            for _ in range(discard_count):
                next(handle, None)
            if keep_order:
                for line_idx, line in enumerate(handle):
                    if line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) != len(fields):
                        continue
                    if not line.endswith("\n"):
                        line += "\n"
                    raw_indexed.setdefault(line_idx, []).append(line)
            else:
                for line in handle:
                    if line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) != len(fields):
                        continue
                    if not line.endswith("\n"):
                        line += "\n"
                    raw_concat.append(line)
        valid_sources.append(path)
        if verbose:
            print(f"Loading COLVAR files: {idx}/{total_files}", end="\r", flush=True)

    if verbose:
        print(f"Loading COLVAR files: {len(valid_sources)}/{total_files} (done)          ")

    if keep_order:
        merged_lines: list[str] = []
        for idx in sorted(raw_indexed.keys()):
            merged_lines.extend(raw_indexed[idx])
    else:
        merged_lines = raw_concat

    if not merged_lines:
        raise RuntimeError("No valid COLVAR data found to merge.")

    row_count = len(merged_lines)
    merged_df: pd.DataFrame
    time_col = None
    if build_dataframe:
        arrays = [np.fromstring(l, sep=" ") for l in merged_lines]
        merged_data = np.vstack(arrays)
        merged_df = pd.DataFrame(merged_data, columns=fields, copy=False)
        time_col = "time" if "time" in merged_df.columns else None
    else:
        merged_df = pd.DataFrame(columns=fields)
        time_col = "time" if "time" in fields else None

    if time_ordered and time_col is None:
        time_col = merged_df.columns[0]

    if build_dataframe and time_ordered and time_col is not None:
        merged_df = merged_df.sort_values(time_col, kind="mergesort", ignore_index=True)

    if build_dataframe and keep_order and not time_ordered:
        merged_df = merged_df.reset_index(drop=True)

    if output_path:
        if verbose:
            print("Writing merged COLVAR...", end="\r", flush=True)
        _write_colvar(Path(output_path), header_lines, merged_df if build_dataframe else merged_lines)
        if verbose:
            print(f"Wrote merged COLVAR to {output_path}          ")

    return MergeResult(
        dataframe=merged_df,
        fields=fields,
        header_lines=header_lines,
        source_files=valid_sources,
        time_column=time_col,
        row_count=row_count,
    )


def _write_colvar(path: Path, header_lines: list[str], df: pd.DataFrame | list[str]) -> None:
    """
    Write merged COLVAR data using the provided header lines.
    """
    with open(path, "w", encoding="utf-8") as out:
        for line in header_lines:
            out.write(line if line.endswith("\n") else f"{line}\n")
        if isinstance(df, list):
            out.writelines(df)
        else:
            np.savetxt(out, df.to_numpy(), fmt="%.10g")
