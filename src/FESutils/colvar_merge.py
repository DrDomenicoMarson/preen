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
    raw_lines: list[str] | None = None


@dataclass
class MergeTextResult:
    lines: list[str]
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
    header_lines, fields, data_lines = _read_header_and_count(path)
    if not header_lines:
        return None
    if expected_fields is not None and list(expected_fields) != fields:
        return None

    drop = int(data_lines * discard_fraction) if discard_fraction > 0 else 0
    if drop >= data_lines:
        return None

    # Fast C-engine read; skip malformed rows
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=fields,
        usecols=range(len(fields)),
        skiprows=len(header_lines) + drop,
        engine="c",
        on_bad_lines="skip",
    )
    if df.empty:
        return None
    return header_lines, fields, df


def merge_colvar_lines(
    base_dir: str | Path,
    basename: str = "COLVAR",
    discard_fraction: float = 0.1,
    time_ordered: bool = True,
    output_path: str | Path | None = None,
    verbose: bool = True,
    allow_header_mismatch: bool = False,
) -> MergeTextResult:
    """
    Merge COLVAR files into in-memory lines (string only; no numeric parsing).
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

    header_lines, fields, data_lines_first = _read_header_and_count(files[0])
    if not header_lines:
        raise RuntimeError("Failed to read COLVAR header")

    valid_sources: list[Path] = []
    total_seen = 0
    total_discarded = 0
    total_malformed = 0
    warned_headers: set[Path] = set()

    interleave_pairs: list[tuple[int, int, str]] = [] if time_ordered else []
    raw_concat: list[str] = [] if not time_ordered else []

    streaming_path = output_path is not None and not time_ordered
    out_handle = None
    if streaming_path:
        out_handle = open(output_path, "w", encoding="utf-8")
        for line in header_lines:
            out_handle.write(line if line.endswith("\n") else f"{line}\n")

    for file_idx, path in enumerate(files):
        hdr_curr, fields_curr, data_lines = _read_header_and_count(path)
        if not hdr_curr or fields_curr != fields:
            continue
        if len(hdr_curr) != len(header_lines):
            if not allow_header_mismatch:
                raise ValueError(
                    f"Header length differs in file {path}: expected {len(header_lines)} line(s), "
                    f"found {len(hdr_curr)}. Use allow_header_mismatch=True to proceed."
                )
            if path not in warned_headers and verbose:
                print(
                    f"Warning: header length differs in {path} "
                    f"(expected {len(header_lines)}, found {len(hdr_curr)})"
                )
                warned_headers.add(path)
        discard_count = int(data_lines * discard_fraction)
        with open_text_file(str(path)) as handle:
            for _ in range(len(hdr_curr)):
                next(handle, None)
            for _ in range(discard_count):
                if next(handle, None) is not None:
                    total_discarded += 1
            for line_idx, line in enumerate(handle):
                total_seen += 1
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != len(fields):
                    total_malformed += 1
                    continue
                if not line.endswith("\n"):
                    line += "\n"
                if time_ordered:
                    interleave_pairs.append((line_idx, file_idx, line))
                else:
                    if streaming_path and out_handle is not None:
                        out_handle.write(line)
                    else:
                        raw_concat.append(line)
        valid_sources.append(path)
        if verbose:
            print(f"Loading COLVAR files: {len(valid_sources)}/{total_files}", end="\r", flush=True)

    if verbose:
        print(f"Loading COLVAR files: {len(valid_sources)}/{total_files} (done)          ")
        print(
            f"Lines read: {total_seen + total_discarded}, kept: {total_seen - total_malformed - total_discarded}, "
            f"discarded (fractional skip): {total_discarded}, malformed: {total_malformed}"
        )

    if streaming_path:
        if out_handle is not None:
            out_handle.flush()
            out_handle.close()
        time_col = "time" if "time" in fields else None
        return MergeTextResult(
            lines=[],
            fields=fields,
            header_lines=header_lines,
            source_files=valid_sources,
            time_column=time_col,
            row_count=total_seen - total_malformed - total_discarded,
        )

    if time_ordered:
        interleave_pairs.sort()
        merged_lines = [line for _, _, line in interleave_pairs]
    else:
        merged_lines = raw_concat

    if not merged_lines:
        raise RuntimeError("No valid COLVAR data found to merge.")

    row_count = len(merged_lines)
    time_col = "time" if "time" in fields else None

    return MergeTextResult(
        lines=merged_lines,
        fields=fields,
        header_lines=header_lines,
        source_files=valid_sources,
        time_column=time_col,
        row_count=row_count,
    )


def build_dataframe_from_lines(
    lines: list[str],
    fields: Sequence[str],
    time_ordered: bool,
    requested_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """
    Convert merged lines (strings) into a numeric dataframe.
    """
    if requested_columns is not None:
        missing = [c for c in requested_columns if c not in fields]
        if missing:
            available_str = ", ".join(fields)
            raise ValueError(
                f"Requested columns not found: {', '.join(missing)}. Available columns: {available_str}"
            )
        keep_indices = [fields.index(c) for c in requested_columns]
        out_columns = list(requested_columns)
    else:
        keep_indices = list(range(len(fields)))
        out_columns = list(fields)

    arrays = [np.fromstring(l, sep=" ") for l in lines]
    merged_data = np.vstack(arrays)
    if keep_indices != list(range(merged_data.shape[1])):
        merged_data = merged_data[:, keep_indices]
    merged_df = pd.DataFrame(merged_data, columns=out_columns, copy=False)
    time_col = "time" if "time" in merged_df.columns else None
    if time_ordered and time_col is None:
        time_col = merged_df.columns[0]
    if time_ordered and time_col is not None:
        merged_df = merged_df.reset_index(drop=True)
    return merged_df, time_col


def merge_colvar_files(
    base_dir: str | Path,
    basename: str = "COLVAR",
    discard_fraction: float = 0.1,
    time_ordered: bool = True,
    output_path: str | Path | None = None,
    verbose: bool = True,
    build_dataframe: bool = False,
    allow_header_mismatch: bool = False,
    requested_columns: Sequence[str] | None = None,
) -> MergeResult:
    """
    Merge COLVAR files located under base_dir.

    - discard_fraction: drop that fraction of valid rows from the start of each file.
    - time_ordered: interleave lines by index across files (round-robin). If False, concatenate files.
    - output_path: if provided, write merged data with the first header.
    - verbose: print progress information while loading files.
    - build_dataframe: build numeric dataframe (set False to speed up merge-only CLI).
    - allow_header_mismatch: allow files whose header line count differs from the first file (warns once per file).
    - requested_columns: optional subset of column names to keep when building the dataframe.
    """
    if not (0.0 <= discard_fraction <= 1.0):
        raise ValueError("discard_fraction must be between 0.0 and 1.0")
    text_result = merge_colvar_lines(
        base_dir=base_dir,
        basename=basename,
        discard_fraction=discard_fraction,
        time_ordered=time_ordered,
        output_path=output_path,
        verbose=verbose,
        allow_header_mismatch=allow_header_mismatch,
    )

    if not build_dataframe:
        if output_path:
            if verbose:
                print("Writing merged COLVAR...", end="\r", flush=True)
            _write_colvar(Path(output_path), text_result.header_lines, text_result.lines)
            if verbose:
                print(f"Wrote merged COLVAR to {output_path}          ")
        return MergeResult(
            dataframe=pd.DataFrame(columns=text_result.fields),
            fields=text_result.fields,
            header_lines=text_result.header_lines,
            source_files=text_result.source_files,
            time_column="time" if "time" in text_result.fields else None,
            row_count=text_result.row_count,
            raw_lines=text_result.lines,
        )

    merged_df, time_col = build_dataframe_from_lines(
        text_result.lines,
        text_result.fields,
        time_ordered=time_ordered,
        requested_columns=requested_columns,
    )

    if output_path:
        if verbose:
            print("Writing merged COLVAR...", end="\r", flush=True)
        _write_colvar(Path(output_path), text_result.header_lines, merged_df)
        if verbose:
            print(f"Wrote merged COLVAR to {output_path}          ")

    return MergeResult(
        dataframe=merged_df,
        fields=text_result.fields if requested_columns is None else requested_columns,
        header_lines=text_result.header_lines,
        source_files=text_result.source_files,
        time_column=time_col,
        row_count=text_result.row_count,
        raw_lines=text_result.lines,
    )


def _write_colvar(path: Path, header_lines: list[str], df: pd.DataFrame | list[str]) -> None:
    """
    Write merged COLVAR data using the provided header lines.
    """
    if not header_lines or not header_lines[0].lstrip().startswith("#! FIELDS"):
        raise ValueError("Invalid COLVAR header: first line must start with '#! FIELDS'")
    with open(path, "w", encoding="utf-8") as out:
        for line in header_lines:
            out.write(line if line.endswith("\n") else f"{line}\n")
        if isinstance(df, list):
            out.writelines(df)
        else:
            np.savetxt(out, df.to_numpy(), fmt="%.10g")
