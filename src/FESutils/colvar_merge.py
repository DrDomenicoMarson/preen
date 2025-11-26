"""
Utilities to collect, validate, and merge PLUMED COLVAR files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import warnings
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


def _merge_run_frames(frames: list[pd.DataFrame], time_ordered: bool, columns: Sequence[str]) -> pd.DataFrame:
    """
    Combine per-run dataframes into a single dataframe following the time_ordered
    semantics used by merge_colvar_files (interleave rows across runs by index).
    """
    if not frames:
        return pd.DataFrame(columns=columns)
    reset_frames = [df.reset_index(drop=True) for df in frames]
    if not time_ordered:
        return pd.concat(reset_frames, axis=0, ignore_index=True, copy=False)

    lengths = [len(df) for df in reset_frames]
    max_len = max(lengths)
    if len(set(lengths)) == 1:
        # Fast path: same length -> vectorized interleave
        arrs = [df.to_numpy() for df in reset_frames]
        stacked = np.stack(arrs, axis=1)  # shape: (rows, runs, cols)
        interleaved = stacked.reshape(-1, stacked.shape[-1])
        return pd.DataFrame(interleaved, columns=columns)

    # Fallback: differing lengths; preserve order row-by-row
    rows = []
    for idx in range(max_len):
        for df in reset_frames:
            if idx < len(df):
                rows.append(df.iloc[idx].to_numpy())
    return pd.DataFrame(rows, columns=columns)


def build_dataframe_from_lines(
    lines: list[str],
    fields: Sequence[str],
    time_ordered: bool,
    requested_columns: Sequence[str] | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, str | None]:
    """
    Convert merged lines (strings) into a numeric dataframe.
    """
    if verbose:
        print(f"Building dataframe from {len(lines)} merged lines...", end="\r", flush=True)
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
    if verbose:
        print(f"Building dataframe from {len(lines)} merged lines...done          ")
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
        verbose=verbose,
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


def merge_multiple_colvar_files(
    base_dir: str | Path,
    basenames: Sequence[str],
    discard_fraction: float = 0.1,
    time_ordered: bool = True,
    verbose: bool = True,
    allow_header_mismatch: bool = False,
    requested_columns: Sequence[str] | None = None,
) -> MergeResult:
    """
    Merge data from multiple COLVAR-like basenames per run (e.g., COLVAR and CV_DIHEDRALS).

    - Files are grouped by their parent directory relative to base_dir.
    - Each run must contain exactly one file for each requested basename.
    - Within a run, dataframes are aligned by row index; overlapping columns must match.
    - The resulting dataframe contains the union of columns (or requested_columns subset).
    """
    if not basenames:
        raise ValueError("At least one basename must be provided")
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    seen: set[str] = set()
    duplicates = [b for b in basenames if b in seen or seen.add(b)]
    basenames = list(dict.fromkeys(basenames))
    if duplicates:
        warnings.warn(
            f"Duplicate basenames provided ({', '.join(duplicates)}); using unique order: {', '.join(basenames)}",
            stacklevel=2,
        )

    runs: dict[Path, dict[str, Path]] = {}
    for bname in basenames:
        files = discover_colvar_files(base, basename=bname)
        if not files:
            raise FileNotFoundError(f"No COLVAR files matching '{bname}' found in {base_dir}")
        for path in files:
            try:
                run_key = path.parent.resolve().relative_to(base.resolve())
            except ValueError:
                run_key = path.parent.resolve()
            run = runs.setdefault(run_key, {})
            if bname in run:
                raise ValueError(
                    f"Multiple files for basename '{bname}' in run '{run_key}': {run[bname]} and {path}"
                )
            run[bname] = path

    missing_runs = {
        rk: [b for b in basenames if b not in files] for rk, files in runs.items() if len(files) != len(basenames)
    }
    if missing_runs:
        msgs = []
        for rk, missing in missing_runs.items():
            msgs.append(f"{rk}: missing {', '.join(missing)}")
        raise FileNotFoundError("Missing basenames in runs: " + "; ".join(msgs))

    run_keys = sorted(runs.keys(), key=_natural_key)
    combined_frames: list[pd.DataFrame] = []
    source_files: list[Path] = []
    header_lines_ref: list[str] | None = None
    combined_fields: list[str] | None = None

    for rk in run_keys:
        per_files = [runs[rk][b] for b in basenames]
        per_dfs: list[pd.DataFrame] = []
        row_lengths: list[int] = []

        for path in per_files:
            res = read_colvar_dataframe(path, expected_fields=None, discard_fraction=discard_fraction)
            if res is None:
                raise RuntimeError(f"Failed to read COLVAR data from {path}")
            hdr, fields, df = res
            if header_lines_ref is None:
                header_lines_ref = hdr
            row_lengths.append(len(df))
            per_dfs.append(df)
            source_files.append(path)

        if not per_dfs:
            continue
        min_len = min(row_lengths)
        max_len = max(row_lengths)
        if max_len != min_len:
            if verbose:
                warnings.warn(
                    f"Row count mismatch in run '{rk}': trimming all to {min_len} rows (min of {row_lengths})",
                    stacklevel=2,
                )
            per_dfs = [df.iloc[:min_len].reset_index(drop=True) for df in per_dfs]
        else:
            per_dfs = [df.reset_index(drop=True) for df in per_dfs]

        combined = per_dfs[0].copy(deep=False)
        for idx in range(1, len(per_dfs)):
            other = per_dfs[idx]
            overlap = [c for c in other.columns if c in combined.columns]
            for col in overlap:
                if not np.array_equal(combined[col].to_numpy(), other[col].to_numpy()):
                    raise ValueError(
                        f"Column '{col}' differs between files in run '{rk}'. Cannot merge basenames."
                    )
            new_cols = [c for c in other.columns if c not in combined.columns]
            if new_cols:
                combined = pd.concat([combined, other[new_cols]], axis=1, copy=False)

        if requested_columns is not None:
            missing = [c for c in requested_columns if c not in combined.columns]
            if missing:
                available_str = ", ".join(combined.columns)
                raise ValueError(
                    f"Requested columns not found after merge: {', '.join(missing)}. Available columns: {available_str}"
                )
            combined = combined[list(requested_columns)]

        combined_frames.append(combined)
        if combined_fields is None:
            combined_fields = list(combined.columns)

    if not combined_frames:
        raise RuntimeError("No data merged from provided basenames")

    assert combined_fields is not None
    merged_df = _merge_run_frames(combined_frames, time_ordered=time_ordered, columns=combined_fields)
    time_col = "time" if "time" in merged_df.columns else (merged_df.columns[0] if time_ordered else None)

    header_lines = []
    fields_line = "#! FIELDS " + " ".join(combined_fields)
    header_lines.append(fields_line if fields_line.endswith("\n") else f"{fields_line}\n")
    if header_lines_ref:
        for line in header_lines_ref[1:]:
            header_lines.append(line if line.endswith("\n") else f"{line}\n")

    return MergeResult(
        dataframe=merged_df,
        fields=combined_fields,
        header_lines=header_lines,
        source_files=source_files,
        time_column=time_col,
        row_count=len(merged_df),
        raw_lines=None,
    )


def merge_runs_multiple_colvar_files(
    run_dirs: Sequence[str | Path],
    basenames: Sequence[str],
    discard_fractions: float | Sequence[float] = 0.1,
    time_ordered: bool = True,
    verbose: bool = True,
    allow_header_mismatch: bool = False,
    requested_columns: Sequence[str] | None = None,
) -> MergeResult:
    """
    Merge multiple basenames across multiple run directories (e.g., run_1 and run_2).

    - `run_dirs`: list of directories containing walker subdirs/files for each run.
    - `basenames`: basenames to merge per run (e.g., ["COLVAR", "CV_DIHEDRALS"]).
    - `discard_fractions`: single float for all runs or per-run list matching run_dirs.
    - `time_ordered`: interleave rows across runs by index (handles differing lengths).
    - `requested_columns`: optional subset of columns to keep.
    """
    if not run_dirs:
        raise ValueError("At least one run directory must be provided")
    if not basenames:
        raise ValueError("At least one basename must be provided")

    if isinstance(discard_fractions, (int, float)):
        discards = [float(discard_fractions)] * len(run_dirs)
    else:
        discards = list(discard_fractions)
        if len(discards) != len(run_dirs):
            raise ValueError("discard_fractions must be a single value or match run_dirs length")

    run_results: list[MergeResult] = []
    for run_dir, disc in zip(run_dirs, discards):
        if verbose:
            print(f"Merging run at {run_dir} (discard_fraction={disc})")
        result = merge_multiple_colvar_files(
            base_dir=run_dir,
            basenames=basenames,
            discard_fraction=disc,
            time_ordered=time_ordered,
            verbose=verbose,
            allow_header_mismatch=allow_header_mismatch,
            requested_columns=requested_columns,
        )
        run_results.append(result)

    ref_fields = run_results[0].fields
    for res in run_results[1:]:
        if list(res.fields) != list(ref_fields):
            raise ValueError("Merged runs have differing fields; ensure basenames/columns align.")

    frames = [res.dataframe.reset_index(drop=True) for res in run_results]
    merged_df = _merge_run_frames(frames, time_ordered=time_ordered, columns=list(ref_fields))
    time_col = "time" if "time" in merged_df.columns else (merged_df.columns[0] if time_ordered else None)

    header_lines = []
    fields_line = "#! FIELDS " + " ".join(ref_fields)
    header_lines.append(fields_line if fields_line.endswith("\n") else f"{fields_line}\n")
    if run_results[0].header_lines:
        for line in run_results[0].header_lines[1:]:
            header_lines.append(line if line.endswith("\n") else f"{line}\n")

    source_files: list[Path] = []
    for res in run_results:
        source_files.extend(res.source_files)

    return MergeResult(
        dataframe=merged_df,
        fields=list(ref_fields),
        header_lines=header_lines,
        source_files=source_files,
        time_column=time_col,
        row_count=len(merged_df),
        raw_lines=None,
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
