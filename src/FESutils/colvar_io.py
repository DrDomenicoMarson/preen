from dataclasses import dataclass
from collections.abc import Sequence
import os

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import tarfile
import gzip
import io
import contextlib
from typing import IO, Generator

from .fes_config import FESConfig
from .constants import ERROR_PREFIX, energy_conversion_factor


@dataclass
class CVInfo:
    name: str
    column: int
    period: float = 0.0
    grid_min: float | None = None
    grid_max: float | None = None


@dataclass
class BiasInfo:
    columns: list[int]
    names: list[str]


@dataclass
class ColvarMetadata:
    cvs: tuple[CVInfo, ...]
    bias: BiasInfo
    header_lines: int


@dataclass
class ColvarData:
    metadata: ColvarMetadata
    cv_values: tuple[NDArray, ...]
    bias: NDArray


def load_colvar_data(config: FESConfig, merge_result=None) -> ColvarData:
    """
    Read COLVAR data according to the user configuration.
    If merge_result (from colvar_merge.MergeResult) is provided, use the in-memory
    dataframe and header instead of reading from disk.
    """
    # Lazy import to avoid circular dependency
    from .colvar_merge import MergeResult

    if merge_result is None:
        with open_text_file(config.filename) as f:
            fields = f.readline().split()
            if len(fields) < 2 or fields[1] != "FIELDS":
                raise ValueError(f'{ERROR_PREFIX} no FIELDS found in "{config.filename}"')
            cv_infos = _resolve_cv_infos(fields, config.cv_spec)
            bias_info = _resolve_bias_info(fields, config.bias_spec)
            header_lines = _parse_header_metadata(f, cv_infos, config.calc_der)
            metadata = ColvarMetadata(
                cvs=tuple(cv_infos), bias=bias_info, header_lines=header_lines
            )
        skip_rows = metadata.header_lines + config.skiprows
        required_cols = [info.column for info in metadata.cvs] + metadata.bias.columns
        required_cols = sorted(set(required_cols))
        with open_text_file(config.filename) as f_data:
            data = pd.read_table(
                f_data,
                dtype=float,
                sep=r"\s+",
                comment="#",
                header=None,
                usecols=required_cols,
                skiprows=skip_rows,
            )
    else:
        if not isinstance(merge_result, MergeResult):
            raise TypeError("merge_result must be a MergeResult or None")
        fields = list(merge_result.fields)
        cv_infos = _resolve_cv_infos(fields, config.cv_spec)
        bias_info = _resolve_bias_info(fields, config.bias_spec)
        header_tail = "".join(merge_result.header_lines[1:]) if len(merge_result.header_lines) > 1 else ""
        header_lines = _parse_header_metadata(io.StringIO(header_tail), cv_infos, config.calc_der)
        metadata = ColvarMetadata(
            cvs=tuple(cv_infos), bias=bias_info, header_lines=header_lines
        )
        data = merge_result.dataframe.copy(deep=False)
        if config.skiprows > 0:
            data = data.iloc[config.skiprows:]
        # Ensure column ordering matches fields and drop any extra
        data = data[fields]

    if data.isnull().values.any():
        raise ValueError(
            f"{ERROR_PREFIX} your COLVAR file contains NaNs. Check if last line is truncated"
        )
    if config.reverse:
        data = data.iloc[::-1]
    required_cols = [info.column for info in metadata.cvs] + metadata.bias.columns
    required_cols = sorted(set(required_cols))
    col_names = [fields[idx] for idx in required_cols] if merge_result is not None else required_cols
    sub_data = data[col_names]
    col_map = {col: idx for idx, col in enumerate(required_cols)}
    cv_arrays = []
    for info in metadata.cvs:
        values = np.ascontiguousarray(
            np.array(sub_data.iloc[:, col_map[info.column]], dtype=float)
        )
        cv_arrays.append(values)
    bias = np.zeros(len(cv_arrays[0]))
    for col in metadata.bias.columns:
        bias += np.array(sub_data.iloc[:, col_map[col]], dtype=float)
    energy_factor = energy_conversion_factor(config.input_energy_unit, "kJ/mol")
    bias = np.ascontiguousarray((bias * energy_factor) / config.kbt)
    return ColvarData(metadata=metadata, cv_values=tuple(cv_arrays), bias=bias)


def _resolve_cv_infos(
    fields: Sequence[str], cv_spec: Sequence[str]
) -> tuple[CVInfo, ...]:
    infos = []
    for spec in cv_spec:
        column, name = _resolve_single_cv(fields, spec.strip())
        print(f' using cv "{name}" found at column {column+1}')
        infos.append(CVInfo(name=name, column=column))
    return tuple(infos)


def _resolve_single_cv(fields: Sequence[str], spec: str) -> tuple[int, str]:
    try:
        idx = int(spec) - 1
        if idx < 0:
            raise ValueError
        if idx + 2 >= len(fields):
            raise ValueError
        name = fields[idx + 2]
        return idx, name
    except ValueError:
        target = spec
        for pos, field in enumerate(fields):
            if field == target:
                return pos - 2, target
        raise ValueError(f'{ERROR_PREFIX} cv "{spec}" not found')


def _resolve_bias_info(fields: Sequence[str], bias_spec: str) -> BiasInfo:
    if bias_spec.lower() == "no":
        columns = []
        names = []
    else:
        tokens = [token.strip() for token in bias_spec.split(",") if token.strip()]
        columns = []
        names = []
        try:
            columns = [int(token) - 1 for token in tokens]
            names = []
            for col in columns:
                if col < 0 or col + 2 >= len(fields):
                    raise ValueError(
                        f"{ERROR_PREFIX} bias column {col+1} is out of range"
                    )
                names.append(fields[col + 2])
        except ValueError:
            columns = []
            names = []
            if bias_spec == ".bias":
                for pos, field in enumerate(fields):
                    if field.find(".bias") != -1 or field.find(".rbias") != -1:
                        columns.append(pos - 2)
                        names.append(field)
            else:
                for token in tokens:
                    found = False
                    for pos, field in enumerate(fields):
                        if field == token:
                            columns.append(pos - 2)
                            names.append(field)
                            found = True
                            break
                    if not found:
                        raise ValueError(f'{ERROR_PREFIX} bias "{token}" not found')
                if len(columns) != len(tokens):
                    raise ValueError(
                        f"{ERROR_PREFIX} found {len(columns)} matching biases, but {len(tokens)} were requested. Use columns number to avoid ambiguity"
                    )
    if not columns:
        print(" no bias")
    else:
        for col, name in zip(columns, names):
            print(f' using bias "{name}" found at column {col+1}')
    return BiasInfo(columns=columns, names=names)


def _parse_header_metadata(handle, cv_infos: Sequence[CVInfo], calc_der: bool) -> int:
    header_lines = 1
    name_map: dict[str, CVInfo] = {info.name: info for info in cv_infos}
    line = handle.readline().split()
    while line and line[0] == "#!":
        header_lines += 1
        if len(line) >= 4 and line[2].startswith("min_"):
            cv_name = line[2][4:]
            info = name_map.get(cv_name)
            if info is not None:
                info.grid_min = _convert_token(line[3])
                next_line = handle.readline().split()
                header_lines += 1
                if len(next_line) < 4 or next_line[2] != f"max_{cv_name}":
                    raise ValueError(
                        f"{ERROR_PREFIX} min_{cv_name} was found, but not max_{cv_name} !"
                    )
                info.grid_max = _convert_token(next_line[3])
                info.period = info.grid_max - info.grid_min
                if calc_der:
                    raise ValueError(
                        f"{ERROR_PREFIX} derivatives not supported with periodic CVs, remove --der option"
                    )
                line = handle.readline().split()
                continue
        line = handle.readline().split()
    return header_lines


def _convert_token(value: str) -> float:
    if value == "-pi":
        return -np.pi
    if value == "pi":
        return np.pi
    return float(value)


@contextlib.contextmanager
def open_text_file(filename: str) -> Generator[IO[str], None, None]:
    """
    Open a file as text, handling compression transparently.
    Supports .tgz, .tar.gz, .gz, and plain text.
    For tar files, opens the first member.
    """
    if filename.endswith(".tgz") or filename.endswith(".tar.gz"):
        with tarfile.open(filename, "r:*") as tar:
            # Collect regular files, skipping macOS metadata files (._*)
            members = [
                m
                for m in tar.getmembers()
                if m.isfile() and not os.path.basename(m.name).startswith("._")
            ]
            if not members:
                raise ValueError(f"{ERROR_PREFIX} no valid file found in archive {filename}")
            if len(members) > 1:
                names = ", ".join(os.path.basename(m.name) for m in members[:5])
                extra = "" if len(members) <= 5 else ", ..."
                raise ValueError(
                    f"{ERROR_PREFIX} multiple files found in archive {filename}: {names}{extra}. "
                    "Only single-file archives are supported."
                )
            member = members[0]
            f = tar.extractfile(member)
            if f is None:
                 raise ValueError(f"{ERROR_PREFIX} could not extract {member.name}")
            
            # Wrap in TextIOWrapper
            with io.TextIOWrapper(f, encoding="utf-8") as text_f:
                yield text_f
    elif filename.endswith(".gz"):
        with gzip.open(filename, "rt", encoding="utf-8") as f:
            yield f
    else:
        with open(filename, "r", encoding="utf-8") as f:
            yield f
