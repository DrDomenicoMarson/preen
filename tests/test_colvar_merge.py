import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from FESutils.colvar_merge import merge_colvar_files
from FESutils.colvar_io import load_colvar_data
from FESutils.api import calculate_fes
from FESutils.fes_config import FESConfig
from FESutils.constants import KB_KJ_MOL


def _write_colvar(path: Path, rows: list[str]) -> None:
    header = [
        "#! FIELDS time cv1 .bias\n",
        "#! SET min_cv1 -3.14\n",
        "#! SET max_cv1 3.14\n",
    ]
    with open(path, "w", encoding="utf-8") as f:
        for line in header + rows:
            f.write(line if line.endswith("\n") else f"{line}\n")


def test_merge_drops_malformed_and_discards(tmp_path):
    d0 = tmp_path / "0"
    d1 = tmp_path / "1"
    d0.mkdir()
    d1.mkdir()
    _write_colvar(
        d0 / "COLVAR",
        [
            "0 0.0 0.1",
            "1 0.1 0.2",
            "bad line",
            "2 0.2 0.3",
            "3 0.3 0.4 extra",
            "4 0.4 0.5",
        ],
    )
    _write_colvar(
        d1 / "COLVAR.1",
        [
            "5 0.5 0.6",
            "6 0.6 0.7",
            "oops",
            "7 0.7 0.8",
            "8 0.8 0.9 1.0",
            "9 0.9 1.0",
        ],
    )
    result = merge_colvar_files(
        base_dir=tmp_path,
        discard_fraction=0.5,  # drop half the valid rows from each file
        time_ordered=False,
    )
    # Each file has 4 valid rows; discard_fraction=0.5 keeps 2 per file => 4 total
    assert len(result.dataframe) == 4
    # Natural sort picks 0 then 1 because of numeric suffix
    assert result.source_files[0].name == "COLVAR"
    assert result.source_files[1].name == "COLVAR.1"


def test_merge_time_order_no_stitch(tmp_path):
    d0 = tmp_path / "0"
    d1 = tmp_path / "1"
    d0.mkdir()
    d1.mkdir()
    _write_colvar(
        d0 / "COLVAR",
        [
            "0 0.0 0.1",
            "1 0.1 0.2",
            "2 0.2 0.3",
        ],
    )
    _write_colvar(
        d1 / "COLVAR.1",
        [
            "0 1.0 0.4",  # time resets
            "1 1.1 0.5",
        ],
    )
    result = merge_colvar_files(base_dir=tmp_path, time_ordered=True)
    times = result.dataframe["time"].to_numpy()
    # Sorting should not modify values; just stable sort by time
    assert np.all(times == np.array([0, 0, 1, 1, 2]))
    assert np.all(np.diff(times) >= 0)


def test_invalid_discard_fraction(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    _write_colvar(
        d / "COLVAR",
        [
            "0 0.0 0.0",
            "1 0.1 0.2",
        ],
    )
    with pytest.raises(ValueError):
        merge_colvar_files(base_dir=d, discard_fraction=1.5)


def test_merge_skips_mismatched_headers(tmp_path):
    d0 = tmp_path / "0"
    d1 = tmp_path / "1"
    d0.mkdir()
    d1.mkdir()
    _write_colvar(
        d0 / "COLVAR",
        [
            "0 0.0 0.1",
            "1 0.1 0.2",
        ],
    )
    # Mismatched FIELDS: should be ignored
    with open(d1 / "COLVAR.1", "w", encoding="utf-8") as f:
        f.write("#! FIELDS time cv1 cv2 .bias\n")
        f.write("0 0.0 0.1 0.5\n")
        result = merge_colvar_files(base_dir=tmp_path)
    assert len(result.source_files) == 1
    assert result.source_files[0].name == "COLVAR"


def test_load_colvar_with_merge_result_matches(tmp_path):
    # Prepare a simple COLVAR file
    d = tmp_path / "run"
    d.mkdir()
    _write_colvar(
        d / "COLVAR",
        [
            "0 0.0 0.0",
            "1 0.1 0.0",
            "2 0.2 0.0",
        ],
    )
    result = merge_colvar_files(base_dir=d)
    outfile = tmp_path / "fes.dat"
    config = FESConfig(
        filename="MERGED_IN_MEMORY",
        outfile=str(outfile),
        kbt=300.0 * KB_KJ_MOL,
        grid_bin=(5,),
        sigma=(0.1,),
        cv_spec=("cv1",),
        bias_spec=".bias",
        backup=False,
        plot=False,
    )
    calculate_fes(config, merge_result=result)
    assert outfile.exists()
    data = np.loadtxt(outfile)
    assert data.shape[1] >= 2  # cv, fes
