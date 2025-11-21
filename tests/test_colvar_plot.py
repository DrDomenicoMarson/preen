from pathlib import Path

import pytest

from FESutils.colvar_plot import plot_colvar_timeseries


def _write_colvar(path: Path, rows: list[str]) -> None:
    header = [
        "#! FIELDS time cv1 cv2 .bias\n",
        "#! SET min_cv1 -1.0\n",
        "#! SET max_cv1 1.0\n",
    ]
    with open(path, "w", encoding="utf-8") as f:
        for line in header + rows:
            f.write(line if line.endswith("\n") else f"{line}\n")


def test_plot_timeseries_creates_files(tmp_path):
    d0 = tmp_path / "0"
    d1 = tmp_path / "1"
    d0.mkdir()
    d1.mkdir()
    _write_colvar(
        d0 / "COLVAR",
        [
            "0 0.0 0.0 0.0",
            "1 0.1 0.2 0.3",
        ],
    )
    _write_colvar(
        d1 / "COLVAR.1",
        [
            "0 0.5 0.4 0.1",
            "1 0.6 0.5 0.2",
        ],
    )

    out = tmp_path / "agg.png"
    outputs = plot_colvar_timeseries(
        base_dir=tmp_path,
        output_path=out,
        per_run=True,
        include_hist=True,
    )
    assert "aggregate" in outputs
    assert outputs["aggregate"].exists()
    assert "aggregate_hist" in outputs
    assert outputs["aggregate_hist"].exists()
    # Per-run outputs should be created next to source files
    per_run_paths = [p for k, p in outputs.items() if k not in ("aggregate", "aggregate_hist")]
    assert len(per_run_paths) == 4  # two timeseries + two hist
    for p in per_run_paths:
        assert p.exists(), f"Missing per-run plot {p}"


def test_plot_timeseries_requires_valid_columns(tmp_path):
    d0 = tmp_path / "0"
    d0.mkdir()
    _write_colvar(
        d0 / "COLVAR",
        [
            "0 0.0 0.0 0.0",
            "1 0.1 0.2 0.3",
        ],
    )
    with pytest.raises(ValueError):
        plot_colvar_timeseries(base_dir=tmp_path, columns=["nonexistent"], output_path=None)
