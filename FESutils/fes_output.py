
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import os
import shutil

from .fes_state import GridRuntimeState


@dataclass
class OutputOptions:
    fmt: str
    mintozero: bool
    calc_deltaF: bool
    delta_threshold: float | None
    kbt: float
    backup: bool = True


@dataclass
class SampleStats:
    size: int
    effective_size: float
    blocks_num: int | None = None
    blocks_effective: float | None = None
    delta_f: float | None = None


def write_standard_output(
    outfilename: str,
    grid_state: GridRuntimeState,
    names: tuple[str, str | None],
    options: OutputOptions,
    stats: SampleStats,
    mesh: tuple[NDArray, NDArray] | None = None,
) -> float | None:
    dim2 = grid_state.axis_y is not None
    axis_x = grid_state.axis_x
    axis_y = grid_state.axis_y
    fes = grid_state.fes
    der_x = grid_state.der_fes_x
    der_y = grid_state.der_fes_y
    shift = np.amin(fes) if options.mintozero else 0
    delta = None
    if options.calc_deltaF and options.delta_threshold is not None:
        delta = _compute_delta_f(fes, grid_state, options, mesh)
        stats.delta_f = delta
    if options.backup:
        backup_file(outfilename)
    with open(outfilename, "w") as f:
        fields = ["#! FIELDS", names[0]]
        if dim2 and names[1] is not None:
            fields.append(names[1])
        fields.append("file.free")
        if der_x is not None:
            fields.append(f"der_{names[0]}")
            if dim2 and names[1] is not None and der_y is not None:
                fields.append(f"der_{names[1]}")
        f.write(f'{" ".join(fields)}\n')
        f.write(f"#! SET sample_size {stats.size}\n")
        f.write(f"#! SET effective_sample_size {stats.effective_size}\n")
        if options.calc_deltaF and delta is not None:
            f.write(f"#! SET DeltaF {delta}\n")
        f.write(f"#! SET min_{names[0]} {axis_x.minimum}\n")
        f.write(f"#! SET max_{names[0]} {axis_x.maximum}\n")
        f.write(f"#! SET nbins_{names[0]} {axis_x.bins}\n")
        f.write(
            f'#! SET periodic_{names[0]} {"true" if axis_x.period!=0 else "false"}\n'
        )
        fmt = options.fmt
        if not dim2:
            for i in range(axis_x.bins):
                line = (f"{fmt}  {fmt}") % (axis_x.values[i], fes[i] - shift)
                if der_x is not None:
                    line += (f" {fmt}") % (der_x[i])
                f.write(f"{line}\n")
        else:
            assert axis_y is not None
            f.write(f"#! SET min_{names[1]} {axis_y.minimum}\n")
            f.write(f"#! SET max_{names[1]} {axis_y.maximum}\n")
            f.write(f"#! SET nbins_{names[1]} {axis_y.bins}\n")
            f.write(
                f'#! SET periodic_{names[1]} {"true" if axis_y.period!=0 else "false"}\n'
            )

            stored_mesh = mesh if mesh is not None else grid_state.mesh
            if stored_mesh is not None:
                x_mesh, y_mesh = stored_mesh
            else:
                x_mesh, y_mesh = np.meshgrid(
                    axis_x.values, axis_y.values, indexing="ij"
                )
        
            for i in range(axis_x.bins):
                for j in range(axis_y.bins):
                    line = (f"{fmt} {fmt}  {fmt}") % (
                        x_mesh[i, j],
                        y_mesh[i, j],
                        fes[i, j] - shift,
                    )
                    if der_x is not None and der_y is not None:
                        line += (f" {fmt} {fmt}") % (der_x[i, j], der_y[i, j])
                    f.write(f"{line}\n")
                f.write("\n")
    return delta


def write_block_output(
    outfilename: str,
    grid_state: GridRuntimeState,
    names: tuple[str, str | None],
    options: OutputOptions,
    stats: SampleStats,
    fes_err: NDArray,
    mesh: tuple[NDArray, NDArray] | None = None,
) -> float | None:
    dim2 = grid_state.axis_y is not None
    axis_x = grid_state.axis_x
    axis_y = grid_state.axis_y
    fes = grid_state.fes
    shift = np.amin(fes) if options.mintozero else 0
    delta = None
    if options.calc_deltaF and options.delta_threshold is not None:
        delta = _compute_delta_f(fes, grid_state, options, mesh)
        stats.delta_f = delta
    if options.backup:
        backup_file(outfilename)
    with open(outfilename, "w") as f:
        fields = ["#! FIELDS", names[0]]
        if dim2 and names[1] is not None:
            fields.append(names[1])
        fields.append("file.free uncertainty")
        f.write(f'{" ".join(fields)}\n')
        f.write(f"#! SET sample_size {stats.size}\n")
        f.write(f"#! SET effective_sample_size {stats.effective_size}\n")
        if options.calc_deltaF and delta is not None:
            f.write(f"#! SET DeltaF {delta}\n")
        if stats.blocks_num is not None:
            f.write(f"#! SET blocks_num {stats.blocks_num}\n")
        if stats.blocks_effective is not None:
            f.write(f"#! SET blocks_effective_num {stats.blocks_effective}\n")
        f.write(f"#! SET min_{names[0]} {axis_x.minimum}\n")
        f.write(f"#! SET max_{names[0]} {axis_x.maximum}\n")
        f.write(f"#! SET nbins_{names[0]} {axis_x.bins}\n")
        f.write(
            f'#! SET periodic_{names[0]} {"true" if axis_x.period!=0 else "false"}\n'
        )
        fmt = options.fmt
        if not dim2:
            for i in range(axis_x.bins):
                line = (f"{fmt}  {fmt} {fmt}") % (
                    axis_x.values[i],
                    fes[i] - shift,
                    fes_err[i],
                )
                f.write(f"{line}\n")
        else:
            assert axis_y is not None
            f.write(f"#! SET min_{names[1]} {axis_y.minimum}\n")
            f.write(f"#! SET max_{names[1]} {axis_y.maximum}\n")
            f.write(f"#! SET nbins_{names[1]} {axis_y.bins}\n")
            f.write(
                f'#! SET periodic_{names[1]} {"true" if axis_y.period!=0 else "false"}\n'
            )

            stored_mesh = mesh if mesh is not None else grid_state.mesh
            if stored_mesh is not None:
                x_mesh, y_mesh = stored_mesh
            else:
                x_mesh, y_mesh = np.meshgrid(
                    axis_x.values, axis_y.values, indexing="ij"
                )

            for i in range(axis_x.bins):
                for j in range(axis_y.bins):
                    line = (f"{fmt} {fmt}  {fmt} {fmt}") % (
                        x_mesh[i, j],
                        y_mesh[i, j],
                        fes[i, j] - shift,
                        fes_err[i, j],
                    )
                    f.write(f"{line}\n")
                f.write("\n")
    return delta


def _compute_delta_f(
    fes: NDArray,
    grid_state: GridRuntimeState,
    options: OutputOptions,
    mesh: tuple[NDArray, NDArray] | None,
) -> float:
    kbt = options.kbt
    threshold = options.delta_threshold
    if threshold is None:
        return 0.0
    if grid_state.axis_y is None:
        mask = grid_state.axis_x.values < threshold
        fesA = -kbt * np.logaddexp.reduce(-1 / kbt * fes[mask])
        fesB = -kbt * np.logaddexp.reduce(-1 / kbt * fes[~mask])
    else:
        if mesh is not None:
            x_mesh = mesh[0]
        else:
            stored = grid_state.mesh
            if stored is None:
                x_mesh, _ = np.meshgrid(
                    grid_state.axis_x.values, grid_state.axis_y.values, indexing="ij"
                )
            else:
                x_mesh = stored[0]
        mask = x_mesh < threshold
        fesA = -kbt * np.logaddexp.reduce(-1 / kbt * fes[mask])
        fesB = -kbt * np.logaddexp.reduce(-1 / kbt * fes[~mask])
    return fesB - fesA


def backup_file(path: str) -> None:
    """Backup the file if it exists, using PLUMED-like numbering scheme."""
    if not os.path.exists(path):
        return

    # Find the next available backup number
    n = 1
    while True:
        bck_path = f"{path}.{n}"
        if not os.path.exists(bck_path):
            break
        n += 1

    print(f" backing up {path} to {bck_path}")
    shutil.move(path, bck_path)
