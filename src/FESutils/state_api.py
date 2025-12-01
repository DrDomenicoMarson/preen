import os
import numpy as np
import pandas as pd
from numba import set_num_threads

from .constants import ERROR_PREFIX, energy_conversion_factor
from .colvar_io import open_text_file
from .fes_config import FESStateConfig
from .grid import GridAxis, GridData
from .fes_state import create_grid_runtime_state, symmetrize_grid_state
from .kernel_eval import (
    _numba_calc_prob_1d,
    _numba_calc_der_prob_1d,
    _numba_calc_prob_2d,
    _numba_calc_der_prob_2d,
)
from .fes_output import OutputOptions, SampleStats, write_standard_output
from .fes_plot import PlotManager


def calculate_fes_from_state(config: FESStateConfig):
    """
    Calculate FES from STATE file (sum of kernels).
    """
    if not isinstance(config, FESStateConfig):
        raise TypeError("calculate_fes_from_state expects an FESStateConfig")
    if config.input_file is None:
        raise ValueError("input_file is required for STATE-based FES calculation")

    set_num_threads(config.num_threads)
    data = _read_state_file(config.input_file)
    blocks = _find_field_blocks(data, config.input_file)
    output_conv = energy_conversion_factor("kJ/mol", config.output_energy_unit)

    for idx, (start, end) in enumerate(blocks, start=1):
        total_states = len(blocks)
        print(f"   working... state {idx}/{total_states}", end="\r")
        header, meta, header_end = _parse_state_header(data, start, config.input_file)
        meta, data_start = _parse_periodicity(data, header_end, end, header, meta)
        kernels = _extract_kernels(data, data_start, end, meta.dim2, config.input_file, config, meta)
        grid = _build_grid_from_state(config, meta, kernels)
        grid_state, stats = _evaluate_state_fes(config, meta, kernels, grid)
        
        # Symmetrize grid state if needed
        if config.symmetrize_cvs:
            grid_state = symmetrize_grid_state(grid_state, config.symmetrize_cvs)
            
        _write_state_outputs(config, meta, grid_state, stats, grid_state.mesh, output_conv)
    print("Done.")


def _read_state_file(filename: str) -> np.ndarray:
    try:
        with open_text_file(filename) as f:
            data_df = pd.read_table(f, sep=r"\s+", header=None, dtype=str, comment=None)
        return data_df.to_numpy()
    except Exception as e:
        raise RuntimeError(f"{ERROR_PREFIX} failed to read {filename}: {e}")


def _find_field_blocks(data: np.ndarray, filename: str) -> list[tuple[int, int]]:
    fields_pos = [i for i in range(data.shape[0]) if data[i, 1] == "FIELDS"]
    if not fields_pos:
        raise ValueError(f"{ERROR_PREFIX} no FIELDS found in file {filename}")
    if len(fields_pos) > 1:
        print(f" a total of {len(fields_pos)} stored states were found")
        print("  -> only the last one will be printed.")
        fields_pos = [fields_pos[-1]]
    fields_pos.append(data.shape[0])
    return [(fields_pos[i], fields_pos[i + 1]) for i in range(len(fields_pos) - 1)]


class _StateMeta:
    def __init__(self, dim2: bool, name_cv_x: str, name_cv_y: str | None, sf: float, epsilon: float, cutoff: float, val_at_cutoff: float, zed: float):
        self.dim2 = dim2
        self.name_cv_x = name_cv_x
        self.name_cv_y = name_cv_y
        self.sf = sf
        self.epsilon = epsilon
        self.cutoff = cutoff
        self.val_at_cutoff = val_at_cutoff
        self.zed = zed
        self.period_x = 0.0
        self.period_y = 0.0
        self.grid_min_x_read = None
        self.grid_max_x_read = None
        self.grid_min_y_read = None
        self.grid_max_y_read = None


def _parse_state_header(data: np.ndarray, start: int, filename: str) -> tuple[list[str], _StateMeta, int]:
    valid_row = [x for x in data[start, :] if isinstance(x, str)]
    if len(valid_row) == 6:
        dim2 = False
        name_cv_x = valid_row[3]
        name_cv_y = None
    elif len(valid_row) == 8:
        dim2 = True
        name_cv_x = valid_row[3]
        name_cv_y = valid_row[4]
    else:
        raise ValueError(
            f"{ERROR_PREFIX} wrong number of FIELDS in file {filename}: only 1 or 2 dimensional bias are supported"
        )

    action = data[start + 1, 3]
    if action == "OPES_METAD_state":
        explore = False
    elif action == "OPES_METAD_EXPLORE_state":
        explore = True
    else:
        raise ValueError(
            f"{ERROR_PREFIX} This script works only with OPES_METAD_state and OPES_METAD_EXPLORE_state"
        )

    if data[start + 2, 2] != "biasfactor":
        raise ValueError(f"{ERROR_PREFIX} biasfactor not found")
    sf = 1.0
    if explore:
        sf = float(data[start + 2, 3])

    if data[start + 3, 2] != "epsilon":
        raise ValueError(f"{ERROR_PREFIX} epsilon not found")
    epsilon = float(data[start + 3, 3])

    if data[start + 4, 2] != "kernel_cutoff":
        raise ValueError(f"{ERROR_PREFIX} kernel_cutoff not found")
    cutoff = float(data[start + 4, 3])
    val_at_cutoff = np.exp(-0.5 * cutoff**2)

    if data[start + 6, 2] != "zed":
        raise ValueError(f"{ERROR_PREFIX} zed not found")
    zed = float(data[start + 6, 3])
    if not explore:
        if data[start + 7, 2] != "sum_weights":
            raise ValueError(f"{ERROR_PREFIX} sum_weights not found")
        zed *= float(data[start + 7, 3])
    else:
        if data[start + 9, 2] != "counter":
            raise ValueError(f"{ERROR_PREFIX} counter not found")
        zed *= float(data[start + 9, 3])

    header_lines = valid_row
    meta = _StateMeta(dim2, name_cv_x, name_cv_y, sf, epsilon, cutoff, val_at_cutoff, zed)
    return header_lines, meta, start + 10


def _parse_periodicity(
    data: np.ndarray,
    idx: int,
    end: int,
    header_fields: list[str],
    meta: _StateMeta,
) -> tuple[_StateMeta, int]:
    while idx < end and data[idx, 0] == "#!":
        key = data[idx, 2]
        val = data[idx, 3]
        if key == f"min_{meta.name_cv_x}":
            meta.grid_min_x_read = -np.pi if val == "-pi" else float(val)
        elif key == f"max_{meta.name_cv_x}":
            meta.grid_max_x_read = np.pi if val == "pi" else float(val)
        elif meta.dim2 and key == f"min_{meta.name_cv_y}":
            meta.grid_min_y_read = -np.pi if val == "-pi" else float(val)
        elif meta.dim2 and key == f"max_{meta.name_cv_y}":
            meta.grid_max_y_read = np.pi if val == "pi" else float(val)
        idx += 1

    if meta.grid_min_x_read is not None and meta.grid_max_x_read is not None:
        meta.period_x = meta.grid_max_x_read - meta.grid_min_x_read
    if meta.dim2 and meta.grid_min_y_read is not None and meta.grid_max_y_read is not None:
        meta.period_y = meta.grid_max_y_read - meta.grid_min_y_read

    return meta, idx


def _extract_kernels(data: np.ndarray, start: int, end: int, dim2: bool, filename: str, config: FESStateConfig, meta: _StateMeta):
    if start == end:
        raise ValueError(f"{ERROR_PREFIX} missing data!")
    chunk = data[start:end]
    center_x = np.array(chunk[:, 1], dtype=float)
    
    # Symmetrization logic
    if config.symmetrize_cvs:
        if meta.name_cv_x in config.symmetrize_cvs:
            print(f"   symmetrizing {meta.name_cv_x} (taking absolute value)")
            center_x = np.abs(center_x)

    if dim2:
        center_y = np.array(chunk[:, 2], dtype=float)
        if config.symmetrize_cvs and meta.name_cv_y in config.symmetrize_cvs:
            print(f"   symmetrizing {meta.name_cv_y} (taking absolute value)")
            center_y = np.abs(center_y)
            
        sigma_x = np.array(chunk[:, 3], dtype=float)
        sigma_y = np.array(chunk[:, 4], dtype=float)
        height = np.array(chunk[:, 5], dtype=float)
    else:
        center_y = None
        sigma_x = np.array(chunk[:, 2], dtype=float)
        sigma_y = None
        height = np.array(chunk[:, 3], dtype=float)
    return center_x, center_y, sigma_x, sigma_y, height


def _parse_grid_options_from_state(
    config: FESStateConfig,
    period_x,
    period_y,
    center_x,
    center_y,
    dim2,
    grid_min_x_read,
    grid_max_x_read,
    grid_min_y_read,
    grid_max_y_read,
):
    grid_bin_x = config.grid_bin[0] + 1
    grid_bin_y = None
    if dim2:
        grid_bin_y = config.grid_bin[1] + 1

    grid_min_x = None
    grid_min_y = None
    if config.grid_min is None:
        if period_x == 0:
            grid_min_x = np.min(center_x)
    else:
        grid_min_x = config.grid_min[0]
        if dim2:
            grid_min_y = config.grid_min[1]

    grid_max_x = None
    grid_max_y = None
    if config.grid_max is None:
        if period_x == 0:
            grid_max_x = np.max(center_x)
    else:
        grid_max_x = config.grid_max[0]
        if dim2:
            grid_max_y = config.grid_max[1]

    if grid_min_x is None:
        grid_min_x = grid_min_x_read if grid_min_x_read is not None else np.min(center_x)
    if grid_max_x is None:
        grid_max_x = grid_max_x_read if grid_max_x_read is not None else np.max(center_x)
    if dim2:
        if grid_min_y is None:
            grid_min_y = grid_min_y_read if grid_min_y_read is not None else np.min(center_y)
        if grid_max_y is None:
            grid_max_y = grid_max_y_read if grid_max_y_read is not None else np.max(center_y)

    return grid_bin_x, grid_bin_y, grid_min_x, grid_max_x, grid_min_y, grid_max_y


def _build_grid_from_state(config: FESStateConfig, meta: _StateMeta, kernels):
    center_x, center_y, sigma_x, sigma_y, height = kernels
    gbx, gby, gminx, gmaxx, gminy, gmaxy = _parse_grid_options_from_state(
        config,
        meta.period_x,
        meta.period_y,
        center_x,
        center_y,
        meta.dim2,
        meta.grid_min_x_read,
        meta.grid_max_x_read,
        meta.grid_min_y_read,
        meta.grid_max_y_read,
    )
    grid_cv_x = np.linspace(gminx, gmaxx, gbx)
    if meta.period_x > 0 and np.isclose(meta.period_x, grid_cv_x[-1] - grid_cv_x[0]):
        grid_cv_x = grid_cv_x[:-1]
        gbx -= 1

    axis_x = GridAxis(
        name=meta.name_cv_x,
        values=grid_cv_x,
        minimum=gminx,
        maximum=gmaxx,
        bins=gbx,
        period=meta.period_x,
    )

    axis_y = None
    mesh = None
    if meta.dim2:
        grid_cv_y = np.linspace(gminy, gmaxy, gby)
        if meta.period_y > 0 and np.isclose(meta.period_y, grid_cv_y[-1] - grid_cv_y[0]):
            grid_cv_y = grid_cv_y[:-1]
            gby -= 1
        axis_y = GridAxis(
            name=meta.name_cv_y if meta.name_cv_y is not None else "CV2",
            values=grid_cv_y,
            minimum=gminy,
            maximum=gmaxy,
            bins=gby,
            period=meta.period_y,
        )
        mesh = np.meshgrid(axis_x.values, axis_y.values, indexing="ij")

    grid_data = GridData(axes=(axis_x,) if axis_y is None else (axis_x, axis_y), mesh=mesh)
    return grid_data


def _evaluate_state_fes(config: FESStateConfig, meta: _StateMeta, kernels, grid: GridData):
    center_x, center_y, sigma_x, sigma_y, height = kernels
    grid_state = create_grid_runtime_state(grid, config.calc_der)
    print(f"   calculating FES... (Numba accelerated)")
    if not meta.dim2:
        prob = _numba_calc_prob_1d(
            grid_state.axis_x.values,
            center_x,
            sigma_x,
            height,
            meta.period_x,
            meta.val_at_cutoff,
        )
        if config.calc_der:
            der_prob_x = _numba_calc_der_prob_1d(
                grid_state.axis_x.values,
                center_x,
                sigma_x,
                height,
                meta.period_x,
                meta.val_at_cutoff,
            )
    else:
        x_mesh, y_mesh = np.meshgrid(grid_state.axis_x.values, grid_state.axis_y.values, indexing="ij")
        prob = _numba_calc_prob_2d(
            x_mesh,
            y_mesh,
            center_x,
            center_y,
            sigma_x,
            sigma_y,
            height,
            meta.period_x,
            meta.period_y,
            meta.val_at_cutoff,
        )
        if config.calc_der:
            der_prob_x, der_prob_y = _numba_calc_der_prob_2d(
                x_mesh,
                y_mesh,
                center_x,
                center_y,
                sigma_x,
                sigma_y,
                meta.period_x,
                meta.period_y,
                meta.val_at_cutoff,
            )

    prob = prob / meta.zed + meta.epsilon
    if config.calc_der:
        der_prob_x = der_prob_x / meta.zed
        if meta.dim2:
            der_prob_y = der_prob_y / meta.zed

    max_prob = 1.0
    if config.mintozero:
        max_prob = np.max(prob)

    grid_state.fes = -config.kbt * meta.sf * np.log(prob / max_prob)
    if config.calc_der:
        grid_state.der_fes_x = -config.kbt * meta.sf / prob * der_prob_x
        if meta.dim2:
            grid_state.der_fes_y = -config.kbt * meta.sf / prob * der_prob_y

    stats = SampleStats(size=len(center_x), effective_size=len(center_x))
    return grid_state, stats


def _write_state_outputs(
    config: FESStateConfig,
    meta: _StateMeta,
    grid_state,
    stats: SampleStats,
    mesh,
    output_conv: float,
):
    out_name = config.outfile
    out_dir = os.path.dirname(out_name)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    options = OutputOptions(
        fmt=config.fmt,
        mintozero=config.mintozero,
        calc_deltaF=False,
        delta_threshold=None,
        kbt=config.kbt,
        backup=config.backup,
        energy_unit=config.output_energy_unit,
        energy_conversion=output_conv,
    )

    names = (meta.name_cv_x, meta.name_cv_y if meta.dim2 else None)
    mesh_tuple = mesh if meta.dim2 else None

    write_standard_output(out_name, grid_state, names, options, stats, mesh_tuple)
    print(f"   printed to {out_name}")

    if config.plot:
        plot_manager = PlotManager(
            enabled=True,
            dim2=meta.dim2,
            axis_x=grid_state.axis_x,
            axis_y=grid_state.axis_y,
            mintozero=options.mintozero,
            mesh=mesh_tuple,
        )
        plot_manager.record_standard_surface(grid_state.fes * output_conv)
        png_name = out_name[:-4] + ".png" if out_name.lower().endswith(".dat") else f"{out_name}.png"
        plot_manager.save_standard_plot(png_name, names)
        print(f"   plotted to {png_name}")
