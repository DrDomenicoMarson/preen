#! /usr/bin/env python3

###  # pyright: reportUnboundVariable=false

import argparse
import os
import sys
from collections.abc import Sequence

import numpy as np

from FESutils.colvar_io import load_colvar_data
from FESutils.fes_config import FESConfig
from FESutils.grid import build_grid
from FESutils.kernel_eval import KernelEvaluator, KernelParams
from FESutils.fes_output import (
    OutputOptions,
    SampleStats,
    write_block_output,
    write_standard_output,
)
from FESutils.fes_plot import PlotManager
from FESutils.fes_state import (
    create_grid_runtime_state,
    create_sample_state,
    initialize_block_state,
)
from FESutils.constants import ERROR_PREFIX

ERROR_PREFIX = "--- ERROR:"


def parse_cli_args(argv: Sequence[str] | None = None) -> FESConfig:
    parser = argparse.ArgumentParser(
        description="calculate the free energy surfase (FES) along the chosen collective variables (1 or 2) using a reweighted kernel density estimate"
    )
    parser.add_argument(
        "--colvar",
        "-f",
        dest="filename",
        type=str,
        default="COLVAR",
        help="the COLVAR file name, with the collective variables and the bias",
    )
    parser.add_argument(
        "--outfile",
        "-o",
        dest="outfile",
        type=str,
        default="fes-rew.dat",
        help="name of the output file",
    )
    parser.add_argument(
        "--sigma",
        "-s",
        dest="sigma",
        type=str,
        required=True,
        help="the bandwidth for the kernel density estimation. Use e.g. the last value of sigma from an OPES_METAD simulation",
    )
    kbt_group = parser.add_mutually_exclusive_group(required=True)
    kbt_group.add_argument(
        "--kt", dest="kbt", type=float, help="the temperature in energy units"
    )
    kbt_group.add_argument(
        "--temp",
        dest="temp",
        type=float,
        help="the temperature in Kelvin. Energy units is Kj/mol",
    )
    parser.add_argument(
        "--cv",
        dest="cv",
        type=str,
        default="2",
        help="the CVs to be used. Either by name or by column number, starting from 1",
    )
    parser.add_argument(
        "--bias",
        dest="bias",
        type=str,
        default=".bias",
        help="the bias to be used. Either by name or by column number, starting from 1. Set to NO for nonweighted KDE",
    )
    parser.add_argument(
        "--min", dest="grid_min", type=str, help="lower bounds for the grid"
    )
    parser.add_argument(
        "--max", dest="grid_max", type=str, help="upper bounds for the grid"
    )
    parser.add_argument(
        "--bin",
        dest="grid_bin",
        type=str,
        default="100,100",
        help="number of bins for the grid",
    )
    split_group = parser.add_mutually_exclusive_group(required=False)
    split_group.add_argument(
        "--blocks",
        dest="blocks_num",
        type=int,
        default=1,
        help="calculate errors with block average, using this number of blocks",
    )
    split_group.add_argument(
        "--stride",
        dest="stride",
        type=int,
        default=0,
        help="print running FES estimate with this stride. Use --blocks for stride without history",
    )
    parser.add_argument(
        "--random-blocks",
        dest="random_blocks",
        action="store_true",
        default=False,
        help="shuffle data before splitting into blocks to sample without repetition",
    )
    parser.add_argument(
        "--block-seed",
        dest="block_seed",
        type=int,
        help="random seed for --random-blocks",
    )
    parser.add_argument(
        "--deltaFat",
        dest="deltaFat",
        type=float,
        help="calculate the free energy difference between left and right of given cv1 value",
    )
    parser.add_argument(
        "--skiprows",
        dest="skiprows",
        type=int,
        default=0,
        help="skip this number of initial rows",
    )
    parser.add_argument(
        "--reverse",
        dest="reverse",
        action="store_true",
        default=False,
        help="reverse the time. Should be combined with --stride, without --skiprows",
    )
    parser.add_argument(
        "--nomintozero",
        dest="nomintozero",
        action="store_true",
        default=False,
        help="do not shift the minimum to zero",
    )
    parser.add_argument(
        "--der",
        dest="der",
        action="store_true",
        default=False,
        help="calculate also FES derivatives",
    )
    parser.add_argument(
        "--fmt",
        dest="fmt",
        type=str,
        default="% 12.6f",
        help="specify the output format",
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        default=False,
        help="save PNG plots of the resulting FES",
    )
    parser.add_argument(
        "--no-backup",
        dest="backup",
        action="store_false",
        default=True,
        help="disable automatic backup of output files",
    )
    args = parser.parse_args(argv)
    cv_spec = _split_csv(str(args.cv))
    dim = len(cv_spec)
    if dim not in (1, 2):
        raise ValueError(f"{ERROR_PREFIX} only 1D and 2D are supported")
    sigma = _parse_sigma(args.sigma, dim)
    if args.kbt is not None:
        kbt = args.kbt
    else:
        kbt = args.temp * 0.0083144621
    grid_min = _parse_optional_float_vector(args.grid_min)
    grid_max = _parse_optional_float_vector(args.grid_max)
    grid_bin = _parse_int_vector(args.grid_bin)
    return FESConfig(
        filename=args.filename,
        outfile=args.outfile,
        sigma=sigma,
        kbt=kbt,
        cv_spec=cv_spec,
        bias_spec=args.bias,
        grid_min=grid_min,
        grid_max=grid_max,
        grid_bin=grid_bin,
        blocks_num=args.blocks_num,
        stride=args.stride,
        random_blocks=args.random_blocks,
        block_seed=args.block_seed,
        delta_f_threshold=args.deltaFat,
        skiprows=args.skiprows,
        mintozero=(not args.nomintozero),
        reverse=args.reverse,
        calc_der=args.der,
        fmt=args.fmt,
        plot=args.plot,
        backup=args.backup,
    )


def _split_csv(value: str) -> tuple[str, ...]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"{ERROR_PREFIX} at least one CV must be specified via --cv")
    return tuple(tokens)


def _parse_sigma(value: str, dim: int) -> tuple[float, ...]:
    tokens = [float(token.strip()) for token in value.split(",") if token.strip()]
    if dim == 1 and len(tokens) != 1:
        raise ValueError(
            f"{ERROR_PREFIX} one float expected after --sigma for 1D cases"
        )
    if dim == 2 and len(tokens) != 2:
        raise ValueError(
            f"{ERROR_PREFIX} two comma-separated floats expected after --sigma"
        )
    return tuple(tokens)


def _parse_optional_float_vector(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    tokens = [
        _convert_bound_token(token.strip()) for token in raw.split(",") if token.strip()
    ]
    return tuple(tokens)


def _parse_int_vector(raw: str) -> tuple[int, ...]:
    tokens = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"{ERROR_PREFIX} invalid --bin specification")
    return tuple(tokens)


def _convert_bound_token(token: str) -> float:
    if token == "-pi":
        return -np.pi
    if token == "pi":
        return np.pi
    return float(token)


def run(config: FESConfig):
    print("")
    colvar_data = load_colvar_data(config)
    samples = create_sample_state(colvar_data)
    len_tot = samples.len_tot
    name_cv_x = samples.name_cv_x
    name_cv_y = samples.name_cv_y if samples.name_cv_y is not None else ""
    cv_x = samples.cv_x
    cv_y = samples.cv_y
    bias = samples.bias
    dim2 = samples.dim2
    sigma_x = config.sigma[0]
    sigma_y = config.sigma[1] if dim2 else None
    kbt = config.kbt
    fmt = config.fmt
    calc_der = config.calc_der
    mintozero = config.mintozero
    outfile = config.outfile
    if outfile.endswith(os.sep):
        base_dir = outfile.rstrip(os.sep)
        base_name = "fes-rew.dat"
        outfile = os.path.join(base_dir, base_name) if base_dir else base_name
    else:
        base_name = os.path.basename(outfile)
        if base_name == "":
            base_name = "fes-rew.dat"
            outfile = (
                os.path.join(os.path.dirname(outfile), base_name)
                if os.path.dirname(outfile)
                else base_name
            )
    output_dir = os.path.dirname(outfile)
    base_name = os.path.basename(outfile)
    root_name, ext = os.path.splitext(base_name)
    if config.random_blocks:
        if config.blocks_num == 1:
            raise ValueError(f"{ERROR_PREFIX} --random-blocks requires --blocks > 1")
        rng = np.random.default_rng(config.block_seed)
        perm = rng.permutation(samples.len_tot)
        if config.block_seed is None:
            print(" shuffling data randomly across blocks (seed: system entropy)")
        else:
            print(f" shuffling data randomly across blocks (seed: {config.block_seed})")
        samples.apply_permutation(perm)
    size = 0
    effsize = 0
    grid = build_grid(config, colvar_data)
    grid_state = create_grid_runtime_state(grid, calc_der)
    axis_x = grid_state.axis_x
    axis_y = grid_state.axis_y
    grid_cv_x = axis_x.values
    grid_min_x = axis_x.minimum
    grid_max_x = axis_x.maximum
    #grid_bin_x = axis_x.bins
    period_x = axis_x.period
    if dim2 and axis_y is None:
        raise RuntimeError(
            f"{ERROR_PREFIX} internal inconsistency: missing second grid axis"
        )
    if not dim2:
        grid_cv_y = None
        grid_min_y = None
        grid_max_y = None
        grid_bin_y = 0
        period_y = 0
        x = None
        y = None
    else:
        grid_cv_y = axis_y.values
        grid_min_y = axis_y.minimum
        grid_max_y = axis_y.maximum
        grid_bin_y = axis_y.bins
        period_y = axis_y.period
        if grid_state.mesh is not None:
            x, y = grid_state.mesh
        else:
            x, y = np.meshgrid(grid_cv_x, grid_cv_y, indexing="ij")
    fes = grid_state.fes
    der_fes_x = grid_state.der_fes_x
    der_fes_y = grid_state.der_fes_y
    calc_deltaF = config.calculate_delta_f
    ts = config.delta_f_threshold
    if calc_deltaF and (ts <= grid_min_x or ts >= grid_max_x):
        print(" +++ WARNING: the provided --deltaFat is out of the CV grid +++")
        calc_deltaF = False
    if config.blocks_num != 1 and calc_der:
        raise ValueError(
            f"{ERROR_PREFIX} derivatives not supported with --blocks, remove --der option"
        )
    block_state = initialize_block_state(config, samples.len_tot, fes.shape)
    block_av = block_state.enabled
    stride = block_state.stride
    blocks_num = block_state.blocks_num
    block_logweight = block_state.logweight
    block_fes = block_state.fes_storage
    stride_dir = None
    if stride != samples.len_tot:
        chunks = int(samples.len_tot / stride)
        print(f" printing {chunks} FES files, one every {stride} samples")
        stride_dir = os.path.join(output_dir, "stride") if output_dir else "stride"
        os.makedirs(stride_dir, exist_ok=True)
    output_options = OutputOptions(
        fmt=fmt,
        mintozero=mintozero,
        calc_deltaF=calc_deltaF,
        delta_threshold=ts,
        kbt=kbt,
        backup=config.backup,
    )
    names = (name_cv_x, name_cv_y if dim2 else None)
    mesh_tuple = (x, y) if dim2 else None
    kernel_params = KernelParams(
        sigma_x=sigma_x, sigma_y=sigma_y, period_x=period_x, period_y=period_y, kbt=kbt
    )
    kernel_evaluator = KernelEvaluator(dim2, calc_der)
    plot_manager = PlotManager(
        config.plot, dim2, grid_state.axis_x, grid_state.axis_y, mintozero, mesh_tuple
    )

    def _chunk_path(idx: int) -> str:
        if stride_dir is None:
            raise RuntimeError(
                "stride directory requested without stride configuration"
            )
        filename = f"{root_name}_{idx}{ext}"
        return os.path.join(stride_dir, filename)

    s = len_tot % stride
    if s > 1:
        print(f" first {s} samples discarded to fit with given stride")
    it = 1
    for n in range(s + stride, len_tot + 1, stride):
        chunk_start = s
        chunk_end = n
        if stride != len_tot:
            print(f"     working...     0% of {n/(len_tot+1):.0%}", end="\r")
        kernel_evaluator.fill_chunk(samples, s, n, grid_state, kernel_params)
        weights = np.exp(bias[s:n] - np.amax(bias[s:n]))
        size = len(weights)
        effsize = np.sum(weights) ** 2 / np.sum(weights**2)
        if block_av or not mintozero:
            bias_norm_shift = np.logaddexp.reduce(bias[s:n])
            fes += kbt * bias_norm_shift
        if config.plot:
            label = f"{chunk_start}-{chunk_end-1}"
            if stride == len_tot:
                plot_manager.record_standard_surface(fes)
            else:
                plot_manager.add_stride_surface(fes, label)
        if block_av:
            block_logweight[it - 1] = bias_norm_shift
            block_fes[it - 1] = fes
            s = n
        target_file = outfile if stride == len_tot else _chunk_path(it)
        stats = SampleStats(size=size, effective_size=effsize)
        write_standard_output(
            target_file, grid_state, names, output_options, stats, mesh_tuple
        )
        if stride != len_tot:
            it += 1
    if config.plot:
        if stride == len_tot:
            standard_plot_path = (
                os.path.join(output_dir, f"{root_name}.png")
                if output_dir
                else f"{root_name}.png"
            )
            plot_manager.save_standard_plot(standard_plot_path, names)
        elif stride_dir is not None:
            stride_plot_path = os.path.join(stride_dir, f"{root_name}_stride.png")
            plot_manager.save_stride_plots(stride_plot_path, names)
    if block_av:
        block_dir = os.path.join(output_dir, "block") if output_dir else "block"
        os.makedirs(block_dir, exist_ok=True)
        block_outfile = os.path.join(block_dir, base_name)
        block_err_plot = os.path.join(block_dir, f"{root_name}_block_error.png")
        print("  NOTE: try different numbers of blocks and"
              " check for the convergence of the uncertainty estimate")
        print(f" printing final FES with block average to {block_outfile}")
        start = len_tot % stride
        size = len_tot - start
        weights = np.exp(bias[start:] - np.amax(bias[start:]))
        effsize = np.sum(weights) ** 2 / np.sum(weights**2)
        safe_block_weight = np.exp(block_logweight - np.amax(block_logweight))
        blocks_neff = np.sum(safe_block_weight) ** 2 / np.sum(safe_block_weight**2)
        print(
            f" number of blocks is {blocks_num}, while effective number is {blocks_neff:.2f}"
        )
        original_fes = grid_state.fes.copy()
        fes_block = -kbt * np.log(
            np.average(np.exp(-block_fes / kbt), axis=0, weights=safe_block_weight)
        )
        np.copyto(grid_state.fes, fes_block)
        fes_err = kbt * np.sqrt(
            1
            / (blocks_neff - 1)
            * (
                np.average(
                    np.expm1(-(block_fes - fes_block) / kbt) ** 2,
                    axis=0,
                    weights=safe_block_weight,
                )
            )
        )
        print(f" average FES uncertainty is: {np.average(fes_err):.4f}")
        stats = SampleStats(
            size=size,
            effective_size=effsize,
            blocks_num=blocks_num,
            blocks_effective=blocks_neff,
        )
        write_block_output(
            block_outfile, grid_state, names, output_options, stats, fes_err, mesh_tuple
        )
        if config.plot:
            plot_manager.save_block_plots(block_err_plot, fes_block, fes_err, names)
        np.copyto(grid_state.fes, original_fes)
    print("                                                            ")


def main(argv: Sequence[str] | None = None):
    try:
        config = parse_cli_args(argv)
        run(config)
    except (ValueError, RuntimeError) as e:
        print(f"{ERROR_PREFIX} {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
