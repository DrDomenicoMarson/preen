import os
import numpy as np
from numba import set_num_threads

from .constants import KB_KJ_MOL, ERROR_PREFIX, energy_conversion_factor
from .colvar_io import load_colvar_data
from .fes_config import FESConfig
from .grid import build_grid
from .fes_state import create_grid_runtime_state, create_sample_state, initialize_block_state, symmetrize_grid_state, symmetrize_array
from .kernel_eval import KernelEvaluator, KernelParams
from .fes_output import OutputOptions, SampleStats, write_standard_output, write_block_output
from .fes_plot import PlotManager


def calculate_fes(config: FESConfig, merge_result=None):
    """
    Calculate FES from COLVAR data (reweighting).
    """
    set_num_threads(config.num_threads)
    if config.sigma is None:
        raise ValueError(f"{ERROR_PREFIX} sigma is required for reweighting")
    if config.cv_spec is None:
        raise ValueError(f"{ERROR_PREFIX} cv_spec is required for reweighting")
    if config.bias_spec is None:
        raise ValueError(f"{ERROR_PREFIX} bias_spec is required for reweighting")

    if config.dimension > 2:
        raise ValueError(f"{ERROR_PREFIX} only 1 or 2 dimensional bias are supported")

    print("")

    def _display_path(path: str) -> str:
        """Return path relative to current working directory when possible."""
        try:
            return os.path.relpath(path)
        except ValueError:
            return path

    colvar_data = load_colvar_data(config, merge_result=merge_result)
    samples = create_sample_state(colvar_data)
    len_tot = samples.len_tot
    name_cv_x = samples.name_cv_x
    name_cv_y = samples.name_cv_y if samples.name_cv_y is not None else ""
    bias = samples.bias
    dim2 = samples.dim2
    sigma_x = config.sigma[0]
    sigma_y = config.sigma[1] if dim2 else None

    cv_x = samples.cv_x
    cv_y = samples.cv_y

    # Symmetrization logic
    if config.symmetrize_cvs:
        if name_cv_x in config.symmetrize_cvs:
            print(f"   symmetrizing {name_cv_x} (taking absolute value)")
            cv_x = np.abs(cv_x)
            samples.cv_x = cv_x
        
        if dim2 and name_cv_y in config.symmetrize_cvs:
            print(f"   symmetrizing {name_cv_y} (taking absolute value)")
            cv_y = np.abs(cv_y)
            samples.cv_y = cv_y
            
        for name in config.symmetrize_cvs:
            if name != name_cv_x and (not dim2 or name != name_cv_y):
                print(f" +++ WARNING: CV '{name}' specified in symmetrize_cvs but not found in data +++")
    kbt = config.kbt
    fmt = config.fmt
    calc_der = config.calc_der
    mintozero = config.mintozero
    outfile = config.outfile

    # Handle outfile path logic
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
    # Ensure parent directory exists if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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

    grid = build_grid(config, colvar_data)
    grid_state = create_grid_runtime_state(grid, calc_der)
    axis_x = grid_state.axis_x
    axis_y = grid_state.axis_y
    grid_cv_x = axis_x.values
    grid_min_x = axis_x.minimum
    grid_max_x = axis_x.maximum
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
    output_conv = energy_conversion_factor("kJ/mol", config.output_energy_unit)

    if calc_deltaF and (ts <= grid_min_x or ts >= grid_max_x):
        print(" +++ WARNING: the provided --deltaFat is out of the CV grid +++")
        calc_deltaF = False

    if config.blocks_num != 1 and calc_der:
        raise ValueError(
            f"{ERROR_PREFIX} derivatives not supported with --blocks, remove --der option"
        )

    block_state = initialize_block_state(config, samples.len_tot, fes.shape)
    block_av = block_state.enabled
    if block_av and config.stride > 0:
        raise ValueError(
            f"{ERROR_PREFIX} --blocks and --stride are mutually exclusive; choose one"
        )
    stride_mode = (not block_av) and (config.stride > 0)
    stride = block_state.stride if block_av else (
        config.stride if stride_mode and config.stride <= samples.len_tot else samples.len_tot
    )
    blocks_num = block_state.blocks_num
    block_logweight = block_state.logweight
    block_fes = block_state.fes_storage
    chunk_dir = None
    chunk_kind = None  # "block" or "stride"
    if block_av:
        chunk_kind = "block"
        dir_name = f"{root_name}_blocks-{blocks_num}"
    elif stride_mode and stride < samples.len_tot:
        chunk_kind = "stride"
        dir_name = f"{root_name}_strided-{stride}"
    else:
        dir_name = None
    if dir_name is not None:
        chunk_dir = os.path.join(output_dir, dir_name) if output_dir else dir_name
        os.makedirs(chunk_dir, exist_ok=True)
        chunks = max(1, int(samples.len_tot / stride))
        prefix = "blocks" if chunk_kind == "block" else "cumulative stride"
        print(f" printing {chunks} FES files ({prefix}) to {_display_path(chunk_dir)}")

    output_options = OutputOptions(
        fmt=fmt,
        mintozero=mintozero,
        calc_deltaF=calc_deltaF,
        delta_threshold=ts,
        kbt=kbt,
        backup=config.backup,
        energy_unit=config.output_energy_unit,
        energy_conversion=output_conv,
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
        if chunk_dir is None:
            raise RuntimeError(
                "chunk directory requested without chunk configuration"
            )
        if chunk_kind == "block":
            filename = f"{root_name}_block{idx}{ext}"
        else:
            filename = f"{root_name}_stride{idx}{ext}"
        return os.path.join(chunk_dir, filename)

    s = len_tot % stride
    if s > 1:
        print(f" first {s} samples discarded to fit with given stride")
    it = 1
    last_stats: SampleStats | None = None
    output_grid_state = grid_state
    for n in range(s + stride, len_tot + 1, stride):
        chunk_start = s
        chunk_end = n
        if stride != len_tot:
            progress = n / len_tot if len_tot else 1.0
            print(f"     working... {n}/{len_tot} samples ({progress:.0%})", end="\r")
        kernel_evaluator.fill_chunk(samples, s, n, grid_state, kernel_params)
        weights = np.exp(bias[s:n] - np.amax(bias[s:n]))
        size = len(weights)
        effsize = np.sum(weights) ** 2 / np.sum(weights**2)
        if block_av or not mintozero:
            bias_norm_shift = np.logaddexp.reduce(bias[s:n])
            fes += kbt * bias_norm_shift
            
        output_grid_state = grid_state
        if config.symmetrize_cvs:
             output_grid_state = symmetrize_grid_state(grid_state, config.symmetrize_cvs)

        if config.plot:
            if chunk_kind is None:
                plot_fes = fes
                if config.symmetrize_cvs:
                    plot_fes = output_grid_state.fes
                plot_manager.record_standard_surface(plot_fes * output_conv)
            else:
                if chunk_kind == "block":
                    label = f"block{it} ({chunk_start}-{chunk_end-1})"
                else:
                    label = f"{chunk_start}-{chunk_end-1} (cumulative)"
                
                plot_fes = fes
                if config.symmetrize_cvs:
                    plot_fes = output_grid_state.fes
                plot_manager.add_stride_surface(plot_fes * output_conv, label)
        if block_av:
            block_logweight[it - 1] = bias_norm_shift
            block_fes[it - 1] = fes
            s = n
        target_file = outfile if chunk_kind is None else _chunk_path(it)
        stats = SampleStats(size=size, effective_size=effsize)
        last_stats = stats
        
        write_standard_output(
            target_file, output_grid_state, names, output_options, stats, output_grid_state.mesh
        )
        if chunk_kind is not None:
            it += 1

    if config.plot:
        # Plot manager needs to know about symmetrized axes if we used them
        if config.symmetrize_cvs:
            # We need to re-init plot manager with new axes?
            # Or just update them.
            # PlotManager stores axis_x, axis_y.
            # We can update them.
            # output_grid_state is from the last iteration.
            plot_manager.axis_x = output_grid_state.axis_x
            plot_manager.axis_y = output_grid_state.axis_y
            plot_manager.mesh = output_grid_state.mesh
            
        if chunk_kind is None:
            standard_plot_path = (
                os.path.join(output_dir, f"{root_name}.png")
                if output_dir
                else f"{root_name}.png"
            )
            plot_manager.save_standard_plot(standard_plot_path, names)
        elif chunk_kind == "stride":
            stride_plot_path = (
                os.path.join(output_dir, f"{root_name}_strided.png")
                if output_dir
                else f"{root_name}_strided.png"
            )
            plot_manager.save_stride_plots(stride_plot_path, names)
        elif chunk_kind == "block":
            block_plot_path = (
                os.path.join(output_dir, f"{root_name}_blocks.png")
                if output_dir
                else f"{root_name}_blocks.png"
            )
            plot_manager.save_stride_plots(block_plot_path, names)

    # Write final aggregated output in the parent directory for stride mode
    if chunk_kind == "stride" and last_stats is not None:
        write_standard_output(
            outfile, output_grid_state, names, output_options, last_stats, output_grid_state.mesh
        )

    if block_av:
        block_dir = chunk_dir if chunk_dir is not None else (
            os.path.join(output_dir, f"{root_name}_blocks-{blocks_num}")
            if output_dir
            else f"{root_name}_blocks-{blocks_num}"
        )
        os.makedirs(block_dir, exist_ok=True)
        block_outfile = (
            os.path.join(output_dir, f"{root_name}_block-avg{ext}")
            if output_dir
            else f"{root_name}_block-avg{ext}"
        )
        block_err_plot = (
            os.path.join(output_dir, f"{root_name}_block-error.png")
            if output_dir
            else f"{root_name}_block-error.png"
        )
        print(f" printing final FES with block average to {_display_path(block_outfile)}")
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
        
        output_grid_state = grid_state
        output_fes_err = fes_err
        if config.symmetrize_cvs:
             output_grid_state = symmetrize_grid_state(grid_state, config.symmetrize_cvs)
             
             # Symmetrize error
             sym_x = grid_state.axis_x.name in config.symmetrize_cvs
             sym_y = grid_state.axis_y is not None and grid_state.axis_y.name in config.symmetrize_cvs
             
             # Need to know if merged. Re-check logic or expose from helper.
             # Helper returns state.
             # Let's just assume we can re-derive it or use helper logic.
             # Actually, symmetrize_grid_state uses _sym_axis which checks merge.
             # We can't easily access that here without duplicating logic.
             # But wait, output_grid_state.fes has the new shape.
             # We can use that shape to guide us? No.
             
             # Let's just use symmetrize_array with manual check?
             # Or better, update symmetrize_grid_state to allow passing extra arrays?
             # Too late, I already wrote it.
             # I'll just duplicate the merge check logic briefly here or trust it matches.
             
             # Check x merge
             vx = grid_state.axis_x.values
             vx_new = np.concatenate((-vx[::-1], vx))
             merged_x = np.isclose(vx_new[len(vx)-1], vx_new[len(vx)])
             
             merged_y = False
             orig_y_bins = 0
             if grid_state.dim2:
                 vy = grid_state.axis_y.values
                 vy_new = np.concatenate((-vy[::-1], vy))
                 merged_y = np.isclose(vy_new[len(vy)-1], vy_new[len(vy)])
                 orig_y_bins = grid_state.axis_y.bins
                 
             output_fes_err = symmetrize_array(fes_err, sym_x, sym_y, merged_x, merged_y, orig_y_bins)

        write_block_output(
            block_outfile, output_grid_state, names, output_options, stats, output_fes_err, output_grid_state.mesh
        )
        if config.plot:
            # Update plot manager axes again just in case
            if config.symmetrize_cvs:
                plot_manager.axis_x = output_grid_state.axis_x
                plot_manager.axis_y = output_grid_state.axis_y
                plot_manager.mesh = output_grid_state.mesh
                
            plot_manager.save_block_plots(
                block_err_plot,
                output_grid_state.fes * output_conv,
                output_fes_err * output_conv,
                names,
            )
        np.copyto(grid_state.fes, original_fes)
    print("                                                            ")
