import os
import sys
import numpy as np
import pandas as pd

from FESutils.constants import KB_KJ_MOL, ERROR_PREFIX, energy_conversion_factor
from FESutils.colvar_io import load_colvar_data, open_text_file
from FESutils.fes_config import FESConfig
from FESutils.grid import build_grid, GridAxis, GridData
from FESutils.fes_state import create_grid_runtime_state, create_sample_state, initialize_block_state
from FESutils.kernel_eval import (
    KernelEvaluator, KernelParams, 
    _numba_calc_prob_1d, _numba_calc_der_prob_1d, 
    _numba_calc_prob_2d, _numba_calc_der_prob_2d
)
from FESutils.fes_output import OutputOptions, SampleStats, write_standard_output, write_block_output
from FESutils.fes_plot import PlotManager
from numba import set_num_threads

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
        if config.plot:
            if chunk_kind is None:
                plot_manager.record_standard_surface(fes * output_conv)
            else:
                if chunk_kind == "block":
                    label = f"block{it} ({chunk_start}-{chunk_end-1})"
                else:
                    label = f"{chunk_start}-{chunk_end-1} (cumulative)"
                plot_manager.add_stride_surface(fes * output_conv, label)
        if block_av:
            block_logweight[it - 1] = bias_norm_shift
            block_fes[it - 1] = fes
            s = n
        target_file = outfile if chunk_kind is None else _chunk_path(it)
        stats = SampleStats(size=size, effective_size=effsize)
        last_stats = stats
        write_standard_output(
            target_file, grid_state, names, output_options, stats, mesh_tuple
        )
        if chunk_kind is not None:
            it += 1
            
    if config.plot:
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
            outfile, grid_state, names, output_options, last_stats, mesh_tuple
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
        write_block_output(
            block_outfile, grid_state, names, output_options, stats, fes_err, mesh_tuple
        )
        if config.plot:
            plot_manager.save_block_plots(
                block_err_plot,
                fes_block * output_conv,
                fes_err * output_conv,
                names,
            )
        np.copyto(grid_state.fes, original_fes)
    print("                                                            ")


def _parse_grid_options_from_state(config: FESConfig, period_x, period_y, center_x, center_y, dim2, grid_min_x_read, grid_max_x_read, grid_min_y_read, grid_max_y_read):
    # Grid bin
    # config.grid_bin is tuple[int, ...]
    grid_bin_x = config.grid_bin[0] + 1
    grid_bin_y = None
    if dim2:
        if len(config.grid_bin) != 2:
             # Should be handled by config parsing but double check
             pass
        grid_bin_y = config.grid_bin[1] + 1

    # Grid min
    grid_min_x = None
    grid_min_y = None
    if config.grid_min is None:
        if period_x == 0:
            grid_min_x = np.min(center_x)
    else:
        # config.grid_min is tuple[float, ...]
        grid_min_x = config.grid_min[0]
        if dim2:
            grid_min_y = config.grid_min[1]
            
    # Grid max
    grid_max_x = None
    grid_max_y = None
    if config.grid_max is None:
        if period_x == 0:
            grid_max_x = np.max(center_x)
    else:
        grid_max_x = config.grid_max[0]
        if dim2:
            grid_max_y = config.grid_max[1]

    # Use read values if not set
    if grid_min_x is None: grid_min_x = grid_min_x_read if grid_min_x_read is not None else np.min(center_x)
    if grid_max_x is None: grid_max_x = grid_max_x_read if grid_max_x_read is not None else np.max(center_x)
    if dim2:
        if grid_min_y is None: grid_min_y = grid_min_y_read if grid_min_y_read is not None else np.min(center_y)
        if grid_max_y is None: grid_max_y = grid_max_y_read if grid_max_y_read is not None else np.max(center_y)

    return grid_bin_x, grid_bin_y, grid_min_x, grid_max_x, grid_min_y, grid_max_y


def calculate_fes_from_state(config: FESConfig):
    """
    Calculate FES from STATE file (sum of kernels).
    """
    set_num_threads(config.num_threads)
    # 1. Load Data
    try:
        with open_text_file(config.filename) as f:
            data_df = pd.read_table(f, sep=r"\s+", header=None, dtype=str, comment=None)
        data = data_df.to_numpy()
    except Exception as e:
        raise RuntimeError(f"{ERROR_PREFIX} failed to read {config.filename}: {e}")

    # 2. Find FIELDS
    fields_pos = []
    tot_lines = data.shape[0]
    for i in range(tot_lines):
        if data[i, 1] == "FIELDS":
            fields_pos.append(i)
    
    if len(fields_pos) == 0:
        raise ValueError(f"{ERROR_PREFIX} no FIELDS found in file {config.filename}")
    output_conv = energy_conversion_factor("kJ/mol", config.output_energy_unit)
    
    # Note: config doesn't have 'all_stored' flag in FESConfig definition yet?
    # We might need to add it or assume False.
    # Assuming False for now as it's not in standard FESConfig, 
    # or we can check if it was passed via kwargs if we extend FESConfig.
    # Let's assume we only process the last one unless we extend FESConfig.
    # But wait, the original script had --all_stored.
    # We should probably add it to FESConfig or handle it.
    # For now, let's default to last one.
    
    if len(fields_pos) > 1:
        print(f" a total of {len(fields_pos)} stored states were found")
        # TODO: Support all_stored in API/Config
        print("  -> only the last one will be printed.")
        fields_pos = [fields_pos[-1]]
    
    fields_pos.append(tot_lines)

    # 3. Process each state
    for n in range(len(fields_pos) - 1):
        total_states = len(fields_pos) - 1
        print(f"   working... state {n+1}/{total_states}", end="\r")
        l = fields_pos[n]
        
        # Parse Header
        dim2 = False
        valid_row = [x for x in data[l, :] if isinstance(x, str)]
        
        fields = valid_row
        if len(fields) == 6:
            name_cv_x = fields[3]
        elif len(fields) == 8:
            dim2 = True
            name_cv_x = fields[3]
            name_cv_y = fields[4]
        else:
             raise ValueError(f"{ERROR_PREFIX} wrong number of FIELDS in file {config.filename}: only 1 or 2 dimensional bias are supported")

        # Parse Metadata
        action = data[l+1, 3]
        explore = 'unset'
        if action == "OPES_METAD_state":
            explore = 'no'
        elif action == "OPES_METAD_EXPLORE_state":
            explore = 'yes'
        else:
            raise ValueError(f"{ERROR_PREFIX} This script works only with OPES_METAD_state and OPES_METAD_EXPLORE_state")

        if data[l+2, 2] != 'biasfactor': raise ValueError(f"{ERROR_PREFIX} biasfactor not found")
        
        sf = 1.0
        if explore == 'yes':
            sf = float(data[l+2, 3])
        
        if data[l+3, 2] != 'epsilon': raise ValueError(f"{ERROR_PREFIX} epsilon not found")
        epsilon = float(data[l+3, 3])
        
        if data[l+4, 2] != 'kernel_cutoff': raise ValueError(f"{ERROR_PREFIX} kernel_cutoff not found")
        cutoff = float(data[l+4, 3])
        val_at_cutoff = np.exp(-0.5 * cutoff**2)
        
        if data[l+6, 2] != 'zed': raise ValueError(f"{ERROR_PREFIX} zed not found")
        Zed = float(data[l+6, 3])
        
        if explore == 'no':
            if data[l+7, 2] != 'sum_weights': raise ValueError(f"{ERROR_PREFIX} sum_weights not found")
            Zed *= float(data[l+7, 3])
        if explore == 'yes':
            if data[l+9, 2] != 'counter': raise ValueError(f"{ERROR_PREFIX} counter not found")
            Zed *= float(data[l+9, 3])
            
        l += 10 # Skip header

        # Parse Periodicity
        period_x = 0.0
        period_y = 0.0
        grid_min_x_read = None
        grid_max_x_read = None
        grid_min_y_read = None
        grid_max_y_read = None

        while l < tot_lines and data[l, 0] == '#!':
            key = data[l, 2]
            val = data[l, 3]
            
            if key == 'min_' + name_cv_x:
                grid_min_x_read = -np.pi if val == '-pi' else float(val)
            elif key == 'max_' + name_cv_x:
                grid_max_x_read = np.pi if val == 'pi' else float(val)
            elif dim2 and key == 'min_' + name_cv_y:
                grid_min_y_read = -np.pi if val == '-pi' else float(val)
            elif dim2 and key == 'max_' + name_cv_y:
                grid_max_y_read = np.pi if val == 'pi' else float(val)
            l += 1
        
        if grid_min_x_read is not None and grid_max_x_read is not None:
            period_x = grid_max_x_read - grid_min_x_read
        if dim2 and grid_min_y_read is not None and grid_max_y_read is not None:
            period_y = grid_max_y_read - grid_min_y_read

        if config.calc_der and (period_x > 0 or period_y > 0):
             raise ValueError(f"{ERROR_PREFIX} derivatives not supported with periodic CVs, remove --der option")

        if l == fields_pos[n+1]:
            raise ValueError(f"{ERROR_PREFIX} missing data!")

        # Extract Kernels
        chunk = data[l:fields_pos[n+1]]
        center_x = np.array(chunk[:, 1], dtype=float)
        
        if dim2:
            center_y = np.array(chunk[:, 2], dtype=float)
            sigma_x = np.array(chunk[:, 3], dtype=float)
            sigma_y = np.array(chunk[:, 4], dtype=float)
            height = np.array(chunk[:, 5], dtype=float)
        else:
            center_y = None
            sigma_x = np.array(chunk[:, 2], dtype=float)
            sigma_y = None
            height = np.array(chunk[:, 3], dtype=float)

        # Prepare Grid
        gbx, gby, gminx, gmaxx, gminy, gmaxy = _parse_grid_options_from_state(
            config, period_x, period_y, center_x, center_y, dim2,
            grid_min_x_read, grid_max_x_read, grid_min_y_read, grid_max_y_read
        )
        
        # Build Grid Objects
        grid_cv_x = np.linspace(gminx, gmaxx, gbx)
        if period_x > 0 and np.isclose(period_x, grid_cv_x[-1] - grid_cv_x[0]):
             grid_cv_x = grid_cv_x[:-1]
             gbx -= 1
        
        axis_x = GridAxis(
            name=name_cv_x,
            values=grid_cv_x,
            minimum=gminx,
            maximum=gmaxx,
            bins=gbx,
            period=period_x
        )
        
        axis_y = None
        if dim2:
            grid_cv_y = np.linspace(gminy, gmaxy, gby)
            if period_y > 0 and np.isclose(period_y, grid_cv_y[-1] - grid_cv_y[0]):
                grid_cv_y = grid_cv_y[:-1]
                gby -= 1
            
            axis_y = GridAxis(
                name=name_cv_y,
                values=grid_cv_y,
                minimum=gminy,
                maximum=gmaxy,
                bins=gby,
                period=period_y
            )
        
        axes_list = [axis_x]
        mesh = None
        if axis_y is not None:
            axes_list.append(axis_y)
            mesh = np.meshgrid(axis_x.values, axis_y.values, indexing="ij")
        
        grid_data = GridData(axes=tuple(axes_list), mesh=mesh)
        grid_state = create_grid_runtime_state(grid_data, config.calc_der)
        
        print(f"   calculating FES... (Numba accelerated)")
        
        if not dim2:
            prob = _numba_calc_prob_1d(
                grid_state.axis_x.values,
                center_x,
                sigma_x,
                height,
                period_x,
                val_at_cutoff
            )
            if config.calc_der:
                der_prob_x = _numba_calc_der_prob_1d(
                    grid_state.axis_x.values,
                    center_x,
                    sigma_x,
                    height,
                    period_x,
                    val_at_cutoff
                )
        else:
            x_mesh, y_mesh = np.meshgrid(axis_x.values, axis_y.values, indexing='ij')
            prob = _numba_calc_prob_2d(
                x_mesh,
                y_mesh,
                center_x,
                center_y,
                sigma_x,
                sigma_y,
                height,
                period_x,
                period_y,
                val_at_cutoff
            )
            if config.calc_der:
                der_prob_x, der_prob_y = _numba_calc_der_prob_2d(
                    x_mesh,
                    y_mesh,
                    center_x,
                    center_y,
                    sigma_x,
                    sigma_y,
                    height,
                    period_x,
                    period_y,
                    val_at_cutoff
                )

        # Normalize
        prob = prob / Zed + epsilon
        
        if config.calc_der:
            der_prob_x = der_prob_x / Zed
            if dim2:
                der_prob_y = der_prob_y / Zed

        # Mintozero
        max_prob = 1.0
        if config.mintozero:
            max_prob = np.max(prob)
        
        # Calc FES
        grid_state.fes = -config.kbt * sf * np.log(prob / max_prob)
        
        if config.calc_der:
            grid_state.der_fes_x = -config.kbt * sf / prob * der_prob_x
            if dim2:
                grid_state.der_fes_y = -config.kbt * sf / prob * der_prob_y

        # Output
        out_name = config.outfile
        out_dir = os.path.dirname(out_name)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        # Handle numbering if we supported multiple
        
        options = OutputOptions(
            fmt=config.fmt,
            mintozero=config.mintozero,
            calc_deltaF=(config.delta_f_threshold is not None),
            delta_threshold=config.delta_f_threshold,
            kbt=config.kbt,
            backup=config.backup,
            energy_unit=config.output_energy_unit,
            energy_conversion=output_conv,
        )
        
        stats = SampleStats(
            size=len(center_x),
            effective_size=len(center_x)
        )
        
        names = (name_cv_x, name_cv_y if dim2 else None)
        mesh_tuple = (x_mesh, y_mesh) if dim2 else None
        
        write_standard_output(out_name, grid_state, names, options, stats, mesh_tuple)
        print(f"   printed to {out_name}")
        
        # Plotting
        if config.plot:
            plot_manager = PlotManager(
                enabled=True,
                dim2=dim2,
                axis_x=axis_x,
                axis_y=axis_y,
                mintozero=options.mintozero,
                mesh=mesh_tuple
            )
            plot_manager.record_standard_surface(grid_state.fes * output_conv)
            
            if out_name.lower().endswith(".dat"):
                png_name = out_name[:-4] + ".png"
            else:
                png_name = out_name + ".png"
                
            plot_manager.save_standard_plot(png_name, names)
            print(f"   plotted to {png_name}")

    print("Done.")
