#! /usr/bin/env python3

"""
Calculate Free Energy Surface (FES) from a dumped OPES state file.
Uses Numba-accelerated kernel density estimation.
"""

import sys
import argparse
import numpy as np
import pandas as pd
from FESutils.constants import KB_KJ_MOL, ERROR_PREFIX
from FESutils.colvar_io import open_text_file
from FESutils.fes_config import FESConfig
from FESutils.grid import GridAxis, GridData
from FESutils.fes_state import GridRuntimeState, create_grid_runtime_state
from FESutils.fes_plot import PlotManager
from FESutils.kernel_eval import KernelEvaluator, KernelParams
from FESutils.fes_output import OutputOptions, SampleStats, write_standard_output
from FESutils import fes_state

def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Get the FES estimate used by OPES, from a dumped state file (STATE_WFILE). 1D or 2D only"
    )
    parser.add_argument(
        "--state",
        "-f",
        dest="filename",
        type=str,
        default="STATE",
        help="the state file name, with the compressed kernels",
    )
    parser.add_argument(
        "--outfile",
        "-o",
        dest="outfile",
        type=str,
        default="fes.dat",
        help="name of the output file",
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
        "--min", dest="grid_min", type=str, required=False, help="lower bounds for the grid"
    )
    parser.add_argument(
        "--max", dest="grid_max", type=str, required=False, help="upper bounds for the grid"
    )
    parser.add_argument(
        "--bin",
        dest="grid_bin",
        type=str,
        default="100,100",
        help="number of bins for the grid",
    )
    parser.add_argument(
        "--fmt",
        dest="fmt",
        type=str,
        default="% 12.6f",
        help="specify the output format",
    )
    parser.add_argument(
        "--deltaFat",
        dest="deltaFat",
        type=float,
        required=False,
        help="calculate the free energy difference between left and right of given c1 value",
    )
    parser.add_argument(
        "--all_stored",
        dest="all_stored",
        action="store_true",
        default=False,
        help="print all the FES stored instead of only the last one",
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
        "--plot",
        dest="plot",
        action="store_true",
        default=True,
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

    if args.kbt is not None:
        kbt = args.kbt
    else:
        kbt = args.temp * KB_KJ_MOL

    return args, kbt

def _parse_grid_options(args, period_x, period_y, center_x, center_y, dim2):
    # Grid bin
    bins = args.grid_bin.split(",")
    grid_bin_x = int(bins[0]) + 1
    grid_bin_y = None
    if dim2:
        if len(bins) != 2:
            raise ValueError(f"{ERROR_PREFIX} two comma separated integers expected after --bin")
        grid_bin_y = int(bins[1]) + 1

    # Grid min
    grid_min_x = None
    grid_min_y = None
    if args.grid_min is None:
        if period_x == 0:
            grid_min_x = np.min(center_x)
    else:
        mins = args.grid_min.split(",")
        if mins[0] == "-pi":
            grid_min_x = -np.pi
        else:
            grid_min_x = float(mins[0])
        if dim2:
            if len(mins) != 2:
                raise ValueError(f"{ERROR_PREFIX} two comma separated floats expected after --min")
            if mins[1] == "-pi":
                grid_min_y = -np.pi
            else:
                grid_min_y = float(mins[1])
            
    # Grid max
    grid_max_x = None
    grid_max_y = None
    if args.grid_max is None:
        if period_x == 0:
            grid_max_x = np.max(center_x)
    else:
        maxs = args.grid_max.split(",")
        if maxs[0] == "pi":
            grid_max_x = np.pi
        else:
            grid_max_x = float(maxs[0])
        if dim2:
            if len(maxs) != 2:
                raise ValueError(f"{ERROR_PREFIX} two comma separated floats expected after --max")
            if maxs[1] == "pi":
                grid_max_y = np.pi
            else:
                grid_max_y = float(maxs[1])

    # Handle periodic defaults if not set
    # Note: The original script logic for defaults when periodic is a bit complex/implicit.
    # Here we assume if it's periodic, min/max MUST be set or inferred from period if possible,
    # but usually periodic CVs have defined ranges.
    # If they are still None here, we might have an issue if period != 0.
    # However, let's rely on what we extracted.
    
    return grid_bin_x, grid_bin_y, grid_min_x, grid_max_x, grid_min_y, grid_max_y

def main(argv=None):
    args, kbt = parse_cli_args(argv)

    # 1. Load Data
    try:
        # Try using pandas which is standard
        # Use open_text_file to handle compression
        with open_text_file(args.filename) as f:
            data_df = pd.read_table(f, sep=r"\s+", header=None, dtype=str, comment=None)
        data = data_df.to_numpy()
    except Exception as e:
        raise RuntimeError(f"{ERROR_PREFIX} failed to read {args.filename}: {e}")

    # 2. Find FIELDS
    fields_pos = []
    tot_lines = data.shape[0]
    for i in range(tot_lines):
        if data[i, 1] == "FIELDS":
            fields_pos.append(i)
    
    if len(fields_pos) == 0:
        raise ValueError(f"{ERROR_PREFIX} no FIELDS found in file {args.filename}")
    
    if len(fields_pos) > 1:
        print(f" a total of {len(fields_pos)} stored states were found")
        if args.all_stored:
            print("  -> all will be printed")
        else:
            print("  -> only the last one will be printed. use --all_stored to instead print them all")
            fields_pos = [fields_pos[-1]]
    
    fields_pos.append(tot_lines)

    # 3. Process each state
    for n in range(len(fields_pos) - 1):
        print(f"   working...   0% of {n/(len(fields_pos)-1):.0%}", end="\r")
        l = fields_pos[n]
        
        # Parse Header
        dim2 = False
        row_len = len(data[l, :])
        # Clean None/NaNs from row to get actual length
        valid_row = [x for x in data[l, :] if isinstance(x, str)]
        row_len = len(valid_row)
        
        # Original script logic:
        # if len(data[l,:])==6: name_cv_x=data[l,3]
        # elif len(data[l,:])==8: dim2=True...
        # But pandas might pad with NaNs/Nones.
        
        # Let's look at the FIELDS line content directly
        # FIELDS time cv1 (cv2) sigma_cv1 (sigma_cv2) height biasfactor
        # 0      1    2   3     4         5           6      7
        
        # Actually original script says:
        # FIELDS time cv1 sigma_cv1 height biasfactor (len 6) -> 1D
        # FIELDS time cv1 cv2 sigma_cv1 sigma_cv2 height biasfactor (len 8) -> 2D
        
        # Let's be robust
        fields = valid_row
        if len(fields) == 6:
            name_cv_x = fields[3]
        elif len(fields) == 8:
            dim2 = True
            name_cv_x = fields[3]
            name_cv_y = fields[4]
        else:
             raise ValueError(f"{ERROR_PREFIX} wrong number of FIELDS in file {args.filename}: only 1 or 2 dimensional bias are supported")

        # Parse Metadata (Action, biasfactor, epsilon, cutoff, zed)
        # Assuming fixed offset from FIELDS line as in original script
        # l+1: Action
        action = data[l+1, 3]
        explore = 'unset'
        if action == "OPES_METAD_state":
            explore = 'no'
        elif action == "OPES_METAD_EXPLORE_state":
            explore = 'yes'
        else:
            raise ValueError(f"{ERROR_PREFIX} This script works only with OPES_METAD_state and OPES_METAD_EXPLORE_state")

        # l+2: biasfactor
        if data[l+2, 2] != 'biasfactor': raise ValueError(f"{ERROR_PREFIX} biasfactor not found")
        
        sf = 1.0
        if explore == 'yes':
            sf = float(data[l+2, 3])
        
        # l+3: epsilon
        if data[l+3, 2] != 'epsilon': raise ValueError(f"{ERROR_PREFIX} epsilon not found")
        epsilon = float(data[l+3, 3])
        
        # l+4: kernel_cutoff
        if data[l+4, 2] != 'kernel_cutoff': raise ValueError(f"{ERROR_PREFIX} kernel_cutoff not found")
        cutoff = float(data[l+4, 3])
        val_at_cutoff = np.exp(-0.5 * cutoff**2)
        
        # l+6: zed
        if data[l+6, 2] != 'zed': raise ValueError(f"{ERROR_PREFIX} zed not found")
        Zed = float(data[l+6, 3])
        
        if explore == 'no':
            if data[l+7, 2] != 'sum_weights': raise ValueError(f"{ERROR_PREFIX} sum_weights not found")
            Zed *= float(data[l+7, 3])
        if explore == 'yes':
            if data[l+9, 2] != 'counter': raise ValueError(f"{ERROR_PREFIX} counter not found")
            Zed *= float(data[l+9, 3])
            
        l += 10 # Skip header

        # Parse Periodicity from #! SET min_...
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

        if args.der and (period_x > 0 or period_y > 0):
             raise ValueError(f"{ERROR_PREFIX} derivatives not supported with periodic CVs, remove --der option")

        if l == fields_pos[n+1]:
            raise ValueError(f"{ERROR_PREFIX} missing data!")

        # Extract Kernels
        # Columns:
        # 1D: time(0), cv1(1), sigma(2), height(3), biasfactor(4) -> Wait, let's check original script indexing
        # Original: center_x=data[l:..., 1], sigma_x=data[..., 2], height=data[..., 3]
        # 2D: center_x=data[..., 1], center_y=data[..., 2], sigma_x=data[..., 3], sigma_y=data[..., 4], height=data[..., 5]
        
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
        gbx, gby, gminx, gmaxx, gminy, gmaxy = _parse_grid_options(
            args, period_x, period_y, center_x, center_y, dim2
        )
        
        # If defaults were not set by CLI, use read values or data min/max
        if gminx is None: gminx = grid_min_x_read if grid_min_x_read is not None else np.min(center_x)
        if gmaxx is None: gmaxx = grid_max_x_read if grid_max_x_read is not None else np.max(center_x)
        if dim2:
            if gminy is None: gminy = grid_min_y_read if grid_min_y_read is not None else np.min(center_y)
            if gmaxy is None: gmaxy = grid_max_y_read if grid_max_y_read is not None else np.max(center_y)

        # Build Grid Objects
        # GridAxis: name, values, minimum, maximum, bins, period
        
        # X Axis
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
        
        # Y Axis
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
        
        # Create GridData
        axes_list = [axis_x]
        mesh = None
        if axis_y is not None:
            axes_list.append(axis_y)
            mesh = np.meshgrid(axis_x.values, axis_y.values, indexing="ij")
        
        grid_data = GridData(axes=tuple(axes_list), mesh=mesh)
        
        grid_state = create_grid_runtime_state(grid_data, args.der)
        
        # Prepare Kernel Evaluation
        # We need to adapt the problem to use KernelEvaluator.
        # KernelEvaluator expects: bias, cv_x, cv_y
        # And computes: sum exp(bias - dist^2/2sigma^2)
        # Here we have: sum height * exp(-dist^2/2sigma^2)
        # So we set bias = log(height)
        # BUT: height can be negative? No, probability kernels are positive.
        # Wait, OPES kernels can be negative? "height" in OPES state file.
        # Usually they are probability increments.
        # Let's assume height > 0. If not, we might have issues with log.
        # Original script: kernels_i=height*(np.maximum(np.exp(-0.5*dist_x*dist_x)-val_at_cutoff,0))
        # It subtracts val_at_cutoff!
        # Our KernelEvaluator does NOT support subtracting val_at_cutoff inside the exp summation directly.
        # Our KernelEvaluator computes standard KDE.
        # OPES kernels are truncated.
        # "kernels_i=height*(np.maximum(np.exp(-0.5*dist_x*dist_x)-val_at_cutoff,0))"
        
        # If we ignore val_at_cutoff (usually small), we can use KernelEvaluator.
        # If cutoff is large, val_at_cutoff is negligible.
        # OPES usually has cutoff=4 sigma, so exp(-8) ~ 3e-4.
        # If we want exact reproduction, we might need to modify KernelEvaluator or use a custom loop.
        # Given the user wants Numba acceleration, and our KernelEvaluator is Numba accelerated,
        # we should try to use it.
        # However, the subtraction of `val_at_cutoff` is a specific feature of OPES compressed kernels to make them strictly compact support.
        
        # Let's check if we can support this in KernelEvaluator.
        # It would require adding `val_at_cutoff` and `height` arrays to it.
        # But KernelEvaluator is designed for Reweighting (bias values), not direct kernel summation.
        # In Reweighting: weight = exp(bias - ...).
        # Here: weight = height * (exp(...) - C).
        
        # If we assume exp(...) >> C, then weight ~ height * exp(...) = exp(log(height) - ...).
        # This matches Reweighting form with bias = log(height).
        
        # Is the approximation acceptable?
        # The original script is "Get the FES estimate used by OPES".
        # OPES uses the truncated kernels.
        # If we drop the truncation term, we introduce a small error at the tails of the kernels.
        # However, for FES visualization, this might be acceptable.
        
        # BUT, there is another issue: `sigma` is an array here!
        # In `KernelEvaluator`, `sigma` is a scalar (KernelParams).
        # OPES kernels can have variable widths (adaptive sigma).
        # `KernelEvaluator` assumes fixed sigma for the whole chunk.
        
        # If sigmas are variable, we CANNOT use the current `KernelEvaluator` as is.
        # We would need to pass sigma arrays.
        
        # Let's check `KernelEvaluator` again.
        # `_numba_fes_1d` takes `sigma_x_local` which is a float.
        
        # So, to support `FES_from_State` correctly (variable sigma + truncation),
        # we need a NEW Numba kernel or modify the existing one.
        # Since `calcFES.py` (reweighting) uses fixed sigma, we shouldn't break that.
        
        # I will define a specialized Numba function for this script LOCALLY, 
        # or add it to `kernel_eval.py` if it's general enough.
        # Since this is "FES from State", it's a different algorithm (Sum of Kernels vs Reweighting).
        # I'll define it locally in this script to avoid complicating `FESutils` with OPES-specific state logic for now,
        # OR I can add it to `kernel_eval.py` as `_numba_sum_kernels`.
        
        # Given the instructions to "integrate", putting it in `kernel_eval.py` seems better for code organization.
        # But `kernel_eval.py` is currently focused on the Reweighting approach.
        
        # Let's define the Numba functions inside this script for now.
        # It's cleaner than modifying the shared module with single-use code.
        
        # We need to handle:
        # 1. Variable sigma (array).
        # 2. Height (array).
        # 3. Truncation (val_at_cutoff).
        
        # Also, we need to handle the `epsilon` and `Zed` normalization.
        # prob = sum(kernels) / Zed + epsilon
        # fes = -kbt * sf * log(prob / max_prob)
        
        # Let's implement the Numba kernel for this.
        
        # Prepare arrays for Numba
        # center_x, center_y, sigma_x, sigma_y, height are already arrays.
        
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
            if args.der:
                der_prob_x = _numba_calc_der_prob_1d(
                    grid_state.axis_x.values,
                    center_x,
                    sigma_x,
                    height,
                    period_x,
                    val_at_cutoff
                )
        else:
            # Meshgrid for 2D
            # grid_state.mesh might be None if not initialized?
            # GridRuntimeState doesn't auto-init mesh.
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
            if args.der:
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
        
        if args.der:
            der_prob_x = der_prob_x / Zed
            if dim2:
                der_prob_y = der_prob_y / Zed

        # Mintozero
        max_prob = 1.0
        if not args.nomintozero:
            max_prob = np.max(prob)
        
        # Calc FES
        # fes = -kbt * sf * log(prob/max_prob)
        #     = -kbt * sf * (log(prob) - log(max_prob))
        grid_state.fes = -kbt * sf * np.log(prob / max_prob)
        
        if args.der:
            # der_fes = -kbt * sf * (1/prob) * der_prob
            grid_state.der_fes_x = -kbt * sf / prob * der_prob_x
            if dim2:
                grid_state.der_fes_y = -kbt * sf / prob * der_prob_y

        # Output
        out_name = args.outfile
        if args.all_stored:
            # Handle filename numbering: outfile_n % (n+1)
            # Original logic:
            # if outfile.rfind('/')==-1: ...
            # Simplified:
            base, ext = out_name.rsplit('.', 1) if '.' in out_name else (out_name, '')
            out_name = f"{base}_{n+1}.{ext}" if ext else f"{base}_{n+1}"

        options = OutputOptions(
            fmt=args.fmt,
            mintozero=(not args.nomintozero),
            calc_deltaF=(args.deltaFat is not None),
            delta_threshold=args.deltaFat,
            kbt=kbt,
            backup=args.backup
        )
        
        stats = SampleStats(
            size=len(center_x),
            effective_size=len(center_x) # Not really applicable here, but required by dataclass
        )
        
        names = (name_cv_x, name_cv_y if dim2 else None)
        mesh = (x_mesh, y_mesh) if dim2 else None
        
        write_standard_output(out_name, grid_state, names, options, stats, mesh)
        print(f"   printed to {out_name}")
        
        # Plotting
        if args.plot:
            plot_manager = PlotManager(
                enabled=True,
                dim2=dim2,
                axis_x=axis_x,
                axis_y=axis_y,
                mintozero=options.mintozero,
                mesh=mesh
            )
            plot_manager.record_standard_surface(grid_state.fes)
            
            # Generate png name
            if out_name.lower().endswith(".dat"):
                png_name = out_name[:-4] + ".png"
            else:
                png_name = out_name + ".png"
                
            plot_manager.save_standard_plot(png_name, names)
            print(f"   plotted to {png_name}")

    print("Done.")

# --- Numba Kernels ---
from numba import njit, prange

@njit
def _normalized_distance(point, value, sigma, period):
    if period <= 0.0:
        return (point - value) / sigma
    delta = np.abs(point - value)
    wrapped = period - delta
    if wrapped < delta:
        delta = wrapped
    return delta / sigma

@njit(parallel=True)
def _numba_calc_prob_1d(grid_x, center_x, sigma_x, height, period_x, val_at_cutoff):
    n_grid = len(grid_x)
    n_kernels = len(center_x)
    prob = np.zeros(n_grid)
    
    for i in prange(n_grid):
        p_val = 0.0
        gx = grid_x[i]
        for k in range(n_kernels):
            dist = _normalized_distance(gx, center_x[k], sigma_x[k], period_x)
            # kernels_i=height*(np.maximum(np.exp(-0.5*dist_x*dist_x)-val_at_cutoff,0))
            gauss = np.exp(-0.5 * dist * dist)
            val = gauss - val_at_cutoff
            if val > 0:
                p_val += height[k] * val
        prob[i] = p_val
    return prob

@njit(parallel=True)
def _numba_calc_der_prob_1d(grid_x, center_x, sigma_x, height, period_x, val_at_cutoff):
    n_grid = len(grid_x)
    n_kernels = len(center_x)
    der = np.zeros(n_grid)
    
    for i in prange(n_grid):
        d_val = 0.0
        gx = grid_x[i]
        for k in range(n_kernels):
            dist_norm = _normalized_distance(gx, center_x[k], sigma_x[k], period_x)
            dist_sq = dist_norm * dist_norm
            gauss = np.exp(-0.5 * dist_sq)
            val = gauss - val_at_cutoff
            if val > 0:
                # der = sum( -dist/sigma * height * gauss )
                # Note: derivative of max(f(x)-C, 0) is f'(x) * step(f(x)-C)
                # So we only add derivative if val > 0
                # grad of exp(-0.5*(x-c)^2/s^2) -> exp(...) * (-0.5) * 2 * (x-c)/s^2 * (1)
                # = - (x-c)/s^2 * gauss
                # = - (dist_norm * s / s^2) * gauss ? No.
                # dist_norm = |x-c|/s.
                # let y = (x-c)/s. exp(-0.5 y^2).
                # d/dx = exp(...) * (-y) * dy/dx = exp(...) * (-y) * (1/s)
                # = - (y/s) * gauss = - (dist_norm/s) * gauss * sign(x-c)
                
                # Wait, _normalized_distance returns ABSOLUTE distance / sigma.
                # We need signed distance for derivative.
                
                # Let's re-implement distance logic inline or return signed dist.
                # Or just use: dist_real = (x-c) wrapped.
                # grad = - dist_real / sigma^2 * gauss
                
                # Re-eval distance for derivative:
                dx = gx - center_x[k]
                if period_x > 0:
                    # Wrap dx to [-period/2, period/2]
                    half_p = period_x / 2.0
                    if dx > half_p: dx -= period_x
                    elif dx < -half_p: dx += period_x
                
                grad_term = -dx / (sigma_x[k]**2) * gauss
                d_val += height[k] * grad_term
        der[i] = d_val
    return der

@njit(parallel=True)
def _numba_calc_prob_2d(grid_x, grid_y, center_x, center_y, sigma_x, sigma_y, height, period_x, period_y, val_at_cutoff):
    nx, ny = grid_x.shape
    n_kernels = len(center_x)
    prob = np.zeros((nx, ny))
    
    for i in prange(nx):
        for j in range(ny):
            gx = grid_x[i, j]
            gy = grid_y[i, j]
            p_val = 0.0
            for k in range(n_kernels):
                dx = _normalized_distance(gx, center_x[k], sigma_x[k], period_x)
                dy = _normalized_distance(gy, center_y[k], sigma_y[k], period_y)
                dist_sq = dx*dx + dy*dy
                gauss = np.exp(-0.5 * dist_sq)
                val = gauss - val_at_cutoff
                if val > 0:
                    p_val += height[k] * val
            prob[i, j] = p_val
    return prob

@njit(parallel=True)
def _numba_calc_der_prob_2d(grid_x, grid_y, center_x, center_y, sigma_x, sigma_y, height, period_x, period_y, val_at_cutoff):
    nx, ny = grid_x.shape
    n_kernels = len(center_x)
    der_x = np.zeros((nx, ny))
    der_y = np.zeros((nx, ny))
    
    for i in prange(nx):
        for j in range(ny):
            gx = grid_x[i, j]
            gy = grid_y[i, j]
            dx_val = 0.0
            dy_val = 0.0
            for k in range(n_kernels):
                # Need signed distances
                diff_x = gx - center_x[k]
                if period_x > 0:
                    hp = period_x/2
                    if diff_x > hp: diff_x -= period_x
                    elif diff_x < -hp: diff_x += period_x
                
                diff_y = gy - center_y[k]
                if period_y > 0:
                    hp = period_y/2
                    if diff_y > hp: diff_y -= period_y
                    elif diff_y < -hp: diff_y += period_y
                
                sx = sigma_x[k]
                sy = sigma_y[k]
                
                # Normalized abs dists for check
                ndx = np.abs(diff_x)/sx
                ndy = np.abs(diff_y)/sy
                
                dist_sq = ndx*ndx + ndy*ndy
                gauss = np.exp(-0.5 * dist_sq)
                val = gauss - val_at_cutoff
                
                if val > 0:
                    # grad_x = - diff_x / sx^2 * gauss
                    dx_val += height[k] * (-diff_x / (sx*sx) * gauss)
                    dy_val += height[k] * (-diff_y / (sy*sy) * gauss)
            
            der_x[i, j] = dx_val
            der_y[i, j] = dy_val
    return der_x, der_y


if __name__ == "__main__":
    main()
