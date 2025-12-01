
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from .colvar_io import ColvarData
from .fes_config import FESConfig
from .grid import GridAxis, GridData


@dataclass
class SampleState:
    name_cv_x: str
    cv_x: NDArray
    bias: NDArray
    len_tot: int
    name_cv_y: str | None = None
    cv_y: NDArray | None = None

    @property
    def dim2(self) -> bool:
        return self.cv_y is not None

    def apply_permutation(self, perm: NDArray) -> None:
        self.cv_x = self.cv_x[perm]
        if self.cv_y is not None:
            self.cv_y = self.cv_y[perm]
        self.bias = self.bias[perm]


def create_sample_state(colvar_data: ColvarData) -> SampleState:
    name_cv_x = colvar_data.metadata.cvs[0].name
    cv_x = colvar_data.cv_values[0]
    bias = colvar_data.bias
    if len(colvar_data.cv_values) > 1:
        name_cv_y = colvar_data.metadata.cvs[1].name
        cv_y = colvar_data.cv_values[1]
    else:
        name_cv_y = None
        cv_y = None
    return SampleState(
        name_cv_x=name_cv_x,
        cv_x=cv_x,
        bias=bias,
        len_tot=len(cv_x),
        name_cv_y=name_cv_y,
        cv_y=cv_y,
    )


@dataclass
class GridRuntimeState:
    grid: GridData
    axis_x: GridAxis
    axis_y: GridAxis | None
    mesh: tuple[NDArray, NDArray] | None
    fes: NDArray
    der_fes_x: NDArray | None
    der_fes_y: NDArray | None

    @property
    def dim2(self) -> bool:
        return self.axis_y is not None


def create_grid_runtime_state(grid: GridData, calc_der: bool) -> GridRuntimeState:
    axis_x = grid.axes[0]
    axis_y = grid.axes[1] if len(grid.axes) > 1 else None
    if axis_y is None:
        fes = np.zeros(axis_x.bins)
        der_x = np.zeros(axis_x.bins) if calc_der else None
        der_y = None
        mesh = None
    else:
        fes = np.zeros((axis_x.bins, axis_y.bins))
        if calc_der:
            der_x = np.zeros_like(fes)
            der_y = np.zeros_like(fes)
        else:
            der_x = None
            der_y = None
        mesh = grid.mesh
        if mesh is None:
            mesh = np.meshgrid(axis_x.values, axis_y.values, indexing="ij")
    return GridRuntimeState(
        grid=grid,
        axis_x=axis_x,
        axis_y=axis_y,
        mesh=grid.mesh if axis_y is None else mesh,
        fes=fes,
        der_fes_x=der_x,
        der_fes_y=der_y,
    )


@dataclass
class BlockRuntimeState:
    enabled: bool
    stride: int
    blocks_num: int
    logweight: NDArray | None
    fes_storage: NDArray | None


def initialize_block_state(
    config: FESConfig, total_samples: int, fes_shape: Sequence[int]
) -> BlockRuntimeState:
    stride = config.stride
    blocks_num = config.blocks_num
    logweight = None
    fes_storage = None
    enabled = False
    if blocks_num != 1:
        enabled = True
        stride = max(1, int(total_samples / blocks_num))
        blocks_num = max(1, int(total_samples / stride))
        logweight = np.zeros(blocks_num)
        fes_storage = np.zeros((blocks_num,) + tuple(fes_shape))
    if stride == 0 or stride > total_samples:
        stride = total_samples
    return BlockRuntimeState(
        enabled=enabled,
        stride=stride,
        blocks_num=blocks_num,
        logweight=logweight,
        fes_storage=fes_storage,
    )


def symmetrize_array(arr: NDArray, sym_x: bool, sym_y: bool, merged_x: bool, merged_y: bool, axis_y_bins: int) -> NDArray:
    """Helper to symmetrize a 1D or 2D array."""
    new_arr = arr
    if arr.ndim == 1:
        if sym_x:
            neg = arr[::-1]
            new_arr = np.concatenate((neg, arr))
            if merged_x:
                new_arr = np.delete(new_arr, len(arr)-1)
    elif arr.ndim == 2:
        if sym_x:
            neg = new_arr[::-1, :]
            new_arr = np.concatenate((neg, new_arr), axis=0)
            if merged_x:
                new_arr = np.delete(new_arr, len(arr)-1, axis=0)
        if sym_y:
            neg = new_arr[:, ::-1]
            new_arr = np.concatenate((neg, new_arr), axis=1)
            if merged_y:
                # axis_y_bins is the original length
                new_arr = np.delete(new_arr, axis_y_bins - 1, axis=1)
    return new_arr

def symmetrize_grid_state(grid_state: GridRuntimeState, symmetrize_cvs: list[str] | None) -> GridRuntimeState:
    """
    Unfold the grid state for symmetric CVs.
    Mirrors the grid and data from [0, max] to [-max, max].
    """
    if not symmetrize_cvs:
        return grid_state

    axis_x = grid_state.axis_x
    axis_y = grid_state.axis_y
    fes = grid_state.fes
    der_x = grid_state.der_fes_x
    der_y = grid_state.der_fes_y
    
    sym_x = axis_x.name in symmetrize_cvs
    sym_y = axis_y is not None and axis_y.name in symmetrize_cvs
    
    if not sym_x and not sym_y:
        return grid_state

    # Helper to symmetrize an axis
    def _sym_axis(axis: GridAxis):
        v = axis.values
        v_neg = -v[::-1]
        v_new = np.concatenate((v_neg, v))
        merged = False
        if np.isclose(v_new[len(v)-1], v_new[len(v)]):
             v_new = np.delete(v_new, len(v)-1)
             merged = True
        return v_new, merged

    merged_x = False
    merged_y = False

    if sym_x:
        vals_x, merged_x = _sym_axis(axis_x)
        new_axis_x = GridAxis(
            name=axis_x.name,
            values=vals_x,
            minimum=vals_x[0],
            maximum=vals_x[-1],
            bins=len(vals_x),
            period=axis_x.period
        )
    else:
        new_axis_x = axis_x

    if sym_y:
        vals_y, merged_y = _sym_axis(axis_y)
        new_axis_y = GridAxis(
            name=axis_y.name,
            values=vals_y,
            minimum=vals_y[0],
            maximum=vals_y[-1],
            bins=len(vals_y),
            period=axis_y.period
        )
    else:
        new_axis_y = axis_y

    # Symmetrize data
    orig_y_bins = axis_y.bins if axis_y else 0
    new_fes = symmetrize_array(fes, sym_x, sym_y, merged_x, merged_y, orig_y_bins)
    
    new_der_x = None
    new_der_y = None
    if der_x is not None:
        # For derivatives, we need to flip sign when mirroring
        # dF/d(-x) = -dF/dx (odd)
        # dF/dy(-x, y) = dF/dy(x, y) (even wrt x)
        
        # This is getting complicated with the helper.
        # Let's do it manually for derivatives or expand helper.
        # Actually, let's just do it inline for derivatives to be safe.
        
        dx = der_x
        dy = der_y
        
        if not grid_state.dim2:
            if sym_x:
                dx_neg = -dx[::-1]
                new_der_x = np.concatenate((dx_neg, dx))
                if merged_x:
                    new_der_x = np.delete(new_der_x, len(dx)-1)
            else:
                new_der_x = dx
            new_der_y = None
        else:
            # 2D
            if sym_x:
                dx_neg = -dx[::-1, :]
                dx = np.concatenate((dx_neg, dx), axis=0)
                if merged_x: dx = np.delete(dx, len(der_x)-1, axis=0)
                
                dy_neg = dy[::-1, :] # Even wrt x
                dy = np.concatenate((dy_neg, dy), axis=0)
                if merged_x: dy = np.delete(dy, len(der_y)-1, axis=0)
            
            if sym_y:
                dx_neg = dx[:, ::-1] # Even wrt y
                dx = np.concatenate((dx_neg, dx), axis=1)
                if merged_y: dx = np.delete(dx, orig_y_bins-1, axis=1)
                
                dy_neg = -dy[:, ::-1] # Odd wrt y
                dy = np.concatenate((dy_neg, dy), axis=1)
                if merged_y: dy = np.delete(dy, orig_y_bins-1, axis=1)
            
            new_der_x = dx
            new_der_y = dy

    # Rebuild mesh
    new_mesh = None
    if new_axis_y is not None:
        new_mesh = np.meshgrid(new_axis_x.values, new_axis_y.values, indexing="ij")
        
    return GridRuntimeState(
        grid=GridData(axes=(new_axis_x,) if new_axis_y is None else (new_axis_x, new_axis_y), mesh=new_mesh),
        axis_x=new_axis_x,
        axis_y=new_axis_y,
        mesh=new_mesh,
        fes=new_fes,
        der_fes_x=new_der_x,
        der_fes_y=new_der_y,
    )
