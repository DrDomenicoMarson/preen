
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
