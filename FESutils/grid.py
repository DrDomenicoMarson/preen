
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .colvar_io import ColvarData, CVInfo
from .fes_config import FESConfig
from .constants import ERROR_PREFIX


@dataclass
class GridAxis:
    name: str
    values: NDArray
    minimum: float
    maximum: float
    bins: int
    period: float


@dataclass
class GridData:
    axes: tuple[GridAxis, ...]
    mesh: tuple[NDArray, NDArray] | None


def build_grid(config: FESConfig, data: ColvarData) -> GridData:
    axis_x = _build_axis(0, config, data)
    axes = [axis_x]
    mesh = None
    if config.dimension == 2:
        axis_y = _build_axis(1, config, data)
        axes.append(axis_y)
        mesh = np.meshgrid(axis_x.values, axis_y.values, indexing="ij")
    return GridData(axes=tuple(axes), mesh=mesh)


def _build_axis(idx: int, config: FESConfig, data: ColvarData) -> GridAxis:
    cv_info = data.metadata.cvs[idx]
    samples = data.cv_values[idx]
    bin_count = _resolve_bin_count(idx, config)
    min_value = _resolve_bound(idx, config.grid_min, cv_info, samples, "min")
    max_value = _resolve_bound(idx, config.grid_max, cv_info, samples, "max")
    grid_values = np.linspace(min_value, max_value, bin_count + 1)
    period = cv_info.period
    if period != 0 and np.isclose(period, grid_values[-1] - grid_values[0]):
        grid_values = grid_values[:-1]
    bins = grid_values.shape[0]
    return GridAxis(
        name=cv_info.name,
        values=grid_values,
        minimum=min_value,
        maximum=max_value,
        bins=bins,
        period=period,
    )


def _resolve_bin_count(idx: int, config: FESConfig) -> int:
    if idx >= len(config.grid_bin):
        if config.dimension == 2:
            raise ValueError(
                f"{ERROR_PREFIX} two comma separated integers expected after --bin"
            )
        return config.grid_bin[0]
    return config.grid_bin[idx]


def _resolve_bound(
    idx: int,
    bounds: Sequence[float] | None,
    cv_info: CVInfo,
    samples: NDArray,
    option: str,
) -> float:
    explicit = None
    if bounds is not None:
        if idx >= len(bounds):
            raise ValueError(
                f"{ERROR_PREFIX} two comma separated floats expected after --{option}"
            )
        explicit = bounds[idx]
    if explicit is not None:
        return explicit
    if cv_info.period != 0 and option == "max" and cv_info.grid_max is not None:
        return cv_info.grid_max
    if cv_info.period != 0 and option == "min" and cv_info.grid_min is not None:
        return cv_info.grid_min
    if option == "min":
        return float(np.min(samples))
    return float(np.max(samples))
