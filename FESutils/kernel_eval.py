from dataclasses import dataclass
import numpy as np
from numba import njit, prange

@dataclass
class KernelParams:
    sigma_x: float
    sigma_y: float | None
    period_x: float
    period_y: float
    kbt: float


class KernelEvaluator:
    def __init__(self, dim2: bool, calc_der: bool):
        self.dim2 = dim2
        self.calc_der = calc_der

    def fill_chunk(
        self, sample_state, start: int, end: int, grid_state, params: KernelParams
    ):
        cv_x = sample_state.cv_x
        bias = sample_state.bias
        grid_x = grid_state.axis_x.values
        
        cv_chunk = cv_x[start:end]
        bias_chunk = bias[start:end]
        
        if not self.dim2:
            if not self.calc_der:
                grid_state.fes[:] = _numba_fes_1d(
                    cv_chunk,
                    bias_chunk,
                    params.sigma_x,
                    grid_x,
                    params.period_x,
                    params.kbt,
                )
            else:
                fes_chunk, der_chunk = _numba_fes_der_1d(
                    cv_chunk,
                    bias_chunk,
                    params.sigma_x,
                    grid_x,
                    params.period_x,
                    params.kbt,
                )
                grid_state.fes[:] = fes_chunk
                grid_state.der_fes_x[:] = der_chunk
        else:
            cv_y = sample_state.cv_y
            if cv_y is None:
                raise RuntimeError("missing cv_y for 2D evaluation")
            cvy_chunk = cv_y[start:end]
            grid_y = grid_state.axis_y.values
            if not self.calc_der:
                grid_state.fes[:, :] = _numba_fes_2d(
                    cv_chunk,
                    cvy_chunk,
                    bias_chunk,
                    params.sigma_x,
                    params.sigma_y,
                    grid_x,
                    grid_y,
                    params.period_x,
                    params.period_y,
                    params.kbt,
                )
            else:
                fes_chunk, derx_chunk, dery_chunk = _numba_fes_der_2d(
                    cv_chunk,
                    cvy_chunk,
                    bias_chunk,
                    params.sigma_x,
                    params.sigma_y,
                    grid_x,
                    grid_y,
                    params.period_x,
                    params.period_y,
                    params.kbt,
                )
                grid_state.fes[:, :] = fes_chunk
                grid_state.der_fes_x[:, :] = derx_chunk
                grid_state.der_fes_y[:, :] = dery_chunk


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
def _numba_fes_1d(
    cv_x_local, bias_local, sigma_x_local, grid_x, period_x_local, kbt_local
):
    grid_len = grid_x.shape[0]
    fes_out = np.empty(grid_len, dtype=np.float64)
    samples = bias_local.shape[0]
    for grid_idx in prange(grid_len):
        point_x = grid_x[grid_idx]
        max_arg = -np.inf
        total = 0.0
        for sample_idx in range(samples):
            dist_x = _normalized_distance(
                point_x, cv_x_local[sample_idx], sigma_x_local, period_x_local
            )
            arg = bias_local[sample_idx] - 0.5 * dist_x * dist_x
            if max_arg == -np.inf:
                max_arg = arg
                total = 1.0
            elif arg > max_arg:
                scale = np.exp(max_arg - arg)
                total = total * scale + 1.0
                max_arg = arg
            else:
                total += np.exp(arg - max_arg)
        fes_out[grid_idx] = -kbt_local * (max_arg + np.log(total))
    return fes_out

@njit(parallel=True)
def _numba_fes_der_1d(
    cv_x_local, bias_local, sigma_x_local, grid_x, period_x_local, kbt_local
):
    grid_len = grid_x.shape[0]
    fes_out = np.empty(grid_len, dtype=np.float64)
    der_out = np.empty(grid_len, dtype=np.float64)
    samples = bias_local.shape[0]
    for grid_idx in prange(grid_len):
        point_x = grid_x[grid_idx]
        max_arg = -np.inf
        total = 0.0
        der_sum = 0.0
        for sample_idx in range(samples):
            dist_x = _normalized_distance(
                point_x, cv_x_local[sample_idx], sigma_x_local, period_x_local
            )
            arg = bias_local[sample_idx] - 0.5 * dist_x * dist_x
            grad = -dist_x / sigma_x_local
            if max_arg == -np.inf:
                max_arg = arg
                total = 1.0
                der_sum = grad
            elif arg > max_arg:
                scale = np.exp(max_arg - arg)
                total = total * scale + 1.0
                der_sum = der_sum * scale + grad
                max_arg = arg
            else:
                exp_term = np.exp(arg - max_arg)
                total += exp_term
                der_sum += grad * exp_term
        fes_out[grid_idx] = -kbt_local * (max_arg + np.log(total))
        der_out[grid_idx] = -kbt_local * (der_sum / total)
    return fes_out, der_out

@njit(parallel=True)
def _numba_fes_2d(
    cv_x_local,
    cv_y_local,
    bias_local,
    sigma_x_local,
    sigma_y_local,
    grid_x,
    grid_y,
    period_x_local,
    period_y_local,
    kbt_local,
):
    grid_x_len = grid_x.shape[0]
    grid_y_len = grid_y.shape[0]
    fes_out = np.empty((grid_x_len, grid_y_len), dtype=np.float64)
    samples = bias_local.shape[0]
    for grid_i in prange(grid_x_len):
        point_x = grid_x[grid_i]
        for grid_j in range(grid_y_len):
            point_y = grid_y[grid_j]
            max_arg = -np.inf
            total = 0.0
            for sample_idx in range(samples):
                dist_x = _normalized_distance(
                    point_x, cv_x_local[sample_idx], sigma_x_local, period_x_local
                )
                dist_y = _normalized_distance(
                    point_y, cv_y_local[sample_idx], sigma_y_local, period_y_local
                )
                arg = bias_local[sample_idx] - 0.5 * (
                    dist_x * dist_x + dist_y * dist_y
                )
                if max_arg == -np.inf:
                    max_arg = arg
                    total = 1.0
                elif arg > max_arg:
                    scale = np.exp(max_arg - arg)
                    total = total * scale + 1.0
                    max_arg = arg
                else:
                    total += np.exp(arg - max_arg)
            fes_out[grid_i, grid_j] = -kbt_local * (max_arg + np.log(total))
    return fes_out

@njit(parallel=True)
def _numba_fes_der_2d(
    cv_x_local,
    cv_y_local,
    bias_local,
    sigma_x_local,
    sigma_y_local,
    grid_x,
    grid_y,
    period_x_local,
    period_y_local,
    kbt_local,
):
    grid_x_len = grid_x.shape[0]
    grid_y_len = grid_y.shape[0]
    fes_out = np.empty((grid_x_len, grid_y_len), dtype=np.float64)
    der_x_out = np.empty((grid_x_len, grid_y_len), dtype=np.float64)
    der_y_out = np.empty((grid_x_len, grid_y_len), dtype=np.float64)
    samples = bias_local.shape[0]
    for grid_i in prange(grid_x_len):
        point_x = grid_x[grid_i]
        for grid_j in range(grid_y_len):
            point_y = grid_y[grid_j]
            max_arg = -np.inf
            total = 0.0
            der_x_sum = 0.0
            der_y_sum = 0.0
            for sample_idx in range(samples):
                dist_x = _normalized_distance(
                    point_x, cv_x_local[sample_idx], sigma_x_local, period_x_local
                )
                dist_y = _normalized_distance(
                    point_y, cv_y_local[sample_idx], sigma_y_local, period_y_local
                )
                arg = bias_local[sample_idx] - 0.5 * (
                    dist_x * dist_x + dist_y * dist_y
                )
                grad_x = -dist_x / sigma_x_local
                grad_y = -dist_y / sigma_y_local
                if max_arg == -np.inf:
                    max_arg = arg
                    total = 1.0
                    der_x_sum = grad_x
                    der_y_sum = grad_y
                elif arg > max_arg:
                    scale = np.exp(max_arg - arg)
                    total = total * scale + 1.0
                    der_x_sum = der_x_sum * scale + grad_x
                    der_y_sum = der_y_sum * scale + grad_y
                    max_arg = arg
                else:
                    exp_term = np.exp(arg - max_arg)
                    total += exp_term
                    der_x_sum += grad_x * exp_term
                    der_y_sum += grad_y * exp_term
            fes_out[grid_i, grid_j] = -kbt_local * (max_arg + np.log(total))
            der_x_out[grid_i, grid_j] = -kbt_local * (der_x_sum / total)
            der_y_out[grid_i, grid_j] = -kbt_local * (der_y_sum / total)
    return fes_out, der_x_out, der_y_out
