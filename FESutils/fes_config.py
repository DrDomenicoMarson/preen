from dataclasses import dataclass

@dataclass(frozen=True)
class FESConfig:
    """Configuration options for FES calculations."""

    filename: str
    outfile: str
    sigma: tuple[float, ...]
    kbt: float
    cv_spec: tuple[str, ...]
    bias_spec: str
    grid_min: None | tuple[float, ...]
    grid_max: None | tuple[float, ...]
    grid_bin: tuple[int, ...]
    blocks_num: int
    block_seed: None | int
    stride: int
    random_blocks: bool
    skiprows: int
    mintozero: bool
    reverse: bool
    calc_der: bool
    delta_f_threshold: None | float
    fmt: str
    plot: bool = False
    backup: bool = False

    @property
    def dimension(self) -> int:
        return len(self.cv_spec)

    @property
    def calculate_delta_f(self) -> bool:
        return self.delta_f_threshold is not None
