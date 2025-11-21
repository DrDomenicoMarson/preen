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
    grid_min: None | tuple[float, ...] = None
    grid_max: None | tuple[float, ...] = None
    grid_bin: tuple[int, ...] = (100, 100)
    blocks_num: int = 1
    block_seed: None | int = None
    stride: int = 0
    random_blocks: bool = False
    skiprows: int = 0
    mintozero: bool = True
    reverse: bool = False
    calc_der: bool = False
    delta_f_threshold: None | float = None
    fmt: str = "% 12.6f"
    plot: bool = False
    backup: bool = False

    @property
    def dimension(self) -> int:
        return len(self.cv_spec)

    @property
    def calculate_delta_f(self) -> bool:
        return self.delta_f_threshold is not None
