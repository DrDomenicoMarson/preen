import math
import os
from dataclasses import dataclass
from FESutils.constants import KB_KJ_MOL, normalize_energy_unit


@dataclass(frozen=True)
class BaseFESConfig:
    """Common configuration for FES calculations."""

    filename: str
    outfile: str
    kbt: None | float = None
    temp: None | float = None
    grid_min: None | tuple[float, ...] = None
    grid_max: None | tuple[float, ...] = None
    grid_bin: tuple[int, ...] = (100, 100)
    mintozero: bool = True
    reverse: bool = False
    calc_der: bool = False
    fmt: str = "% 12.6f"
    plot: bool = False
    backup: bool = False
    input_energy_unit: str = "kJ/mol"
    output_energy_unit: str = "kJ/mol"
    num_threads: int | None = None

    def __post_init__(self):
        if self.kbt is None:
            if self.temp is None:
                raise ValueError("Either 'kbt' or 'temp' must be provided.")
            if self.temp <= 0:
                raise ValueError("Temperature must be positive.")
            object.__setattr__(self, "kbt", self.temp * KB_KJ_MOL)
        elif self.temp is not None:
            if self.temp <= 0:
                raise ValueError("Temperature must be positive.")
            if self.kbt <= 0:
                raise ValueError("kbt must be positive.")
            expected_kbt = self.temp * KB_KJ_MOL
            if not math.isclose(self.kbt, expected_kbt, rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError(
                    "Conflicting kbt and temp provided: "
                    f"kbt={self.kbt} vs temp-derived {expected_kbt} (kJ/mol). "
                    "Provide only one or make them consistent."
                )
        elif self.kbt is not None and self.kbt <= 0:
            raise ValueError("kbt must be positive.")
        object.__setattr__(self, "input_energy_unit", normalize_energy_unit(self.input_energy_unit))
        object.__setattr__(self, "output_energy_unit", normalize_energy_unit(self.output_energy_unit))
        threads = self.num_threads
        if threads is None:
            threads = max(1, min(16, os.cpu_count() or 1))
        elif threads <= 0:
            raise ValueError("num_threads must be positive")
        object.__setattr__(self, "num_threads", int(threads))


@dataclass(frozen=True)
class FESConfig(BaseFESConfig):
    """Configuration options for COLVAR-based FES calculations."""

    sigma: None | tuple[float, ...] = None
    cv_spec: None | tuple[str, ...] = None
    bias_spec: None | str = None
    blocks_num: int = 1
    block_seed: None | int = None
    stride: int = 0
    random_blocks: bool = False
    skiprows: int = 0
    delta_f_threshold: None | float = None

    @property
    def dimension(self) -> int:
        return len(self.cv_spec)

    @property
    def calculate_delta_f(self) -> bool:
        return self.delta_f_threshold is not None


@dataclass(frozen=True)
class FESStateConfig(BaseFESConfig):
    """Configuration options for STATE-to-FES calculations."""
    # Inherits common options; STATE files embed CV/bias info so no sigma/cv_spec/bias_spec needed.
