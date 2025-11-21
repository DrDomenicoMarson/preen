import math
from dataclasses import dataclass
from FESutils.constants import KB_KJ_MOL, normalize_energy_unit

@dataclass(frozen=True)
class FESConfig:
    """Configuration options for FES calculations."""

    filename: str
    outfile: str
    kbt: None | float = None
    temp: None | float = None
    sigma: None | tuple[float, ...] = None
    cv_spec: None | tuple[str, ...] = None
    bias_spec: None | str = None
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
    input_energy_unit: str = "kJ/mol"
    output_energy_unit: str = "kJ/mol"

    def __post_init__(self):
        if self.kbt is None:
            if self.temp is None:
                raise ValueError("Either 'kbt' or 'temp' must be provided.")
            # Calculate kbt from temp
            object.__setattr__(self, 'kbt', self.temp * KB_KJ_MOL)
        elif self.temp is not None:
            expected_kbt = self.temp * KB_KJ_MOL
            if not math.isclose(self.kbt, expected_kbt, rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError(
                    "Conflicting kbt and temp provided: "
                    f"kbt={self.kbt} vs temp-derived {expected_kbt} (kJ/mol). "
                    "Provide only one or make them consistent."
                )
        # Normalize energy units
        object.__setattr__(
            self, "input_energy_unit", normalize_energy_unit(self.input_energy_unit)
        )
        object.__setattr__(
            self, "output_energy_unit", normalize_energy_unit(self.output_energy_unit)
        )

    @property
    def dimension(self) -> int:
        return len(self.cv_spec)

    @property
    def calculate_delta_f(self) -> bool:
        return self.delta_f_threshold is not None
