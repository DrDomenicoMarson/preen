"""Constants for FES reweighting scripts."""

KB_KJ_MOL = 0.0083144621
ERROR_PREFIX = "--- ERROR:"

# Energy unit handling
_ENERGY_UNITS = {
    "kj/mol": 1.0,
    "kcal/mol": 4.184,  # 1 kcal/mol = 4.184 kJ/mol
}


def normalize_energy_unit(unit: str) -> str:
    """Normalize energy unit strings."""
    normalized = unit.strip().lower()
    if normalized in ("kj", "kjmol"):
        normalized = "kj/mol"
    if normalized in ("kcal", "kcalmol"):
        normalized = "kcal/mol"
    if normalized not in _ENERGY_UNITS:
        raise ValueError(f'Unsupported energy unit "{unit}". Use kJ/mol or kcal/mol.')
    return normalized


def energy_conversion_factor(from_unit: str, to_unit: str) -> float:
    """
    Return multiplicative factor to convert values from one energy unit to another.

    Example: convert value in kcal/mol to kJ/mol -> multiply by factor returned by
    energy_conversion_factor("kcal/mol", "kJ/mol")
    """
    src = normalize_energy_unit(from_unit)
    dst = normalize_energy_unit(to_unit)
    return _ENERGY_UNITS[src] / _ENERGY_UNITS[dst]
