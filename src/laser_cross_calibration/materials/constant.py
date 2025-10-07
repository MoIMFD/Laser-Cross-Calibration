"""Simple materials with constant refractive index."""

from __future__ import annotations

from laser_cross_calibration.materials.base import BaseMaterial


class ConstantMaterial(BaseMaterial):
    """
    Simple material with constant refractive index.

    Ignores all environmental parameters - suitable for basic materials
    where wavelength/temperature dependence is negligible.
    """

    def __init__(self, name: str, ior: float):
        super().__init__(name)
        if ior <= 0:
            raise ValueError(f"Index of refraction must be positive, got {ior}")
        self.ior = ior

    def n(self, wavelength: float = 589.3) -> float:
        """Constant refractive index - wavelength is ignored."""
        return self.ior

    def __str__(self) -> str:
        return f"ConstantMaterial(name='{self.name}', ior={self.ior})"
