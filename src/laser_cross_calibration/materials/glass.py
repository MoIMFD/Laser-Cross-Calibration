"""Glass materials with Sellmeier dispersion formula."""

from __future__ import annotations

import numpy as np

from laser_cross_calibration.materials.base import BaseMaterial


class SellmeierGlass(BaseMaterial):
    """
    Glass with wavelength-dependent Sellmeier dispersion formula.

    Implements the standard Sellmeier equation:
    n^2(lambda) = 1 + sum_i B_i lambda_^2/(lambda_^2 - C_i)
    """

    def __init__(self, name: str, B_coeffs: list[float], C_coeffs: list[float]):
        super().__init__(name)
        if len(B_coeffs) != len(C_coeffs):
            raise ValueError("B and C coefficient lists must have same length")
        self.B = np.array(B_coeffs)
        self.C = np.array(C_coeffs)

    def n(self, wavelength: float = 589.3, temperature: float | None = None) -> float:
        """
        Calculate refractive index using Sellmeier formula.

        Args:
            wavelength: Wavelength in nm
            temperature: Temperature in C (for thermal correction if implemented)

        Returns:
            Refractive index at given wavelength
        """
        lambda_ = wavelength / 1000.0

        n_squared = 1.0 + np.sum(self.B * lambda_**2 / (lambda_**2 - self.C))
        n_base = np.sqrt(n_squared)

        if temperature is not None:
            dn_dt = self._thermal_coefficient()
            n_base += dn_dt * (temperature - 20.0)

        return float(n_base)

    def _thermal_coefficient(self) -> float:
        """
        Thermal coefficient dn/dT in K^(-1).

        Override in subclasses for specific glass types.
        """
        return 0.0

    def __str__(self) -> str:
        return (
            "SellmeierGlass("
            f"name='{self.name}', "
            f"B={self.B.tolist()}, "
            f"C={self.C.tolist()})"
        )
