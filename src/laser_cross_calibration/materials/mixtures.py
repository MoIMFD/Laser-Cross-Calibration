"""Mixture materials with composition and temperature dependence."""

from __future__ import annotations

from laser_cross_calibration.materials.base import BaseMaterial


class WaterGlycerolMixture(BaseMaterial):
    """
    Water-glycerol mixture with composition and temperature dependence.

    Based on empirical formulas for refractive index as a function of
    glycerol volume fraction and temperature.
    """

    def __init__(self, name: str = "water-glycerol", glycerol_fraction: float = 0.0):
        super().__init__(name)
        if not (0.0 <= glycerol_fraction <= 1.0):
            raise ValueError(
                f"Glycerol fraction must be between 0 and 1, got {glycerol_fraction}"
            )
        self.glycerol_fraction = glycerol_fraction

    def n(
        self,
        wavelength: float = 589.3,
        temperature: float = 20.0,
    ) -> float:
        """
        Calculate refractive index of water-glycerol mixture.

        Args:
            wavelength: Wavelength in nm (currently unused - assumes monochromatic)
            temperature: Temperature in C

        Returns:
            Refractive index of the mixture
        """
        n_water = self._n_water(temperature)
        n_glycerol = self._n_glycerol(temperature)

        return n_water + self.glycerol_fraction * (n_glycerol - n_water)

    def _n_water(self, temperature: float) -> float:
        """
        Water refractive index with temperature dependence.

        Based on empirical data: dn/dT = -1.0 * 10^(-4) K^(-1)
        """
        n0 = 1.3330
        dn_dt = -1.0e-4
        return n0 + dn_dt * (temperature - 20.0)

    def _n_glycerol(self, temperature: float) -> float:
        """
        Glycerol refractive index with temperature dependence.

        Based on empirical data: dn/dT = -2.5 * 10^(-4) K^(-1)
        """
        n0 = 1.4729
        dn_dt = -2.5e-4
        return n0 + dn_dt * (temperature - 20.0)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(name='{self.name}', "
            f"glycerol_fraction={self.glycerol_fraction})"
        )
