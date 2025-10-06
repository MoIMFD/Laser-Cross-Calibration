"""Material definitions for optical ray tracing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np


class BaseMaterial(ABC):
    """
    Base class for all optical materials.

    Following Optiland's explicit parameter approach where each material type
    defines exactly what parameters it supports.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def n(self, wavelength: float = 589.3) -> float:
        """
        Calculate refractive index.

        Args:
            wavelength: Wavelength in nm (sodium D-line default)

        Returns:
            Refractive index (dimensionless)
        """
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        return self.__str__()


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
            glycerol_fraction: Volume fraction of glycerol (0.0 = pure water, 1.0 = pure glycerol)

        Returns:
            Refractive index of the mixture
        """
        # Temperature-dependent pure component indices
        n_water = self._n_water(temperature)
        n_glycerol = self._n_glycerol(temperature)

        # Linear mixing rule (could be improved with Lorentz-Lorenz if needed)
        return n_water + self.glycerol_fraction * (n_glycerol - n_water)

    def _n_water(self, temperature: float) -> float:
        """
        Water refractive index with temperature dependence.

        Based on empirical data: dn/dT = -1.0 * 10^(-4) K^(-1)
        """
        n0 = 1.3330  # at 20C, 589nm
        dn_dt = -1.0e-4  # K^(-1)
        return n0 + dn_dt * (temperature - 20.0)

    def _n_glycerol(self, temperature: float) -> float:
        """
        Glycerol refractive index with temperature dependence.

        Based on empirical data: dn/dT = -2.5 * 10^(-4) K^(-1)
        """
        n0 = 1.4729  # at 20C, 589nm
        dn_dt = -2.5e-4  # K^(-1)
        return n0 + dn_dt * (temperature - 20.0)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(name='{self.name}', "
            f"glycerol_fraction={self.glycerol_fraction})"
        )


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
        self.C = np.array(C_coeffs)  # in um^2

    def n(
        self, wavelength: float = 589.3, temperature: Optional[float] = None
    ) -> float:
        """
        Calculate refractive index using Sellmeier formula.

        Args:
            wavelength: Wavelength in nm
            temperature: Temperature in C (for thermal correction if implemented)

        Returns:
            Refractive index at given wavelength
        """
        lambda_ = wavelength / 1000.0  # convert nm to um

        # Sellmeier formula: n^2 = 1 + sum(B_i lambda_^2/(lambda_^2 - C_i))
        n_squared = 1.0 + np.sum(self.B * lambda_**2 / (lambda_**2 - self.C))
        n_base = np.sqrt(n_squared)

        # Optional temperature correction
        if temperature is not None:
            dn_dt = self._thermal_coefficient()
            n_base += dn_dt * (temperature - 20.0)

        return n_base

    def _thermal_coefficient(self) -> float:
        """
        Thermal coefficient dn/dT in K^(-1).

        Override in subclasses for specific glass types.
        """
        return 0.0  # No thermal correction by default

    def __str__(self) -> str:
        return f"SellmeierGlass(name='{self.name}', B={self.B.tolist()}, C={self.C.tolist()})"


class MaterialRegistry:
    """
    Global registry for optical materials.

    Provides centralized access to materials by name and enables easy
    extension of the material library.
    """

    def __init__(self):
        self._materials: Dict[str, BaseMaterial] = {}
        self._aliases: Dict[str, str] = {}

    def register(
        self, material: BaseMaterial, aliases: Optional[list[str]] = None
    ) -> None:
        """
        Register a material in the global registry.

        Args:
            material: BaseMaterial object to register
            aliases: Optional list of alternative names for the material
        """
        self._materials[material.name] = material

        if aliases:
            for alias in aliases:
                self._aliases[alias] = material.name

    def get(self, name: str) -> Optional[BaseMaterial]:
        """
        Get material by name or alias.

        Args:
            name: Material name or alias

        Returns:
            BaseMaterial object if found, None otherwise
        """
        # Check direct name first
        if name in self._materials:
            return self._materials[name]

        # Check aliases
        if name in self._aliases:
            return self._materials[self._aliases[name]]

        return None

    def get_or_raise(self, name: str) -> BaseMaterial:
        """
        Get material by name or raise ValueError if not found.

        Args:
            name: Material name or alias

        Returns:
            BaseMaterial object

        Raises:
            ValueError: If material not found
        """
        material = self.get(name)
        if material is None:
            available = list(self._materials.keys()) + list(self._aliases.keys())
            raise ValueError(f"Material '{name}' not found. Available: {available}")
        return material

    def list_materials(self) -> Dict[str, BaseMaterial]:
        """Get all registered materials."""
        return self._materials.copy()

    def list_aliases(self) -> Dict[str, str]:
        """Get all registered aliases."""
        return self._aliases.copy()


# Global material registry instance
MATERIAL_REGISTRY = MaterialRegistry()

# Create material instances using new classes
AIR = ConstantMaterial("air", 1.0)
WATER = ConstantMaterial("water", 1.33)
GLYCEROL = ConstantMaterial("glycerol", 1.47)
PMMA = ConstantMaterial("PMMA", 1.49)
POLYCARBONATE = ConstantMaterial("polycarbonate", 1.59)
WATER_GLYCEROL_MIXTURE_90 = WaterGlycerolMixture(
    "water-glycerol", glycerol_fraction=0.9
)

# BK7 glass with Sellmeier coefficients (from literature)
GLASS_BK7 = SellmeierGlass(
    "BK7 glass",
    [1.03961212, 0.231792344, 1.01046945],
    [6.00069867e-3, 2.00179144e-2, 103.560653],
)

# Fused silica with Sellmeier coefficients
GLASS_FUSED_SILICA = SellmeierGlass(
    "fused silica",
    [0.696166300, 0.407942600, 0.897479400],
    [4.67914826e-3, 1.35120631e-2, 97.9340025],
)


# Register all materials in the global registry
MATERIAL_REGISTRY.register(AIR, ["Air", "vacuum"])
MATERIAL_REGISTRY.register(WATER, ["Water", "H2O"])
MATERIAL_REGISTRY.register(GLYCEROL, ["glycerin"])
MATERIAL_REGISTRY.register(PMMA, ["pmma", "acrylic", "plexiglass"])
MATERIAL_REGISTRY.register(POLYCARBONATE, ["PC", "pc"])
MATERIAL_REGISTRY.register(WATER_GLYCEROL_MIXTURE_90, ["water-glycerin-90"])
MATERIAL_REGISTRY.register(GLASS_BK7, ["BK7", "bk7", "glass", "Glass"])
MATERIAL_REGISTRY.register(GLASS_FUSED_SILICA, ["fused-silica", "quartz"])
MATERIAL_REGISTRY.register(GLASS_FUSED_SILICA, ["fused-silica", "quartz"])


def get_material(name: str) -> Optional[BaseMaterial]:
    """
    Convenience function to get material from global registry.

    Args:
        name: Material name or alias

    Returns:
        BaseMaterial object if found, None otherwise
    """
    return MATERIAL_REGISTRY.get(name)


def get_material_or_raise(name: str) -> BaseMaterial:
    """
    Convenience function to get material from global registry or raise error.

    Args:
        name: Material name or alias

    Returns:
        BaseMaterial object

    Raises:
        ValueError: If material not found
    """
    return MATERIAL_REGISTRY.get_or_raise(name)
