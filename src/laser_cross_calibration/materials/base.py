"""Base class for optical materials."""

from __future__ import annotations

from abc import ABC, abstractmethod


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
