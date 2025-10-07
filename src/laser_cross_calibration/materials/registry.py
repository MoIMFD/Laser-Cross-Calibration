"""Global registry for optical materials."""

from __future__ import annotations

from typing import Dict, Optional

from laser_cross_calibration.materials.base import BaseMaterial


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
        if name in self._materials:
            return self._materials[name]

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
