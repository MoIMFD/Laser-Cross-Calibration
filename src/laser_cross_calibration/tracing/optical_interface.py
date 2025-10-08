"""Optical interface coupling surfaces with materials."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from laser_cross_calibration.materials.base import BaseMaterial
    from laser_cross_calibration.surfaces.base import Surface
    from laser_cross_calibration.tracing.ray import OpticalRay


@dataclass
class OpticalInterface:
    """
    Enhanced surface with explicit material properties.

    Defines an optical interface with materials on both sides,
    following the Optiland pattern of surface-centric material definition.

    Attributes:
        geometry: The surface geometry for ray intersection
        material_pre: BaseMaterial before the surface (incident side)
        material_post: BaseMaterial after the surface (transmitted side)
        surface_id: Optional identifier for the interface
        is_bounded: Whether this interface has finite dimensions
    """

    geometry: Surface
    material_pre: BaseMaterial
    material_post: BaseMaterial
    surface_id: int | None = None
    is_bounded: bool = field(default=True)

    def intersect(self, ray: OpticalRay):
        """
        Delegate intersection to the geometry.

        For bounded interfaces, this automatically handles the boundary checking
        through the surface geometry (e.g., RectangularPlane vs Plane).
        """
        return self.geometry.intersect(ray)

    def get_next_medium(self, current_medium: BaseMaterial) -> BaseMaterial:
        """
        Determine the next medium based on current medium.

        Args:
            current_medium: The medium the ray is currently in

        Returns:
            The medium the ray should enter after this interface
        """
        if current_medium.name == self.material_pre.name:
            return self.material_post
        elif current_medium.name == self.material_post.name:
            return self.material_pre
        else:
            return self.material_post
