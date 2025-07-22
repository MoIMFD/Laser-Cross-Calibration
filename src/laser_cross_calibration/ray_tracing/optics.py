"""Enhanced optical components inspired by Optiland architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np

from ..types import POINT3, VECTOR3
from .materials import Medium, AIR, GLASS_BK7, WATER
from .surfaces import Surface
from .ray import OpticalRay


@dataclass
class OpticalInterface:
    """
    Enhanced surface with explicit material properties.

    Defines an optical interface with materials on both sides,
    following the Optiland pattern of surface-centric material definition.

    Attributes:
        geometry: The surface geometry for ray intersection
        material_pre: Medium before the surface (incident side)
        material_post: Medium after the surface (transmitted side)
        surface_id: Optional identifier for the interface
        is_bounded: Whether this interface has finite dimensions
    """

    geometry: Surface
    material_pre: Medium
    material_post: Medium
    surface_id: Optional[str] = None
    is_bounded: bool = field(default=True)  # Assume bounded unless specified

    def intersect(self, ray: OpticalRay):
        """
        Delegate intersection to the geometry.

        For bounded interfaces, this automatically handles the boundary checking
        through the surface geometry (e.g., RectangularPlane vs Plane).
        """
        intersection = self.geometry.intersect(ray)

        # If intersection exists but interface is bounded,
        # the geometry itself handles the boundary check
        return intersection

    def get_next_medium(self, current_medium: Medium) -> Medium:
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
            # Default behavior: assume ray enters from pre to post
            return self.material_post

    def is_ray_within_bounds(self, ray_position: POINT3) -> bool:
        """
        Check if a ray position is within the physical bounds of this interface.

        This is primarily used for bounded surfaces like rectangular plates.
        For infinite surfaces, this always returns True.

        Args:
            ray_position: 3D position to check

        Returns:
            True if position is within bounds, False otherwise
        """
        if not self.is_bounded:
            return True

        # For bounded surfaces, delegate to the geometry
        from .surfaces import RectangularPlane

        if isinstance(self.geometry, RectangularPlane):
            # Check if point is within rectangular bounds
            return self.geometry._is_point_within_bounds(ray_position)
        else:
            # Other bounded surfaces would implement their own bounds checking
            return True


@dataclass
class OpticalElement:
    """
    Container for optical elements with front/back surfaces.

    Represents a physical optical element like a glass plate, tank wall,
    or optical window with defined thickness and bulk material.

    Attributes:
        name: Human-readable identifier
        front_interface: Entry interface
        back_interface: Exit interface
        bulk_material: Material between front and back surfaces
        thickness: Physical thickness of the element
    """

    name: str
    front_interface: OpticalInterface
    back_interface: OpticalInterface
    bulk_material: Medium
    thickness: float

    @property
    def all_interfaces(self) -> List[OpticalInterface]:
        """Get all interfaces for this element."""
        return [self.front_interface, self.back_interface]

    @classmethod
    def create_glass_plate(
        cls,
        name: str,
        front_surface: Surface,
        back_surface: Surface,
        thickness: float,
        glass_material: Medium = GLASS_BK7,
        ambient_material: Medium = AIR,
    ) -> "OpticalElement":
        """
        Factory method for creating a glass plate element.
        
        Args:
            name: Element identifier
            front_surface: Entry surface geometry
            back_surface: Exit surface geometry  
            thickness: Plate thickness
            glass_material: Glass material
            ambient_material: Surrounding medium
            
        Returns:
            OpticalElement representing the glass plate
        """
        front_interface = OpticalInterface(
            geometry=front_surface,
            material_pre=ambient_material,
            material_post=glass_material,
            surface_id=f"{name}_front",
        )

        back_interface = OpticalInterface(
            geometry=back_surface,
            material_pre=glass_material,
            material_post=ambient_material,
            surface_id=f"{name}_back",
        )

        return cls(
            name=name,
            front_interface=front_interface,
            back_interface=back_interface,
            bulk_material=glass_material,
            thickness=thickness,
        )

    @classmethod
    def create_bounded_glass_plate(
        cls,
        name: str,
        center_position: POINT3,
        normal_direction: VECTOR3,
        width: float,
        height: float,
        thickness: float,
        glass_material: Medium = GLASS_BK7,
        ambient_material: Medium = AIR,
    ) -> "OpticalElement":
        """
        Factory method for creating a bounded glass plate with defined dimensions.

        Args:
            name: Element identifier
            center_position: 3D position of plate center
            normal_direction: Direction normal to plate surface
            width: Plate width (perpendicular to normal)
            height: Plate height (perpendicular to normal)
            thickness: Plate thickness (along normal direction)
            glass_material: Glass material
            ambient_material: Surrounding medium

        Returns:
            OpticalElement representing the bounded glass plate
        """
        from .surfaces import RectangularPlane

        center = np.array(center_position)
        normal = np.array(normal_direction)
        normal = normal / np.linalg.norm(normal)

        # Calculate front and back surface positions
        front_position = center - (thickness / 2) * normal
        back_position = center + (thickness / 2) * normal

        # Create bounded rectangular surfaces
        front_surface = RectangularPlane(
            center=front_position, normal=normal, width=width, height=height
        )

        back_surface = RectangularPlane(
            center=back_position, normal=normal, width=width, height=height
        )

        # Create interfaces with bounds
        front_interface = OpticalInterface(
            geometry=front_surface,
            material_pre=ambient_material,
            material_post=glass_material,
            surface_id=f"{name}_front",
            is_bounded=True,
        )

        back_interface = OpticalInterface(
            geometry=back_surface,
            material_pre=glass_material,
            material_post=ambient_material,
            surface_id=f"{name}_back",
            is_bounded=True,
        )

        return cls(
            name=name,
            front_interface=front_interface,
            back_interface=back_interface,
            bulk_material=glass_material,
            thickness=thickness,
        )
        """
        Factory method for creating a glass plate element.
        
        Args:
            name: Element identifier
            front_surface: Entry surface geometry
            back_surface: Exit surface geometry  
            thickness: Plate thickness
            glass_material: Glass material
            ambient_material: Surrounding medium
            
        Returns:
            OpticalElement representing the glass plate
        """
        front_interface = OpticalInterface(
            geometry=front_surface,
            material_pre=ambient_material,
            material_post=glass_material,
            surface_id=f"{name}_front",
        )

        back_interface = OpticalInterface(
            geometry=back_surface,
            material_pre=glass_material,
            material_post=ambient_material,
            surface_id=f"{name}_back",
        )

        return cls(
            name=name,
            front_interface=front_interface,
            back_interface=back_interface,
            bulk_material=glass_material,
            thickness=thickness,
        )


@dataclass
class MaterialLibrary:
    """
    Central material database for optical systems.

    Provides centralized material management similar to Optiland's
    material system, with support for common optical materials.
    """

    materials: Dict[str, Medium] = field(default_factory=dict)

    def add_material(self, medium: Medium) -> None:
        """Add a material to the library."""
        self.materials[medium.name] = medium

    def get_material(self, name: str) -> Medium:
        """Retrieve a material by name."""
        if name not in self.materials:
            raise ValueError(f"Material '{name}' not found in library")
        return self.materials[name]

    def has_material(self, name: str) -> bool:
        """Check if material exists in library."""
        return name in self.materials

    @classmethod
    def create_standard_library(cls) -> "MaterialLibrary":
        """Create a library with common optical materials."""
        lib = cls()

        # Add standard materials
        lib.add_material(AIR)
        lib.add_material(WATER)
        lib.add_material(GLASS_BK7)

        # Add additional materials from materials module
        from .materials import GLASS_FUSED_SILICA, PMMA, POLYCARBONATE

        lib.add_material(GLASS_FUSED_SILICA)
        lib.add_material(PMMA)
        lib.add_material(POLYCARBONATE)

        return lib


@dataclass
class WaterTank:
    """
    Complex optical assembly representing a water tank.

    Generates multiple interfaces for a rectangular tank with glass walls
    and water interior, following realistic optical geometries.
    """

    name: str
    inner_dimensions: VECTOR3  # [width, height, depth] of water region
    wall_thickness: float
    wall_material: Medium = field(default_factory=lambda: GLASS_BK7)
    interior_material: Medium = field(default_factory=lambda: WATER)
    ambient_material: Medium = field(default_factory=lambda: AIR)

    def generate_interfaces(self) -> List[OpticalInterface]:
        """
        Generate all optical interfaces for the tank.

        Creates interfaces for:
        - Front wall: ambient -> glass -> water
        - Back wall: water -> glass -> ambient
        - Side walls (if rays might intersect them)

        Returns:
            List of all optical interfaces in the tank
        """
        interfaces = []

        # For now, implement front and back walls
        # TODO: Add full 3D tank geometry

        from .surfaces import Plane

        # Front wall outer surface (ambient -> glass)
        front_outer = Plane(point=[0, 0, -self.wall_thickness / 2], normal=[0, 0, 1])
        interfaces.append(
            OpticalInterface(
                geometry=front_outer,
                material_pre=self.ambient_material,
                material_post=self.wall_material,
                surface_id=f"{self.name}_front_outer",
            )
        )

        # Front wall inner surface (glass -> water)
        front_inner = Plane(point=[0, 0, self.wall_thickness / 2], normal=[0, 0, 1])
        interfaces.append(
            OpticalInterface(
                geometry=front_inner,
                material_pre=self.wall_material,
                material_post=self.interior_material,
                surface_id=f"{self.name}_front_inner",
            )
        )

        # Back wall inner surface (water -> glass)
        back_inner = Plane(
            point=[0, 0, self.inner_dimensions[2] + self.wall_thickness / 2],
            normal=[0, 0, -1],
        )
        interfaces.append(
            OpticalInterface(
                geometry=back_inner,
                material_pre=self.interior_material,
                material_post=self.wall_material,
                surface_id=f"{self.name}_back_inner",
            )
        )

        # Back wall outer surface (glass -> ambient)
        back_outer = Plane(
            point=[
                0,
                0,
                self.inner_dimensions[2]
                + self.wall_thickness
                + self.wall_thickness / 2,
            ],
            normal=[0, 0, -1],
        )
        interfaces.append(
            OpticalInterface(
                geometry=back_outer,
                material_pre=self.wall_material,
                material_post=self.ambient_material,
                surface_id=f"{self.name}_back_outer",
            )
        )

        return interfaces
