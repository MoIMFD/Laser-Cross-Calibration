"""OpticalSystem container for complete laser cross-calibration setups."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from itertools import chain

import numpy as np
import pint
import plotly.graph_objects as go


from ..types import POINT3
from ..constants import VSMALL
from ..calibration import Stage, CalibrationPoint, ureg, DEFAULT_UNIT
from .materials import Medium, AIR
from .surfaces import Surface
from .ray import OpticalRay, ray_intersection
from .optics import OpticalInterface, OpticalElement, MaterialLibrary


@dataclass
class OpticalSystem:
    """
    Complete optical system for laser cross-calibration.

    Encapsulates stage, optical surfaces, and media to provide a unified
    interface for ray tracing and intersection calculations.

    Attributes:
        stage: Motorized stage with laser beam positioning
        media: Named library of optical media
        surfaces: List of optical interfaces/surfaces
        ambient_medium_name: Default medium for ray propagation
        max_propagation_distance: Maximum distance for ray segments
        intersection_threshold: Distance threshold for ray intersections
    """

    stage: Stage
    media: dict[str, Medium] = field(default_factory=lambda: {"air": AIR})
    _surfaces: list[Surface] = field(default_factory=list)
    interfaces: list[OpticalInterface] = field(default_factory=list)
    elements: list[OpticalElement] = field(default_factory=list)
    material_library: Optional[MaterialLibrary] = None
    ambient_medium_name: str = "air"
    max_propagation_distance: float = 100.0
    intersection_threshold: float = VSMALL

    @property
    def ambient_medium(self) -> Medium:
        """Get the ambient medium object."""
        return self.media[self.ambient_medium_name]

    @property
    def surfaces(self) -> list[Surface]:
        """
        Get all surfaces from both direct surfaces and interface geometries.

        Creates a unified view of all surfaces by chaining individual surfaces
        with the geometry surfaces from all optical interfaces.

        Returns:
            Combined list of all surfaces in the system
        """
        interface_surfaces = [interface.geometry for interface in self.interfaces]
        return list(chain(self._surfaces, interface_surfaces))

    def add_medium(self, medium: Medium) -> None:
        """Add a medium to the system's media library."""
        self.media[medium.name] = medium

    def add_surface(self, surface: Surface) -> None:
        """Add an optical surface to the system."""
        self._surfaces.append(surface)

    def add_interface(self, interface: OpticalInterface) -> None:
        """Add an optical interface to the system."""
        self.interfaces.append(interface)

    def add_element(self, element: OpticalElement) -> None:
        """Add an optical element to the system."""
        self.elements.append(element)
        # Also add all interfaces from the element
        for interface in element.all_interfaces:
            self.add_interface(interface)

    def generate_rays(self) -> tuple[OpticalRay, OpticalRay]:
        """
        Generate laser rays from current stage position.

        Returns:
            Tuple of (ray1, ray2) from stage arms with beam directions
        """
        # Convert pint quantities to numpy arrays for ray creation
        # Ensure all quantities are in mm before extracting magnitude
        arm1_pos_mm = self.stage.arm1_local_position.to(DEFAULT_UNIT)
        arm2_pos_mm = self.stage.arm2_local_position.to(DEFAULT_UNIT)
        beam1_dir_mm = self.stage.beam1_normal.to(DEFAULT_UNIT)
        beam2_dir_mm = self.stage.beam2_normal.to(DEFAULT_UNIT)

        # Extract magnitudes as numpy arrays
        arm1_pos = np.asarray(arm1_pos_mm.magnitude)
        arm2_pos = np.asarray(arm2_pos_mm.magnitude)
        beam1_dir = np.asarray(beam1_dir_mm.magnitude)
        beam2_dir = np.asarray(beam2_dir_mm.magnitude)

        ray1 = OpticalRay(arm1_pos, beam1_dir)
        ray2 = OpticalRay(arm2_pos, beam2_dir)

        return ray1, ray2

    def propagate_ray_through_surfaces(self, ray: OpticalRay) -> OpticalRay:
        """
        Propagate a ray through all surfaces in the system.

        Args:
            ray: Ray to propagate

        Returns:
            The same ray object after propagation (modified in-place)
        """
        current_medium = self.ambient_medium

        # Continue until ray dies or no more intersections
        max_iterations = len(self.surfaces) * 2 + 10  # Safety limit
        iteration = 0

        while ray.is_alive and iteration < max_iterations:
            iteration += 1

            # Find closest intersection with any surface
            closest_distance = float("inf")
            closest_surface = None
            closest_intersection = None

            for surface in self.surfaces:
                intersection = surface.intersect(ray)
                if (
                    intersection is not None
                    and intersection.hit
                    and intersection.distance < closest_distance
                ):
                    closest_distance = intersection.distance
                    closest_surface = surface
                    closest_intersection = intersection

            # If no intersection found, propagate to max distance and stop
            if closest_surface is None:
                ray.propagate(self.max_propagation_distance, current_medium)
                break

            # Propagate to intersection point
            if closest_intersection is not None:
                ray.propagate(closest_intersection.distance, current_medium)

            # Apply refraction at surface
            # For now, assume all surfaces are air-glass interfaces
            # TODO: Add surface material properties
            if current_medium.name == "air":
                next_medium = self.media.get(
                    "glass", self.media.get("BK7 glass", current_medium)
                )
            else:
                next_medium = self.ambient_medium

                success = ray.refract(
                    surface_normal=closest_intersection.normal,
                    medium_from=current_medium,
                    medium_to=next_medium,
                )

                if success:
                    current_medium = next_medium
                else:
                    # Total internal reflection - ray continues in same medium
                    pass

        return ray

    def propagate_ray_sequential(self, ray: OpticalRay) -> OpticalRay:
        """
        Sequential surface-by-surface ray propagation (Optiland-inspired).

        Propagates ray through interfaces in order, using each interface's
        material properties to determine the next medium.

        Args:
            ray: Ray to propagate

        Returns:
            The same ray object after propagation (modified in-place)
        """
        current_medium = self.ambient_medium

        # Continue until ray dies or no more intersections
        max_iterations = len(self.interfaces) * 2 + 10  # Safety limit
        iteration = 0

        while ray.is_alive and iteration < max_iterations:
            iteration += 1

            # Find closest intersection with any interface
            closest_distance = float("inf")
            closest_interface = None
            closest_intersection = None

            for interface in self.interfaces:
                intersection = interface.intersect(ray)
                if (
                    intersection is not None
                    and intersection.hit
                    and intersection.distance < closest_distance
                ):
                    closest_distance = intersection.distance
                    closest_interface = interface
                    closest_intersection = intersection

            # If no intersection found, propagate to max distance and stop
            if closest_interface is None:
                ray.propagate(self.max_propagation_distance, current_medium)
                break

            # Propagate to intersection point
            if closest_intersection is not None:
                ray.propagate(closest_intersection.distance, current_medium)

                # Use interface's material properties for clean refraction logic
                next_medium = closest_interface.get_next_medium(current_medium)

                # Apply refraction at interface
                success = ray.refract(
                    surface_normal=closest_intersection.normal,
                    medium_from=current_medium,
                    medium_to=next_medium,
                )

                if success:
                    current_medium = next_medium
                else:
                    # Total internal reflection - ray continues in same medium
                    pass

        return ray

    def propagate_rays_through_system(self) -> tuple[OpticalRay, OpticalRay]:
        """
        Generate and propagate both laser rays through the complete optical system.

        Returns:
            Tuple of (ray1, ray2) after propagation through all surfaces
        """
        ray1, ray2 = self.generate_rays()

        # Choose propagation method based on available interfaces
        if self.interfaces:
            # Use enhanced sequential propagation if interfaces are defined
            self.propagate_ray_sequential(ray1)
            self.propagate_ray_sequential(ray2)
        else:
            # Fall back to legacy surface propagation
            self.propagate_ray_through_surfaces(ray1)
            self.propagate_ray_through_surfaces(ray2)

        return ray1, ray2

    def find_intersections(self) -> list[POINT3]:
        """
        Find all intersection points between the two laser beams.

        Returns:
            List of 3D intersection points
        """
        ray1, ray2 = self.propagate_rays_through_system()
        return ray_intersection(ray1, ray2, threshold=self.intersection_threshold)

    def get_calibration_point(self) -> Optional[CalibrationPoint]:
        """
        Calculate calibration point for current stage position.

        Returns:
            CalibrationPoint with intersection and stage positions, or None if no intersection
        """
        intersections = self.find_intersections()

        if not intersections:
            return None

        # For now, take the first intersection point
        # TODO: Add logic for selecting best intersection point
        intersection_pos = intersections[0]
        # Extract stage position (already in mm from grid module)
        if hasattr(self.stage.position_local, "magnitude"):
            stage_pos = np.asarray(self.stage.position_local.magnitude)
        else:
            stage_pos = np.asarray(self.stage.position_local)

        return CalibrationPoint(
            intersection_position=intersection_pos, stage_position=stage_pos
        )

    def set_stage_position(self, position: POINT3) -> None:
        """Set the stage to a new local position."""
        self.stage.set_local_position(position)

    def visualize(self, **kwargs):
        """
        Visualize the complete optical system.

        Args:
            **kwargs: Additional arguments passed to visualize_scene

        Returns:
            Plotly figure object
        """
        from .visualization import visualize_scene

        ray1, ray2 = self.propagate_rays_through_system()
        rays = [ray1, ray2]

        fig = visualize_scene(rays=rays, surfaces=self.surfaces, **kwargs)
        x, y, z = np.array(self.stage.get_world_arm_positions()).T
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z, mode="markers", marker=dict(size=3, color="red")
            )
        )
        return fig
