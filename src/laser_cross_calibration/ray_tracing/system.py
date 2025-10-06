"""OpticalSystem container for complete laser cross-calibration setups."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import plotly.graph_objects as go

from laser_cross_calibration.calibration import DEFAULT_UNIT, CalibrationPoint, Stage
from laser_cross_calibration.constants import VSMALL
from laser_cross_calibration.types import POINT3
from laser_cross_calibration.utils import normalize
from laser_cross_calibration.ray_tracing.optics import MaterialLibrary, OpticalElement, OpticalInterface
from laser_cross_calibration.ray_tracing.ray import OpticalRay, ray_intersection
from laser_cross_calibration.ray_tracing.surfaces import Surface
from laser_cross_calibration import backend as be


@dataclass
class OpticalSystem:
    """
    Complete optical system for laser cross-calibration.

    Encapsulates stage, optical surfaces, and media to provide a unified
    interface for ray tracing and intersection calculations.

    Attributes:
        stage: Motorized stage with laser beam positioning
        interfaces: List of optical interfaces with material definitions
        elements: List of optical elements
        material_library: Optional library of materials
        max_propagation_distance: Maximum distance for ray segments
        intersection_threshold: Distance threshold for ray intersections
    """

    stage: Stage
    _surfaces: list[Surface] = field(default_factory=list)
    interfaces: list[OpticalInterface] = field(default_factory=list)
    elements: list[OpticalElement] = field(default_factory=list)
    material_library: Optional[MaterialLibrary] = None
    max_propagation_distance: float = 100.0
    intersection_threshold: float = VSMALL

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
        arm1_pos = be.array(arm1_pos_mm.magnitude)
        arm2_pos = be.array(arm2_pos_mm.magnitude)
        beam1_dir = be.array(beam1_dir_mm.magnitude)
        beam2_dir = be.array(beam2_dir_mm.magnitude)

        ray1 = OpticalRay(arm1_pos, beam1_dir)
        ray2 = OpticalRay(arm2_pos, beam2_dir)

        return ray1, ray2

    def propagate_ray_sequential(self, ray: OpticalRay) -> OpticalRay:
        """
        Sequential interface-by-interface ray propagation.

        Ray travels in straight lines through homogeneous media until hitting
        interfaces, which handle material transitions via their pre/post materials.

        Args:
            ray: Ray to propagate

        Returns:
            The same ray object after propagation (modified in-place)
        """
        current_medium = None  # Will be determined at first interface

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
                # For the final propagation, we need a medium for the ray path
                # Use the last known medium, or default fallback
                if current_medium is None:
                    # If we never hit an interface, assume starting in air
                    from laser_cross_calibration.ray_tracing.materials import get_material_or_raise

                    current_medium = get_material_or_raise("air")
                ray.propagate(self.max_propagation_distance, current_medium)
                break

            # Propagate to intersection point
            if closest_intersection is not None:
                # Determine current medium if this is the first interface
                if current_medium is None:
                    # At first interface, determine which side we're coming from
                    # by checking ray direction vs surface normal
                    ray_dir = normalize(ray.direction)
                    dot_product = be.dot(ray_dir, closest_intersection.normal)

                    if dot_product < 0:
                        # Ray approaching from material_pre side
                        current_medium = closest_interface.material_pre
                    else:
                        # Ray approaching from material_post side
                        current_medium = closest_interface.material_post

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

        # Propagate both rays through the interface-based system
        self.propagate_ray_sequential(ray1)
        self.propagate_ray_sequential(ray2)

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
            stage_pos = be.array(self.stage.position_local.magnitude)
        else:
            stage_pos = be.array(self.stage.position_local)

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
        from laser_cross_calibration.ray_tracing.visualization import visualize_scene

        ray1, ray2 = self.propagate_rays_through_system()
        rays = [ray1, ray2]

        fig = visualize_scene(rays=rays, surfaces=self.surfaces, **kwargs)
        x, y, z = be.array(self.stage.get_world_arm_positions()).T
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(size=3, color="red"),
                name="beam origin",
            )
        )
        return fig
