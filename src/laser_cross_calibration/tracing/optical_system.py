"""Decoupled optical system for non-sequential ray tracing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from laser_cross_calibration.materials import AIR
from laser_cross_calibration.utils import normalize

if TYPE_CHECKING:
    from laser_cross_calibration.materials.base import BaseMaterial
    from laser_cross_calibration.surfaces import IntersectionResult
    from laser_cross_calibration.tracing.optical_interface import OpticalInterface
    from laser_cross_calibration.tracing.ray import OpticalRay


class OpticalSystem:
    """
    Optical system for non-sequential ray tracing.

    Manages optical interfaces and provides ray propagation through
    multiple surfaces. Decoupled from any specific source (no Stage dependency).

    Attributes:
        interfaces: List of optical interfaces
        max_propagation_distance: Maximum ray travel distance
    """

    def __init__(self, max_propagation_distance: float = 1000.0):
        """
        Initialize optical system.

        Args:
            max_propagation_distance: Maximum distance for ray segments
        """
        self.interfaces: list[OpticalInterface] = []
        self.max_propagation_distance = max_propagation_distance

    def add_interface(self, interface: OpticalInterface) -> None:
        """Add an optical interface to the system."""
        self.interfaces.append(interface)

    def find_next_intersection(
        self, ray: OpticalRay
    ) -> tuple[OpticalInterface | None, IntersectionResult | None]:
        """
        Find closest intersection with any interface.

        This is the core non-sequential raytracing logic.

        Args:
            ray: Ray to test against all interfaces

        Returns:
            Tuple of (closest_interface, intersection_result) or (None, None)
        """
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

        return closest_interface, closest_intersection

    def trace_ray(
        self,
        ray: OpticalRay,
        current_medium: BaseMaterial = AIR,
        max_bounces: int = 100,
    ) -> OpticalRay:
        """
        Trace single ray through the optical system.

        Uses non-sequential raytracing - finds closest intersection
        at each step rather than following a fixed surface order.

        Args:
            ray: Ray to trace
            current_medium: Starting medium (default: air)
            max_bounces: Maximum number of surface interactions

        Returns:
            The same ray object after propagation (modified in-place)
        """
        iteration = 0

        while ray.is_alive and iteration < max_bounces:
            iteration += 1

            interface, intersection = self.find_next_intersection(ray)

            if interface is None:
                ray.propagate(self.max_propagation_distance, current_medium)
                break

            if intersection is not None:
                if current_medium is None:
                    ray_dir = normalize(ray.direction)
                    dot_product = np.dot(ray_dir, intersection.normal)

                    if dot_product < 0:
                        current_medium = interface.material_pre
                    else:
                        current_medium = interface.material_post

                ray.propagate(intersection.distance, current_medium)

                next_medium = interface.get_next_medium(current_medium)

                success = ray.refract(
                    surface_normal=intersection.normal,
                    medium_from=current_medium,
                    medium_to=next_medium,
                )

                if success:
                    current_medium = next_medium

        return ray
