"""Ray tracer for multi-source systems."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laser_cross_calibration.constants import VSMALL
from laser_cross_calibration.tracing.intersection import ray_intersection

if TYPE_CHECKING:
    from laser_cross_calibration.sources.base import LaserSource
    from laser_cross_calibration.tracing.optical_system import OpticalSystem
    from laser_cross_calibration.tracing.ray import OpticalRay
    from laser_cross_calibration.types import POINT3


class RayTracer:
    """
    Ray tracer for systems with multiple laser sources.

    Handles tracing rays from multiple sources through an optical
    system and finding intersection points between ray paths.
    """

    def __init__(self, optical_system: OpticalSystem):
        """
        Initialize ray tracer.

        Args:
            optical_system: Optical system to trace rays through
        """
        self.system = optical_system

    def trace_sources(self, sources: list[LaserSource]) -> list[OpticalRay]:
        """
        Trace all rays from all sources through the optical system.

        Args:
            sources: List of laser sources

        Returns:
            List of traced rays
        """
        all_rays = []
        for source in sources:
            for ray in source.get_rays():
                traced_ray = self.system.trace_ray(ray)
                all_rays.append(traced_ray)
        return all_rays

    def find_beam_crossings(
        self, rays: list[OpticalRay], threshold: float = VSMALL
    ) -> list[POINT3]:
        """
        Find all crossing points between ray paths.

        Args:
            rays: List of traced rays
            threshold: Distance threshold for intersection detection

        Returns:
            List of 3D intersection points
        """
        crossings = []
        for i, ray1 in enumerate(rays):
            for ray2 in rays[i + 1 :]:
                intersections = ray_intersection(ray1, ray2, threshold)
                crossings.extend(intersections)
        return crossings

    def trace_and_find_crossings(
        self, sources: list[LaserSource], threshold: float = VSMALL
    ) -> tuple[list[OpticalRay], list[POINT3]]:
        """
        Trace rays and find crossings in one call.

        Args:
            sources: List of laser sources
            threshold: Distance threshold for intersection detection

        Returns:
            Tuple of (traced_rays, crossing_points)
        """
        rays = self.trace_sources(sources)
        crossings = self.find_beam_crossings(rays, threshold)
        return rays, crossings
