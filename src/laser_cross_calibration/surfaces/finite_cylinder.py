from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from laser_cross_calibration.surfaces.infinite_cylinder import InfiniteCylinder

if TYPE_CHECKING:
    import plotly.graph_objects as go
    from hazy import Point, Vector

    from laser_cross_calibration.surfaces.base import (
        IntersectionResult,
    )
    from laser_cross_calibration.tracing import OpticalRay


class FiniteCylinder(InfiniteCylinder):
    def __init__(
        self, center: Point, axis: Vector, radius: float, length: float, **kwargs
    ):
        """
        Finite cylinder with specified length.

        Args:
            center: Center point of cylinder
            axis: Direction of cylinder axis (will be normalized)
            radius: Cylinder radius
            length: Length of cylinder along axis
        """
        super().__init__(center, axis, radius, **kwargs)
        self.length = float(length)
        # set cylinder length as display bounds
        self.display_size = self.length / 2

    def intersect(self, ray: OpticalRay) -> IntersectionResult:
        # Get intersection from infinite cylinder
        result = super().intersect(ray)

        if not result.hit:
            return result

        # Check if intersection point is within cylinder length
        point_on_axis = result.point - self.center
        axis_distance = np.dot(point_on_axis, self.axis)

        # Check if within bounds [-length/2, +length/2]
        if abs(axis_distance) > self.length / 2:
            result.hit = False
            result.distance = np.inf
            result.point = self.center.frame.point(np.nan, np.nan, np.nan)
            result.normal = self.center.frame.vector(np.nan, np.nan, np.nan)
            result.surface_id = -1

        return result

    def to_plotly_surface(self, show_normals: bool = False) -> list[go.Surface]:
        # Use parent class implementation with proper bounds
        return super().to_plotly_surface()
