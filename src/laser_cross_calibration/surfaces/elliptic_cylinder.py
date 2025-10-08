from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from laser_cross_calibration.constants import VSMALL, VVSMALL
from laser_cross_calibration.surfaces.base import (
    IntersectionResult,
    Surface,
    get_colorscale,
)
from laser_cross_calibration.utils import normalize

if TYPE_CHECKING:
    from laser_cross_calibration.tracing import OpticalRay
    from laser_cross_calibration.types import POINT3, VECTOR3


class EllipticCylinder(Surface):
    def __init__(
        self,
        center: POINT3,
        axis: VECTOR3,
        major_radius: float,
        minor_radius: float,
        major_axis_direction: VECTOR3,
        display_bounds: tuple[float, float] = (-5, 5),
        **kwargs,
    ):
        """
        Elliptic cylinder with elliptical cross-section.

        Args:
            center: Center point of cylinder
            axis: Direction of cylinder axis (will be normalized)
            major_radius: Radius along major axis of ellipse
            minor_radius: Radius along minor axis of ellipse
            major_axis_direction: Direction of major axis in plane perpendicular
                to the cylinder axis
        """
        super().__init__(**kwargs)
        self.center = np.array(center, dtype=np.float64)
        self.axis = np.array(axis, dtype=np.float64)
        self.axis = normalize(self.axis)
        self.major_radius = float(major_radius)
        self.minor_radius = float(minor_radius)
        self.display_bounds = display_bounds

        # Create orthonormal basis for ellipse
        major_dir = np.array(major_axis_direction, dtype=np.float64)
        # Project major direction onto plane perpendicular to axis
        major_dir = major_dir - np.dot(major_dir, self.axis) * self.axis
        self.major_axis = normalize(major_dir)
        self.minor_axis = normalize(np.cross(self.axis, self.major_axis))

    def intersect(self, ray: OpticalRay) -> IntersectionResult:
        result = IntersectionResult()

        # Transform to ellipse coordinate system
        oc = ray.position - self.center

        # Project ray direction and oc onto ellipse plane
        d_u = np.dot(ray.current_direction, self.major_axis)
        d_v = np.dot(ray.current_direction, self.minor_axis)
        oc_u = np.dot(oc, self.major_axis)
        oc_v = np.dot(oc, self.minor_axis)

        # Ellipse equation: (u/a)² + (v/b)² = 1
        # Ray: u = oc_u + t*d_u, v = oc_v + t*d_v
        # Substitute: ((oc_u + t*d_u)/a)² + ((oc_v + t*d_v)/b)² = 1

        a = self.major_radius
        b = self.minor_radius

        # Quadratic coefficients
        A = (d_u / a) ** 2 + (d_v / b) ** 2
        B = 2 * ((oc_u * d_u) / (a**2) + (oc_v * d_v) / (b**2))
        C = (oc_u / a) ** 2 + (oc_v / b) ** 2 - 1

        discriminant = B**2 - 4 * A * C

        if discriminant < 0:
            return result

        # Calculate intersections
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-B - sqrt_disc) / (2 * max(A, VVSMALL))
        t2 = (-B + sqrt_disc) / (2 * max(A, VVSMALL))

        # Choose nearest positive intersection
        t = t1 if t1 > VSMALL else t2
        if t <= VSMALL:
            return result

        result.hit = True
        result.distance = t
        result.point = ray.position + t * ray.current_direction

        # Calculate normal at intersection point
        point_rel = result.point - self.center
        u_comp = np.dot(point_rel, self.major_axis)
        v_comp = np.dot(point_rel, self.minor_axis)

        # Ellipse normal: ∇((u/a)² + (v/b)²) = (2u/a², 2v/b²)
        normal_local = (2 * u_comp / (a**2)) * self.major_axis + (
            2 * v_comp / (b**2)
        ) * self.minor_axis
        result.normal = normalize(normal_local)
        result.surface_id = self.surface_id

        return result

    def to_plotly_surface(self) -> list[go.Surface]:
        # Generate elliptic cylinder surface in local coordinates
        theta = np.linspace(0, 2 * np.pi, 30)
        s_range = np.linspace(
            self.display_bounds[0], self.display_bounds[1], 20
        )  # Parameter along axis
        theta_grid, s_grid = np.meshgrid(theta, s_range)

        # Ellipse parameters
        a = self.major_radius
        b = self.minor_radius

        # Generate elliptic cylinder points
        # Ellipse in major-minor axis plane: x = a*cos(θ), y = b*sin(θ)
        x_local = a * np.cos(theta_grid)
        y_local = b * np.sin(theta_grid)

        # Transform to world coordinates using the orthonormal basis
        points = (
            self.center[:, np.newaxis, np.newaxis]
            + s_grid[np.newaxis, :, :] * self.axis[:, np.newaxis, np.newaxis]
            + x_local[np.newaxis, :, :] * self.major_axis[:, np.newaxis, np.newaxis]
            + y_local[np.newaxis, :, :] * self.minor_axis[:, np.newaxis, np.newaxis]
        )

        return [
            go.Surface(
                x=points[0],
                y=points[1],
                z=points[2],
                opacity=0.4,
                colorscale=get_colorscale(self.surface_id),
                showscale=False,
                name=f"Elliptic Cylinder {self.surface_id}",
            )
        ]
