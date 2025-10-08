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


class InfiniteCylinder(Surface):
    def __init__(
        self,
        center: POINT3,
        axis: VECTOR3,
        radius: float,
        display_bounds: tuple[float, float] = (-5, 5),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.center = np.array(center, dtype=np.float64)
        self.axis = np.array(axis, dtype=np.float64)
        self.axis = normalize(self.axis)
        self.radius = float(radius)
        self.display_bounds = display_bounds

    def intersect(self, ray: OpticalRay) -> IntersectionResult:
        result = IntersectionResult()

        # Vector from cylinder center to ray origin
        oc = ray.position - self.center

        # Project ray direction and oc onto plane perpendicular to cylinder axis
        d_perp = (
            ray.current_direction - np.dot(ray.current_direction, self.axis) * self.axis
        )
        oc_perp = oc - np.dot(oc, self.axis) * self.axis

        # Quadratic equation in the perpendicular plane
        a = np.dot(d_perp, d_perp)
        b = 2.0 * np.dot(oc_perp, d_perp)
        c = np.dot(oc_perp, oc_perp) - self.radius * self.radius

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return result

        # Calculate intersections
        sqrt_disc = np.sqrt(discriminant)
        # check if a is zero
        if abs(a) > VVSMALL:
            t1 = (-b - sqrt_disc) / (2 * a)
            t2 = (-b + sqrt_disc) / (2 * a)
        else:
            t1 = (-b - sqrt_disc) / (2 * VVSMALL)
            t2 = (-b + sqrt_disc) / (2 * VVSMALL)

        # Choose nearest positive intersection
        t = t1 if t1 > VSMALL else t2
        if t <= VSMALL:
            return result

        result.hit = True
        result.distance = t
        result.point = ray.position + t * ray.current_direction

        # Normal is perpendicular to axis
        point_to_axis = result.point - self.center
        axis_component = np.dot(point_to_axis, self.axis) * self.axis
        result.normal = normalize(point_to_axis - axis_component)
        result.surface_id = self.surface_id

        return result

    def to_plotly_surface(
        self,
    ) -> list[go.Surface]:
        # Generate cylinder surface in local coordinates
        theta = np.linspace(0, 2 * np.pi, 30)
        s_range = np.linspace(
            self.display_bounds[0], self.display_bounds[1], 20
        )  # Parameter along axis
        theta_grid, s_grid = np.meshgrid(theta, s_range)

        # Create orthonormal basis with cylinder.axis as one axis
        axis = self.axis

        # Find two perpendicular vectors to the axis
        if abs(axis[0]) < 0.9:
            u = np.cross(axis, [1, 0, 0])
        else:
            u = np.cross(axis, [0, 1, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(axis, u)
        v = v / np.linalg.norm(v)

        # Generate cylinder points
        # Radial component in u-v plane + axial component
        x_local = self.radius * np.cos(theta_grid)
        y_local = self.radius * np.sin(theta_grid)

        # Transform to world coordinates
        points = (
            self.center[:, np.newaxis, np.newaxis]
            + s_grid[np.newaxis, :, :] * axis[:, np.newaxis, np.newaxis]
            + x_local[np.newaxis, :, :] * u[:, np.newaxis, np.newaxis]
            + y_local[np.newaxis, :, :] * v[:, np.newaxis, np.newaxis]
        )

        return [
            go.Surface(
                x=points[0],
                y=points[1],
                z=points[2],
                opacity=0.4,
                colorscale=get_colorscale(self.surface_id),
                showscale=False,
                name=f"Cylinder {self.surface_id}",
            )
        ]
