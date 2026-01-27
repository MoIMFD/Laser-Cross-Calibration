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

if TYPE_CHECKING:
    from hazy import Point, Vector

    from laser_cross_calibration.tracing import OpticalRay


class InfiniteCylinder(Surface):
    def __init__(
        self,
        center: Point,
        axis: Vector,
        radius: float,
        display_size: float = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.center = center
        self.axis = axis.normalize()
        self.radius = float(radius)
        self.display_size = display_size

    def intersect(self, ray: OpticalRay) -> IntersectionResult:
        local_ray = ray.localize(frame=self.center.frame)
        result = IntersectionResult()

        # Vector from cylinder center to ray origin
        oc = local_ray.current_position - self.center

        # Project ray direction and oc onto plane perpendicular to cylinder axis
        d_perp = (
            local_ray.current_direction
            - np.dot(local_ray.current_direction, self.axis) * self.axis
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

        local_ray.propagate(t, medium=None)

        result.hit = True
        result.distance = t
        result.point = local_ray.current_position

        # Normal is perpendicular to axis
        point_to_axis = result.point - self.center
        axis_component = np.dot(point_to_axis, self.axis) * self.axis
        result.normal = (point_to_axis - axis_component).normalize()
        result.surface_id = self.id

        return result

    def to_plotly_surface(
        self,
    ) -> list[go.Surface]:
        # Generate cylinder surface in local coordinates
        theta = np.linspace(0, 2 * np.pi, 30)
        s_range = np.linspace(
            -self.display_size, self.display_size, 20
        )  # Parameter along axis
        theta_grid, s_grid = np.meshgrid(theta, s_range)

        # Create orthonormal basis with cylinder.axis as one axis
        axis = self.axis

        # Find two perpendicular vectors to the axis
        if abs(axis[0]) < 0.9:
            u = axis.cross(self.axis.frame.vector([1, 0, 0]))
        else:
            u = axis.cross(self.axis.frame.vector([0, 1, 0]))
        u = np.array(u.normalize())
        v = axis.cross(u)
        v = np.array(v.normalize())

        # Generate cylinder points
        # Radial component in u-v plane + axial component
        x_local = self.radius * np.cos(theta_grid)
        y_local = self.radius * np.sin(theta_grid)

        # Transform to world coordinates
        points = (
            np.array(self.center)[np.newaxis, np.newaxis, :]
            + s_grid[:, :, np.newaxis] * np.array(axis)[np.newaxis, np.newaxis, :]
            + x_local[:, :, np.newaxis] * u[np.newaxis, np.newaxis, :]
            + y_local[:, :, np.newaxis] * v[np.newaxis, np.newaxis, :]
        )

        points = self.center.frame.batch_transform_points_global(points)

        return [
            go.Surface(
                x=points[:, :, 0],
                y=points[:, :, 1],
                z=points[:, :, 2],
                opacity=0.4,
                colorscale=get_colorscale(self.id),
                showscale=False,
                name=f"Cylinder {self.id}",
            )
        ]
