from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from laser_cross_calibration.constants import (
    ORIGIN_POINT3,
    UNIT_X_VECTOR3,
    UNIT_Y_VECTOR3,
    UNIT_Z_VECTOR3,
    VSMALL,
)
from laser_cross_calibration.coordinate_system import check_same_frame
from laser_cross_calibration.materials import AIR
from laser_cross_calibration.surfaces.base import (
    IntersectionResult,
    Surface,
    get_colorscale,
)

if TYPE_CHECKING:
    from laser_cross_calibration.coordinate_system import Point, Vector
    from laser_cross_calibration.tracing import OpticalRay


class Plane(Surface):
    """
    Infinite plane surface defined by a point and normal vector.

    Attributes:
        point: Reference point on the plane
        normal: Normalized normal vector to the plane
    """

    def __init__(
        self,
        point: Point,
        normal: Vector,
        display_size: float = 2.0,
        **kwargs,
    ) -> None:
        """
        Initialize infinite plane.

        Args:
            point: Any point lying on the plane
            normal: Normal vector to the plane (will be normalized)
            **kwargs: Additional arguments passed to Surface constructor

        Raises:
            ValueError: If normal vector is zero
        """
        check_same_frame(point, normal)
        super().__init__(**kwargs)

        self.display_size = display_size
        self.point = point
        self.normal = normal.normalize()

    def intersect(self, ray: OpticalRay) -> IntersectionResult:
        """Calculate the intersection between a ray and this plane.

        The intersection is calculated in the coordinate system of the surface.
        Therefor the ray gets localized, the intersection is calculated and the
        result transformed back to the world coordinate space.
        """
        local_ray = ray.localize(frame=self.point.frame)
        result = IntersectionResult()

        denom = np.dot(local_ray.current_direction, self.normal)

        if abs(denom) < VSMALL:
            return result

        t = np.dot(self.point - local_ray.current_position, self.normal) / denom

        if t < VSMALL:
            return result

        # since the ray will be discared the medium is just a dummy
        local_ray.propagate(t, medium=AIR)

        result.hit = True
        result.distance = t
        result.point = local_ray.current_position
        result.normal = self.normal if denom < 0.0 else self.normal * (-1)
        result.surface_id = self.id

        return result

    def to_plotly_surface(
        self, show_normals: bool = False
    ) -> list[go.Surface] | list[go.Surface | go.Scatter3d]:
        # Create grid points on the plane
        u_range = np.linspace(-self.display_size, self.display_size, 2)
        v_range = np.linspace(-self.display_size, self.display_size, 2)
        u_grid, v_grid = np.meshgrid(u_range, v_range)

        # Create two orthogonal vectors in the plane
        if abs(self.normal[0]) < 0.9:
            u_vec = np.cross(self.normal, [1, 0, 0])
        else:
            u_vec = np.cross(self.normal, [0, 1, 0])
        u_vec = u_vec / np.linalg.norm(u_vec)
        v_vec = np.cross(self.normal, u_vec)

        # Generate plane points
        points = (
            self.point[np.newaxis, np.newaxis, :]
            + u_vec[np.newaxis, np.newaxis, :] * u_grid[:, :, np.newaxis]
            + v_vec[np.newaxis, np.newaxis, :] * v_grid[:, :, np.newaxis]
        )

        points = self.point.frame.batch_transform_global(points)

        surface = go.Surface(
            x=points[:, :, 0],
            y=points[:, :, 1],
            z=points[:, :, 2],
            opacity=0.3,
            colorscale=get_colorscale(self.id),
            showscale=False,
            name=f"Plane {self.id}",
        )

        if not show_normals:
            return [surface]

        # Add normal vector
        normal_end = self.point + self.normal * 0.5
        normal_trace = go.Scatter3d(
            x=[self.point[0], normal_end[0]],
            y=[self.point[1], normal_end[1]],
            z=[self.point[2], normal_end[2]],
            mode="lines",
            line={"width": 4, "color": "green"},
            name=f"Normal {self.id}",
            showlegend=False,
        )

        return [surface, normal_trace]

    @classmethod
    def create_xy(
        cls, display_size: float = 2.0, surface_id: int | None = None, **kwargs
    ) -> Plane:
        return cls(
            point=ORIGIN_POINT3,
            normal=UNIT_Z_VECTOR3,
            display_size=display_size,
            **kwargs,
        )

    @classmethod
    def create_xz(
        cls, display_size: float = 2.0, surface_id: int | None = None, **kwargs
    ) -> Plane:
        return cls(
            point=ORIGIN_POINT3,
            normal=UNIT_Y_VECTOR3,
            display_size=display_size,
            **kwargs,
        )

    @classmethod
    def create_yz(
        cls, display_size: float = 2.0, surface_id: int | None = None, **kwargs
    ) -> Plane:
        return cls(
            point=ORIGIN_POINT3,
            normal=UNIT_X_VECTOR3,
            display_size=display_size,
            **kwargs,
        )
