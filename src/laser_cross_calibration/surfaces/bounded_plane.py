from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from laser_cross_calibration.constants import VSMALL
from laser_cross_calibration.surfaces.base import (
    IntersectionResult,
    Surface,
    get_surface_color,
)
from laser_cross_calibration.utils import normalize

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from laser_cross_calibration.tracing import OpticalRay
    from laser_cross_calibration.types import POINT3, VECTOR3


class RectangularPlane(Surface):
    def __init__(
        self,
        center: POINT3,
        normal: VECTOR3,
        width: float,
        height: float,
        width_direction: VECTOR3 | None = None,
        **kwargs,
    ):
        """
        Rectangular plane defined by center, normal, and dimensions.

        Args:
            center: Center point of rectangle
            normal: Normal vector to the plane
            width: Width of rectangle
            height: Height of rectangle
            width_direction: Direction of width axis (will be projected to plane).
                If None, auto-generated.
        """
        super().__init__(**kwargs)
        self.center = np.array(center, dtype=np.float64)
        self.normal = np.array(normal, dtype=np.float64)
        self.normal = normalize(self.normal)
        self.width = float(width)
        self.height = float(height)

        # Create orthonormal basis for rectangle
        if width_direction is not None:
            width_dir = np.array(width_direction, dtype=np.float64)
            # Project to plane perpendicular to normal
            width_dir = width_dir - np.dot(width_dir, self.normal) * self.normal
            self.width_axis = normalize(width_dir)
        else:
            # Auto-generate width direction
            if abs(self.normal[0]) < 0.9:
                width_dir = np.cross(self.normal, [1, 0, 0])
            else:
                width_dir = np.cross(self.normal, [0, 1, 0])
            self.width_axis = normalize(width_dir)

        self.height_axis = normalize(np.cross(self.normal, self.width_axis))

    @classmethod
    def from_corners(
        cls,
        corner1: POINT3,
        corner2: POINT3,
        corner3: POINT3,
        corner4: POINT3,
        **kwargs,
    ):
        """
        Create rectangular plane from 4 corner points.
        Points should be in order (e.g., clockwise or counter-clockwise).
        """
        c1, c2, c3, c4 = [
            np.array(p, dtype=np.float64) for p in [corner1, corner2, corner3, corner4]
        ]

        # Calculate center
        center = (c1 + c2 + c3 + c4) / 4

        # Calculate normal from two edges
        edge1 = c2 - c1
        edge2 = c4 - c1
        normal = np.cross(edge1, edge2)
        normal = normalize(normal)

        # Calculate dimensions
        width = np.linalg.norm(c2 - c1)
        height = np.linalg.norm(c4 - c1)

        # Width direction
        width_direction = normalize(c2 - c1)

        return cls(
            center, normal, float(width), float(height), width_direction, **kwargs
        )

    @classmethod
    def from_three_points(
        cls, origin: POINT3, point_width: POINT3, point_height: POINT3, **kwargs
    ):
        """
        Create rectangular plane from 3 points defining origin and two edges.
        Similar to CAD 3-point rectangle.
        """
        o = np.array(origin, dtype=np.float64)
        pw = np.array(point_width, dtype=np.float64)
        ph = np.array(point_height, dtype=np.float64)

        # Calculate edges
        width_vec = pw - o
        height_vec = ph - o

        # Calculate dimensions
        width = np.linalg.norm(width_vec)
        height = np.linalg.norm(height_vec)

        # Calculate center (fourth corner is opposite to origin)
        fourth_corner = pw + height_vec
        center = (o + pw + ph + fourth_corner) / 4

        # Normal from cross product
        normal = np.cross(width_vec, height_vec)
        normal = normalize(normal)

        return cls(
            center, normal, float(width), float(height), normalize(width_vec), **kwargs
        )

    def intersect(self, ray: OpticalRay) -> IntersectionResult:
        result = IntersectionResult()

        # First check intersection with infinite plane
        denom = np.dot(ray.current_direction, self.normal)

        if abs(denom) < VSMALL:
            return result

        t = np.dot(self.center - ray.current_position, self.normal) / denom

        if t < VSMALL:
            return result

        # Calculate intersection point
        intersection_point = ray.current_position + t * ray.current_direction

        # Check if point is within rectangle bounds
        point_rel = intersection_point - self.center
        width_coord = np.dot(point_rel, self.width_axis)
        height_coord = np.dot(point_rel, self.height_axis)

        if abs(width_coord) <= self.width / 2 and abs(height_coord) <= self.height / 2:
            result.hit = True
            result.distance = t
            result.point = intersection_point
            result.normal = self.normal if denom < 0.0 else -self.normal
            result.surface_id = self.surface_id

        return result

    def to_plotly_surface(
        self, bounds: tuple | None = None, show_normals: bool = False
    ) -> list[go.Mesh3d] | list[go.Mesh3d | go.Scatter3d]:
        import plotly.graph_objects as go

        # Get the four corners
        corners = self.get_corners()

        # Create two triangles to form the rectangle
        surface = go.Mesh3d(
            x=[corners[0][0], corners[1][0], corners[2][0], corners[3][0]],
            y=[corners[0][1], corners[1][1], corners[2][1], corners[3][1]],
            z=[corners[0][2], corners[1][2], corners[2][2], corners[3][2]],
            i=[0, 0],  # Triangle 1: 0-1-2, Triangle 2: 0-2-3
            j=[1, 2],
            k=[2, 3],
            opacity=0.3,
            color=get_surface_color(self.surface_id),
            name=f"Rectangular Plane {self.surface_id}",
            showscale=False,
        )

        if not show_normals:
            return [surface]

        # Add normal vector at center
        normal_end = self.center + self.normal * 0.5
        normal_trace = go.Scatter3d(
            x=[self.center[0], normal_end[0]],
            y=[self.center[1], normal_end[1]],
            z=[self.center[2], normal_end[2]],
            mode="lines",
            line=dict(width=4, color="green"),
            name=f"Normal {self.surface_id}",
            showlegend=False,
        )

        return [surface, normal_trace]

    def get_corners(self) -> tuple[POINT3, POINT3, POINT3, POINT3]:
        """Get the four corner points of the rectangle."""
        half_width = self.width / 2
        half_height = self.height / 2

        corner1 = (
            self.center - half_width * self.width_axis - half_height * self.height_axis
        )
        corner2 = (
            self.center + half_width * self.width_axis - half_height * self.height_axis
        )
        corner3 = (
            self.center + half_width * self.width_axis + half_height * self.height_axis
        )
        corner4 = (
            self.center - half_width * self.width_axis + half_height * self.height_axis
        )

        return corner1, corner2, corner3, corner4
