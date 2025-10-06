"""Surface definitions for ray-surface intersection calculations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from laser_cross_calibration.constants import VSMALL, VVSMALL
from laser_cross_calibration.types import POINT3, VECTOR3
from laser_cross_calibration.utils import normalize

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from laser_cross_calibration.ray_tracing.ray import OpticalRay


def _get_surface_color(surface_id: int) -> str:
    """
    Get a distinguishable color for a surface based on its ID.
    Uses a palette of visually distinct colors that cycle through.
    """
    # Carefully chosen colors that are visually distinct and colorblind-friendly
    color_palette = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf",  # Cyan
        "#aec7e8",  # Light blue
        "#ffbb78",  # Light orange
        "#98df8a",  # Light green
        "#ff9896",  # Light red
        "#c5b0d5",  # Light purple
    ]
    return color_palette[surface_id % len(color_palette)]


def _get_colorscale(surface_id: int) -> str:
    """
    Get a Plotly colorscale name based on surface ID.
    """
    colorscales = [
        "Blues",
        "Oranges",
        "Greens",
        "Reds",
        "Purples",
        "Greys",
        "YlOrRd",
        "YlGnBu",
        "RdYlBu",
        "Spectral",
        "Viridis",
        "Plasma",
        "Inferno",
        "Magma",
        "Cividis",
    ]
    return colorscales[surface_id % len(colorscales)]


@dataclass
class IntersectionResult:
    """
    Result of ray-surface intersection calculation.

    Attributes:
        hit: Whether an intersection was found
        distance: Distance from ray origin to intersection point
        point: 3D intersection point coordinates
        normal: Surface normal vector at intersection point
        surface_id: Identifier of the intersected surface
        triangle_id: ID of intersected triangle (for triangle meshes, -1 otherwise)
        barycentric_u: Barycentric coordinate u (for triangle meshes)
        barycentric_v: Barycentric coordinate v (for triangle meshes)
        barycentric_w: Barycentric coordinate w (for triangle meshes, w = 1-u-v)
    """

    hit: bool = False
    distance: float = np.inf
    point: POINT3 = field(default_factory=lambda: np.array([np.nan, np.nan, np.nan]))
    normal: VECTOR3 = field(default_factory=lambda: np.array([np.nan, np.nan, np.nan]))
    surface_id: int = -1
    triangle_id: int = -1
    barycentric_u: float = 0.0
    barycentric_v: float = 0.0
    barycentric_w: float = 0.0


class Surface(ABC):
    """
    Abstract base class for optical surfaces.

    All optical surfaces must implement ray intersection and normal calculation.

    Attributes:
        surface_id: Unique identifier for the surface
        material_name: Name of the surface material
    """

    def __init__(self, surface_id: int = 0, material_name: str = "unknown") -> None:
        """
        Initialize surface with identification.

        Args:
            surface_id: Unique identifier for this surface
            material_name: Human-readable material name
        """
        self.surface_id = surface_id
        self.material_name = material_name

    @abstractmethod
    def intersect(self, ray: OpticalRay) -> IntersectionResult:
        """
        Calculate intersection between ray and surface.

        Args:
            ray: Optical ray to intersect with surface

        Returns:
            Intersection result with hit information
        """
        pass

    @abstractmethod
    def get_normal_at_point(self, point: POINT3) -> VECTOR3:
        """
        Calculate surface normal at given point.

        Args:
            point: Point on the surface

        Returns:
            Normalized surface normal vector
        """
        pass

    @abstractmethod
    def to_plotly_surface(
        self, bounds: Optional[tuple] = None, show_normals: bool = False
    ) -> "go.Surface | list[go.Scatter3d]":
        """
        Convert surface to Plotly visualization objects.

        Args:
            bounds: Optional bounds for infinite surfaces (min_coord, max_coord)
            show_normals: Whether to include normal vectors in visualization

        Returns:
            Plotly Surface object, or list of traces if show_normals=True
        """
        pass

    def __repr__(self) -> str:
        """String representation of the surface."""
        return (
            f"{self.__class__.__qualname__}("
            f"surface_id={self.surface_id}, "
            f"material_name='{self.material_name}')"
        )


class Plane(Surface):
    """
    Infinite plane surface defined by a point and normal vector.

    Attributes:
        point: Reference point on the plane
        normal: Normalized normal vector to the plane
    """

    def __init__(self, point: POINT3, normal: VECTOR3, **kwargs) -> None:
        """
        Initialize infinite plane.

        Args:
            point: Any point lying on the plane
            normal: Normal vector to the plane (will be normalized)
            **kwargs: Additional arguments passed to Surface constructor

        Raises:
            ValueError: If normal vector is zero
        """
        super().__init__(**kwargs)

        self.point = np.asarray(point, dtype=np.float64)
        normal_array = np.asarray(normal, dtype=np.float64)

        if np.linalg.norm(normal_array) < VSMALL:
            raise ValueError("Normal vector cannot be zero")

        self.normal = normalize(normal_array)

    def intersect(self, ray: OpticalRay) -> IntersectionResult:
        result = IntersectionResult()

        denom = np.dot(ray.current_direction, self.normal)

        if abs(denom) < VSMALL:
            return result

        t = np.dot(self.point - ray.position, self.normal) / denom

        if t < VSMALL:
            return result

        result.hit = True
        result.distance = t
        result.point = ray.position + t * ray.current_direction
        result.normal = self.normal if denom < 0.0 else -self.normal
        result.surface_id = self.surface_id

        return result

    def get_normal_at_point(self, point: POINT3) -> VECTOR3:
        return self.normal.copy()

    def to_plotly_surface(
        self, bounds: Optional[tuple] = None, show_normals: bool = False
    ) -> "go.Surface | list[go.Surface | go.Scatter3d]":
        import plotly.graph_objects as go

        if bounds is None:
            bounds = (-2, 2)

        # Create grid points on the plane
        u_range = v_range = np.linspace(bounds[0], bounds[1], 10)
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
            self.point[:, np.newaxis, np.newaxis]
            + u_vec[:, np.newaxis, np.newaxis] * u_grid[np.newaxis, :, :]
            + v_vec[:, np.newaxis, np.newaxis] * v_grid[np.newaxis, :, :]
        )

        surface = go.Surface(
            x=points[0],
            y=points[1],
            z=points[2],
            opacity=0.3,
            colorscale=_get_colorscale(self.surface_id),
            showscale=False,
            name=f"Plane {self.surface_id}",
        )

        if not show_normals:
            return surface

        # Add normal vector
        normal_end = self.point + self.normal * 0.5
        normal_trace = go.Scatter3d(
            x=[self.point[0], normal_end[0]],
            y=[self.point[1], normal_end[1]],
            z=[self.point[2], normal_end[2]],
            mode="lines",
            line=dict(width=4, color="green"),
            name=f"Normal {self.surface_id}",
            showlegend=False,
        )

        return [surface, normal_trace]


class RectangularPlane(Surface):
    def __init__(
        self,
        center: POINT3,
        normal: VECTOR3,
        width: float,
        height: float,
        width_direction: VECTOR3 = None,
        **kwargs,
    ):
        """
        Rectangular plane defined by center, normal, and dimensions.

        Args:
            center: Center point of rectangle
            normal: Normal vector to the plane
            width: Width of rectangle
            height: Height of rectangle
            width_direction: Direction of width axis (will be projected to plane). If None, auto-generated.
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

        return cls(center, normal, width, height, width_direction, **kwargs)

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

        return cls(center, normal, width, height, normalize(width_vec), **kwargs)

    def intersect(self, ray: OpticalRay) -> IntersectionResult:
        result = IntersectionResult()

        # First check intersection with infinite plane
        denom = np.dot(ray.current_direction, self.normal)

        if abs(denom) < VSMALL:
            return result

        t = np.dot(self.center - ray.position, self.normal) / denom

        if t < VSMALL:
            return result

        # Calculate intersection point
        intersection_point = ray.position + t * ray.current_direction

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

    def get_normal_at_point(self, point: POINT3) -> VECTOR3:
        return self.normal.copy()

    def to_plotly_surface(
        self, bounds: Optional[tuple] = None, show_normals: bool = False
    ) -> "go.Surface | list[go.Scatter3d]":
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
            color=_get_surface_color(self.surface_id),
            name=f"Rectangular Plane {self.surface_id}",
            showscale=False,
        )

        if not show_normals:
            return surface

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


class InfiniteCylinder(Surface):
    def __init__(self, center: POINT3, axis: VECTOR3, radius: float, **kwargs):
        super().__init__(**kwargs)
        self.center = np.array(center, dtype=np.float64)
        self.axis = np.array(axis, dtype=np.float64)
        self.axis = normalize(self.axis)
        self.radius = float(radius)

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
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

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

    def get_normal_at_point(self, point: POINT3) -> VECTOR3:
        point_to_axis = point - self.center
        axis_component = np.dot(point_to_axis, self.axis) * self.axis
        return normalize(point_to_axis - axis_component)

    def to_plotly_surface(
        self, bounds: Optional[tuple] = None, show_normals: bool = False
    ) -> "go.Surface | list[go.Scatter3d]":
        import plotly.graph_objects as go

        if bounds is None:
            bounds = (-100, 100)

        # Generate cylinder surface in local coordinates
        theta = np.linspace(0, 2 * np.pi, 30)
        s_range = np.linspace(bounds[0], bounds[1], 20)  # Parameter along axis
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

        return go.Surface(
            x=points[0],
            y=points[1],
            z=points[2],
            opacity=0.4,
            colorscale=_get_colorscale(self.surface_id),
            showscale=False,
            name=f"Cylinder {self.surface_id}",
        )


class EllipticCylinder(Surface):
    def __init__(
        self,
        center: POINT3,
        axis: VECTOR3,
        major_radius: float,
        minor_radius: float,
        major_axis_direction: VECTOR3,
        **kwargs,
    ):
        """
        Elliptic cylinder with elliptical cross-section.

        Args:
            center: Center point of cylinder
            axis: Direction of cylinder axis (will be normalized)
            major_radius: Radius along major axis of ellipse
            minor_radius: Radius along minor axis of ellipse
            major_axis_direction: Direction of major axis in plane perpendicular to cylinder axis
        """
        super().__init__(**kwargs)
        self.center = np.array(center, dtype=np.float64)
        self.axis = np.array(axis, dtype=np.float64)
        self.axis = normalize(self.axis)
        self.major_radius = float(major_radius)
        self.minor_radius = float(minor_radius)

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
        t1 = (-B - sqrt_disc) / (2 * A)
        t2 = (-B + sqrt_disc) / (2 * A)

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

    def get_normal_at_point(self, point: POINT3) -> VECTOR3:
        point_rel = point - self.center
        u_comp = np.dot(point_rel, self.major_axis)
        v_comp = np.dot(point_rel, self.minor_axis)

        a = self.major_radius
        b = self.minor_radius

        normal_local = (2 * u_comp / (a**2)) * self.major_axis + (
            2 * v_comp / (b**2)
        ) * self.minor_axis
        return normalize(normal_local)

    def to_plotly_surface(
        self, bounds: Optional[tuple] = None, show_normals: bool = False
    ) -> "go.Surface | list[go.Scatter3d]":
        import plotly.graph_objects as go

        if bounds is None:
            bounds = (-2, 2)

        # Generate elliptic cylinder surface in local coordinates
        theta = np.linspace(0, 2 * np.pi, 30)
        s_range = np.linspace(bounds[0], bounds[1], 20)  # Parameter along axis
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

        return go.Surface(
            x=points[0],
            y=points[1],
            z=points[2],
            opacity=0.4,
            colorscale=_get_colorscale(self.surface_id),
            showscale=False,
            name=f"Elliptic Cylinder {self.surface_id}",
        )


class FiniteCylinder(InfiniteCylinder):
    def __init__(
        self, center: POINT3, axis: VECTOR3, radius: float, length: float, **kwargs
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
            result.point = np.array([np.nan, np.nan, np.nan])
            result.normal = np.array([np.nan, np.nan, np.nan])
            result.surface_id = -1

        return result

    def to_plotly_surface(
        self, bounds: Optional[tuple] = None, show_normals: bool = False
    ) -> "go.Surface | list[go.Scatter3d]":
        # Override parent method to use cylinder length as default bounds
        if bounds is None:
            bounds = (-self.length / 2, self.length / 2)

        # Use parent class implementation with proper bounds
        return super().to_plotly_surface(bounds, show_normals)


class TriSurface(Surface):
    """
    Base class for triangulated surfaces (STL, OBJ, PLY, etc.).

    Handles generic triangle mesh operations including ray-triangle intersection
    and visualization for any triangulated geometry.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        smooth: bool = True,
        scaling=np.ones(3),
        translation=np.zeros(3),
        **kwargs,
    ):
        """
        Initialize triangulated surface.

        Args:
            vertices: (N, 3) array of vertex coordinates
            faces: (M, 3) array of triangle vertex indices
            smooth: Whether to use smooth normal interpolation (True) or flat triangle normals (False)
        """
        super().__init__(**kwargs)

        # Store and validate mesh data
        self.vertices = np.asarray(vertices, dtype=np.float64) * scaling + translation
        self.faces = np.asarray(faces, dtype=np.int32)
        self.smooth = smooth
        self._validate_mesh()

        # Precompute triangle data for performance
        self.triangle_normals = self._compute_triangle_normals()

        # Compute vertex normals for smooth shading
        if self.smooth:
            self.vertex_normals = self._compute_vertex_normals()

    def _validate_mesh(self):
        """Validate mesh geometry."""
        if self.vertices.shape[1] != 3:
            raise ValueError(
                f"Vertices must have shape (N, 3), got {self.vertices.shape}"
            )
        if self.faces.shape[1] != 3:
            raise ValueError(f"Faces must have shape (M, 3), got {self.faces.shape}")
        if self.faces.max() >= len(self.vertices):
            raise ValueError("Face indices exceed vertex array bounds")
        if self.faces.min() < 0:
            raise ValueError("Face indices must be non-negative")

    def _compute_triangle_normals(self) -> np.ndarray:
        """Compute normal vector for each triangle."""
        # Get triangle vertices
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        # Compute triangle edges
        edge1 = v1 - v0
        edge2 = v2 - v0

        # Cross product gives normal
        normals = np.cross(edge1, edge2)

        # Normalize
        norms = np.linalg.norm(normals, axis=1)
        # Avoid division by zero for degenerate triangles
        valid_mask = norms > VSMALL
        normals[valid_mask] = normals[valid_mask] / norms[valid_mask, np.newaxis]

        return normals

    def _compute_vertex_normals(self) -> np.ndarray:
        """
        Compute smooth vertex normals by averaging adjacent triangle normals.
        Uses area-weighted averaging for better quality (standard approach).
        """
        vertex_normals = np.zeros_like(self.vertices)

        # For each triangle, add its contribution to each vertex
        for i, face in enumerate(self.faces):
            triangle_normal = self.triangle_normals[i]

            # Area weighting: triangle area is 0.5 * |normal| before normalization
            v0, v1, v2 = self.vertices[face]
            triangle_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

            # Add area-weighted normal to each vertex
            for vertex_idx in face:
                vertex_normals[vertex_idx] += triangle_normal * triangle_area

        # Normalize vertex normals
        norms = np.linalg.norm(vertex_normals, axis=1)
        valid_mask = norms > VSMALL
        vertex_normals[valid_mask] = (
            vertex_normals[valid_mask] / norms[valid_mask, np.newaxis]
        )

        return vertex_normals

    def intersect(self, ray: OpticalRay) -> IntersectionResult:
        """
        Ray-triangle mesh intersection using Möller-Trumbore algorithm.
        Tests ray against all triangles and returns closest hit.
        """
        result = IntersectionResult()
        closest_distance = np.inf

        # Ray origin and direction
        ray_origin = ray.position
        ray_dir = ray.current_direction

        # Get triangle vertices
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        # Möller-Trumbore algorithm (vectorized)
        edge1 = v1 - v0
        edge2 = v2 - v0

        # Begin calculating determinant - also used to calculate u parameter
        h = np.cross(ray_dir, edge2)
        det = np.sum(edge1 * h, axis=1)

        # If determinant is near zero, ray lies in plane of triangle
        valid_mask = np.abs(det) > VSMALL

        if not np.any(valid_mask):
            return result

        # Calculate distance from v0 to ray origin
        s = ray_origin - v0

        # Calculate u parameter and test bound
        u = np.sum(s * h, axis=1) / (det + VVSMALL)
        u_mask = (u >= 0.0) & (u <= 1.0)

        # Combine masks
        valid_mask = valid_mask & u_mask
        if not np.any(valid_mask):
            return result

        # Prepare to test v parameter
        q = np.cross(s, edge1)

        # Calculate v parameter and test bound
        v = np.sum(ray_dir * q, axis=1) / (det + VVSMALL)
        v_mask = (v >= 0.0) & (u + v <= 1.0)

        # Combine masks
        valid_mask = valid_mask & v_mask
        if not np.any(valid_mask):
            return result

        # Calculate t (distance along ray)
        t = np.sum(edge2 * q, axis=1) / (det + VVSMALL)
        t_mask = t > VSMALL  # Only forward intersections

        # Final valid hits
        valid_mask = valid_mask & t_mask
        if not np.any(valid_mask):
            return result

        # Find closest hit
        valid_distances = t[valid_mask]
        closest_idx = np.argmin(valid_distances)
        triangle_idx = np.where(valid_mask)[0][closest_idx]

        # Get barycentric coordinates from the intersection
        u_closest = u[triangle_idx]
        v_closest = v[triangle_idx]
        w_closest = 1.0 - u_closest - v_closest

        # Build intersection result
        result.hit = True
        result.distance = valid_distances[closest_idx]
        result.point = ray_origin + result.distance * ray_dir
        result.surface_id = self.surface_id
        result.triangle_id = triangle_idx
        result.barycentric_u = u_closest
        result.barycentric_v = v_closest
        result.barycentric_w = w_closest

        # Compute normal using smooth interpolation or flat triangle normal
        if self.smooth:
            # Professional barycentric interpolation of vertex normals
            face = self.faces[triangle_idx]
            result.normal = (
                w_closest * self.vertex_normals[face[0]]  # w corresponds to vertex 0
                + u_closest * self.vertex_normals[face[1]]  # u corresponds to vertex 1
                + v_closest * self.vertex_normals[face[2]]  # v corresponds to vertex 2
            )
            result.normal = normalize(result.normal)
        else:
            # Flat shading - use triangle normal
            result.normal = self.triangle_normals[triangle_idx]

        return result

    def get_normal_at_point(self, point: POINT3) -> VECTOR3:
        """
        Get surface normal at point.
        Note: This is an approximation since we don't have barycentric coordinates.
        For accurate normals, use the normal from intersection results.
        """
        # Find closest triangle center (approximation)
        triangle_centers = np.mean(self.vertices[self.faces], axis=1)
        distances = np.linalg.norm(triangle_centers - point, axis=1)
        closest_triangle = np.argmin(distances)

        if self.smooth:
            # For smooth surfaces, return interpolated normal at triangle center
            face = self.faces[closest_triangle]
            # Use equal weights (1/3 each) for triangle center
            normal = (
                self.vertex_normals[face[0]]
                + self.vertex_normals[face[1]]
                + self.vertex_normals[face[2]]
            ) / 3.0
            return normalize(normal)
        else:
            return self.triangle_normals[closest_triangle]

    def to_plotly_surface(
        self, bounds: Optional[tuple] = None, show_normals: bool = False
    ) -> "go.Surface | list[go.Scatter3d]":
        import plotly.graph_objects as go

        # Create triangle mesh
        mesh = go.Mesh3d(
            x=self.vertices[:, 0],
            y=self.vertices[:, 1],
            z=self.vertices[:, 2],
            i=self.faces[:, 0],
            j=self.faces[:, 1],
            k=self.faces[:, 2],
            opacity=0.2,
            color=_get_surface_color(self.surface_id),
            name=f"Triangle Mesh {self.surface_id}",
            showscale=False,
        )

        if not show_normals:
            return mesh

        # Add triangle normal vectors (sample only some for clarity)
        triangle_centers = np.mean(self.vertices[self.faces], axis=1)
        n_samples = min(50, len(triangle_centers))  # Limit to avoid clutter
        sample_indices = np.linspace(0, len(triangle_centers) - 1, n_samples, dtype=int)

        centers = triangle_centers[sample_indices]
        normals = self.triangle_normals[sample_indices] * 0.1  # Scale for visibility

        normal_traces = []
        for i, (center, normal) in enumerate(zip(centers, normals)):
            end_point = center + normal
            normal_traces.append(
                go.Scatter3d(
                    x=[center[0], end_point[0]],
                    y=[center[1], end_point[1]],
                    z=[center[2], end_point[2]],
                    mode="lines",
                    line=dict(width=2, color="red"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        return [mesh] + normal_traces

    @classmethod
    def from_arrays(cls, vertices: np.ndarray, faces: np.ndarray, **kwargs):
        """Create triangle surface from raw vertex and face arrays."""
        return cls(vertices, faces, **kwargs)


class STLSurface(TriSurface):
    """
    STL (STereoLithography) file format surface.

    Supports both ASCII and binary STL files commonly used in 3D printing
    and CAD applications.
    """

    @classmethod
    def from_file(cls, stl_path: str, **kwargs):
        """
        Load STL file and create surface.

        Args:
            stl_path: Path to STL file
            **kwargs: Additional arguments passed to surface constructor

        Returns:
            STLSurface instance
        """
        # try:
        # Try numpy-stl first (handles both ASCII and binary)
        from stl import mesh as stl_mesh

        stl_data = stl_mesh.Mesh.from_file(str(stl_path))

        # STL stores triangles as vectors, need to extract vertices
        # Check if smooth shading is requested in kwargs
        smooth = kwargs.get("smooth", True)
        vertices, faces = cls._extract_vertices_faces(stl_data.vectors, smooth)

        # except ImportError:
        #     # Fallback to manual parsing if numpy-stl not available
        #     vertices, faces = cls._parse_stl_manual(stl_path)

        return cls(vertices, faces, **kwargs)

    @staticmethod
    def _extract_vertices_faces(
        triangle_vectors: np.ndarray, smooth: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract vertices and face indices from STL triangle vectors.

        Args:
            triangle_vectors: (N, 3, 3) array where each triangle is 3 vertices
            smooth: If True, deduplicate vertices for smooth shading. If False, keep separate vertices.

        Returns:
            Tuple of (vertices, faces) arrays
        """
        if not smooth:
            # For flat shading, no need to deduplicate - keep all vertices separate
            vertices = triangle_vectors.reshape(-1, 3)
            faces = np.arange(len(vertices)).reshape(-1, 3)
            return vertices, faces

        # For smooth shading, deduplicate vertices
        all_vertices = triangle_vectors.reshape(-1, 3)

        # Fast deduplication using lexicographic sorting
        tolerance_decimals = 6
        rounded_vertices = np.round(all_vertices, tolerance_decimals)

        # Use numpy unique with return_inverse and return_index to get mapping
        unique_vertices, first_indices, inverse_indices = np.unique(
            rounded_vertices, axis=0, return_inverse=True, return_index=True
        )

        # Create faces array from inverse indices
        faces = inverse_indices.reshape(-1, 3)

        # Use original precision for actual vertices using first occurrence indices
        final_vertices = all_vertices[first_indices]

        return final_vertices, faces

    # @staticmethod
    # def _parse_stl_manual(stl_path: str) -> tuple[np.ndarray, np.ndarray]:
    #     """
    #     Manual STL parsing fallback (basic ASCII STL support).
    #     """
    #     vertices = []
    #     faces = []

    #     with open(stl_path, "r") as f:
    #         lines = f.readlines()

    #     current_vertices = []
    #     vertex_index = 0

    #     for line in lines:
    #         line = line.strip().lower()
    #         if line.startswith("vertex"):
    #             coords = [float(x) for x in line.split()[1:4]]
    #             vertices.append(coords)
    #             current_vertices.append(vertex_index)
    #             vertex_index += 1

    #         elif line.startswith("endfacet"):
    #             if len(current_vertices) == 3:
    #                 faces.append(current_vertices)
    #             current_vertices = []

    #     return np.array(vertices), np.array(faces)
