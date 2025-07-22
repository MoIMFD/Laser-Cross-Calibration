"""Surface definitions for ray-surface intersection calculations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..constants import VSMALL
from ..types import POINT3, VECTOR3
from ..utils import normalize

if TYPE_CHECKING:
    from .ray import OpticalRay


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
    """
    hit: bool = False
    distance: float = np.inf
    point: POINT3 = field(default_factory=lambda: np.array([np.nan, np.nan, np.nan]))
    normal: VECTOR3 = field(default_factory=lambda: np.array([np.nan, np.nan, np.nan]))
    surface_id: int = -1


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


class RectangularPlane(Surface):
    def __init__(self, center: POINT3, normal: VECTOR3, width: float, height: float, 
                 width_direction: VECTOR3 = None, **kwargs):
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
    def from_corners(cls, corner1: POINT3, corner2: POINT3, corner3: POINT3, corner4: POINT3, **kwargs):
        """
        Create rectangular plane from 4 corner points.
        Points should be in order (e.g., clockwise or counter-clockwise).
        """
        c1, c2, c3, c4 = [np.array(p, dtype=np.float64) for p in [corner1, corner2, corner3, corner4]]
        
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
    def from_three_points(cls, origin: POINT3, point_width: POINT3, point_height: POINT3, **kwargs):
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
        
        if (abs(width_coord) <= self.width / 2 and 
            abs(height_coord) <= self.height / 2):
            result.hit = True
            result.distance = t
            result.point = intersection_point
            result.normal = self.normal if denom < 0.0 else -self.normal
            result.surface_id = self.surface_id
        
        return result
    
    def get_normal_at_point(self, point: POINT3) -> VECTOR3:
        return self.normal.copy()
    
    def get_corners(self) -> tuple[POINT3, POINT3, POINT3, POINT3]:
        """Get the four corner points of the rectangle."""
        half_width = self.width / 2
        half_height = self.height / 2
        
        corner1 = self.center - half_width * self.width_axis - half_height * self.height_axis
        corner2 = self.center + half_width * self.width_axis - half_height * self.height_axis
        corner3 = self.center + half_width * self.width_axis + half_height * self.height_axis
        corner4 = self.center - half_width * self.width_axis + half_height * self.height_axis
        
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
        d_perp = ray.current_direction - np.dot(ray.current_direction, self.axis) * self.axis
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


class EllipticCylinder(Surface):
    def __init__(
        self, 
        center: POINT3, 
        axis: VECTOR3, 
        major_radius: float,
        minor_radius: float,
        major_axis_direction: VECTOR3,
        **kwargs
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
        A = (d_u/a)**2 + (d_v/b)**2
        B = 2 * ((oc_u*d_u)/(a**2) + (oc_v*d_v)/(b**2))
        C = (oc_u/a)**2 + (oc_v/b)**2 - 1
        
        discriminant = B**2 - 4*A*C
        
        if discriminant < 0:
            return result
        
        # Calculate intersections
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-B - sqrt_disc) / (2*A)
        t2 = (-B + sqrt_disc) / (2*A)
        
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
        normal_local = (2*u_comp/(a**2)) * self.major_axis + (2*v_comp/(b**2)) * self.minor_axis
        result.normal = normalize(normal_local)
        result.surface_id = self.surface_id
        
        return result

    def get_normal_at_point(self, point: POINT3) -> VECTOR3:
        point_rel = point - self.center
        u_comp = np.dot(point_rel, self.major_axis)
        v_comp = np.dot(point_rel, self.minor_axis)
        
        a = self.major_radius
        b = self.minor_radius
        
        normal_local = (2*u_comp/(a**2)) * self.major_axis + (2*v_comp/(b**2)) * self.minor_axis
        return normalize(normal_local)
