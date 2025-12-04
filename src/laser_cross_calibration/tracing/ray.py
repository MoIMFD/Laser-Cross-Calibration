"""Ray tracing module for optical simulation."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from hazy import (
    Point,
    Vector,
)
from hazy.utils import check_same_frame
from scipy.spatial.transform import Rotation as R

from laser_cross_calibration.constants import (
    INTERSECTION_THRESHOLD,
    VSMALL,
)

if TYPE_CHECKING:
    from hazy import Frame

    from laser_cross_calibration.materials.base import BaseMaterial


class OpticalRay:
    """
    Represents a ray of light with position, direction, and propagation history.

    The ray tracks its complete path through optical media, including
    positions, directions, and distances traveled in each segment.

    Attributes:
        origin: Initial starting position of the ray
        direction: Initial normalized direction vector
        position: Current position of the ray
        current_direction: Current normalized direction vector
        is_alive: Whether the ray is still propagating
        path_positions: History of all positions along the ray path
        path_directions: History of all direction vectors
        segment_distances: Distance traveled in each segment
        media_history: Media encountered in each segment
    """

    def __init__(self, origin: Point, direction: Vector) -> None:
        """
        Initialize optical ray.

        Args:
            origin: Starting position of the ray
            direction: Initial direction vector (will be normalized)

        Raises:
            ValueError: If direction vector is zero or origin/direction have wrong shape
        """
        # Validate and convert inputs, use np.array to create an independent copy
        check_same_frame(origin, direction)
        # Store initial state
        self.origin = origin
        self.initial_direction = direction.normalize()

        # Current state
        self.current_position: Point = self.origin.copy()
        self.current_direction: Vector = self.initial_direction.copy()
        self.is_alive: bool = True

        # Path history - starts with initial state
        self.path_positions: list[Point] = [self.origin.copy()]
        self.path_directions: list[Vector] = [self.initial_direction.copy()]
        self.segment_distances: list[float] = []
        self.media_history: list[BaseMaterial] = []

    def propagate(self, distance: float, medium: BaseMaterial) -> None:
        """
        Propagate the ray through a medium by a given distance.

        Args:
            distance: Distance to propagate (must be non-negative)
            medium: Medium through which to propagate
        """
        if not self.is_alive or distance <= 0.0:
            return

        # Update position
        self.current_position = self.current_position + (
            self.current_direction * distance
        )

        # Record path history
        self.path_positions.append(self.current_position.copy())
        self.path_directions.append(self.current_direction.copy())
        self.segment_distances.append(distance)
        self.media_history.append(medium)

    def refract(
        self,
        surface_normal: Vector,
        medium_from: BaseMaterial,
        medium_to: BaseMaterial,
    ) -> bool:
        """
        Apply Snell's law refraction at a surface interface.

        Args:
            surface_normal: Surface normal vector (will be normalized)
            medium_from: BaseMaterial the ray is coming from
            medium_to: BaseMaterial the ray is entering

        Returns:
            True if refraction occurred, False if total internal reflection
        """
        if not self.is_alive:
            return False

        local_ray = self.localize(surface_normal.frame)

        normal = surface_normal.normalize()

        # Calculate incident angle
        cos_theta_i = -np.dot(local_ray.current_direction, normal)

        # Ensure normal points toward incoming ray
        if cos_theta_i < 0:
            normal = -1 * normal
            cos_theta_i = -cos_theta_i

        # Apply Snell's law
        n_ratio = medium_from.n() / medium_to.n()
        sin_theta_i_sq = 1.0 - cos_theta_i**2
        sin_theta_t_sq = n_ratio**2 * sin_theta_i_sq

        # Check for total internal reflection
        if sin_theta_t_sq > 1.0:
            # Total internal reflection - reflect ray
            local_ray.current_direction = (
                local_ray.current_direction + 2 * cos_theta_i * normal
            )

            # Update the last direction in path history to reflect the reflection
            if len(local_ray.path_directions) > 0:
                local_ray.path_directions[-1] = local_ray.current_direction.copy()

            return False

        # Calculate refracted direction
        cos_theta_t = np.sqrt(1.0 - sin_theta_t_sq)
        local_ray.current_direction = (
            n_ratio * local_ray.current_direction
            + (n_ratio * cos_theta_i - cos_theta_t) * normal
        )

        local_ray.current_direction = local_ray.current_direction.normalize()

        # Update the last direction in path history to reflect refraction
        # This ensures path_directions[i] represents the direction FROM position[i]
        if len(local_ray.path_directions) > 0:
            local_ray.path_directions[-1] = local_ray.current_direction.copy()

        # Transform the refracted ray back to the original frame
        global_ray = local_ray.globalize()
        self.current_direction = global_ray.current_direction.to_frame(
            self.origin.frame
        )
        if len(self.path_directions) > 0:
            self.path_directions[-1] = self.current_direction.copy()

        return True

    def get_point_at_distance(self, distance: float) -> Point:
        """
        Calculate point along current ray direction at given distance.

        Args:
            distance: Distance from current position

        Returns:
            Point at the specified distance along the ray
        """
        return self.current_position + distance * self.current_direction

    def __repr__(self) -> str:
        """String representation of the optical ray."""
        return (
            f"{self.__class__.__qualname__}("
            f"position={self.current_position}, "
            f"direction={self.current_direction}, "
            f"is_alive={self.is_alive})"
        )

    def translate(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> OpticalRay:
        """Translate the complete ray, including its history, used for coordinate
        transformation.
        """
        translation = Vector(x, y, z, frame=self.origin.frame)
        self.origin += translation
        self.current_position += translation
        self.path_positions = [pos + translation for pos in self.path_positions]
        return self

    def rotate(self, rx: float = 0.0, ry: float = 0.0, rz: float = 0.0) -> OpticalRay:
        """Rotate the complete ray, including its history, used for coordinate
        transformation.
        """
        rotation = R.from_euler("xyz", [rx, ry, rz], degrees=False)

        # rotate positions
        self.origin = rotation.apply(self.origin)
        self.current_position = rotation.apply(self.current_position)
        self.path_positions = [rotation.apply(pos) for pos in self.path_positions]

        # rotate directions
        self.initial_direction = rotation.apply(self.initial_direction)
        self.current_direction = rotation.apply(self.current_direction)
        self.path_directions = [rotation.apply(dir) for dir in self.path_directions]
        return self

    def localize(self, frame: Frame) -> OpticalRay:
        local_ray = self.copy()
        local_ray.origin = local_ray.origin.to_frame(frame)
        local_ray.initial_direction = local_ray.initial_direction.to_frame(frame)
        local_ray.current_position = local_ray.current_position.to_frame(frame)
        local_ray.current_direction = local_ray.current_direction.to_frame(frame)
        local_ray.path_positions = [
            position.to_frame(frame) for position in local_ray.path_positions
        ]
        local_ray.path_directions = [
            direction.to_frame(frame) for direction in local_ray.path_directions
        ]
        return local_ray

    def globalize(self) -> OpticalRay:
        return self.localize(frame=self.origin.frame.root)

    def copy(self) -> OpticalRay:
        """Create a new OpticalRay instance as an independent copy of the current
        one.
        """
        copied_ray = OpticalRay(origin=self.origin, direction=self.initial_direction)
        copied_ray.origin = self.origin.copy()
        copied_ray.initial_direction = self.initial_direction.copy()
        copied_ray.current_position = self.current_position.copy()
        copied_ray.current_direction = self.current_direction.copy()
        copied_ray.is_alive = self.is_alive
        copied_ray.media_history = deepcopy(self.media_history)
        copied_ray.path_positions = [pos.copy() for pos in self.path_positions]
        copied_ray.path_directions = [dir.copy() for dir in self.path_directions]
        copied_ray.segment_distances = self.segment_distances.copy()

        return copied_ray

    @classmethod
    def ray_x(cls, origin: Point) -> OpticalRay:
        """Create a ray facing towards positive x axis."""
        direction = Vector.create_unit_x(frame=origin.frame)
        return cls(origin=origin, direction=direction)

    @classmethod
    def ray_y(cls, origin: Point) -> OpticalRay:
        """Create a ray facing towards positive y axis."""
        direction = Vector.create_unit_y(frame=origin.frame)
        return cls(origin=origin, direction=direction)

    @classmethod
    def ray_z(cls, origin=Point) -> OpticalRay:
        """Create a ray facing towards positive z axis."""
        direction = Vector.create_unit_z(frame=origin.frame)
        return cls(origin=origin, direction=direction)


def line_segment_intersection(
    p1: Point,
    p2: Point,
    p3: Point,
    p4: Point,
    threshold: float = INTERSECTION_THRESHOLD,
) -> tuple[bool, Point | None]:
    """
    Find intersection between two 3D line segments.

    Uses the closest approach method for two line segments in 3D space.

    Args:
        p1, p2: Start and end points of first line segment
        p3, p4: Start and end points of second line segment
        threshold: Distance threshold for considering intersection

    Returns:
        Tuple of (intersection_found, intersection_point)
    """
    check_same_frame(p1, p2, p3, p4)

    # Convert to numpy arrays
    p1, p2, p3, p4 = [np.array(p.coords, dtype=np.float64) for p in [p1, p2, p3, p4]]

    # Direction vectors of the segments
    d1 = p2 - p1  # Segment 1 vector
    d2 = p4 - p3  # Segment 2 vector

    # Vector between segment start points
    w = p1 - p3

    # Check if segments are degenerate (zero length)
    len1_sq = np.dot(d1, d1)
    len2_sq = np.dot(d2, d2)

    if len1_sq < threshold or len2_sq < threshold:
        return False, None

    # Dot products
    a = len1_sq  # d1·d1
    b = np.dot(d1, d2)  # d1·d2
    c = len2_sq  # d2·d2
    d = np.dot(d1, w)  # d1·w
    e = np.dot(d2, w)  # d2·w

    denom = a * c - b * b

    # Check if lines are parallel
    if abs(denom) < threshold:
        # Lines are parallel - check if they overlap
        # Project p3 onto line 1
        if a > threshold:
            t = -d / a
            closest_on_1 = p1 + t * d1
            dist_to_line2 = np.linalg.norm(closest_on_1 - p3)

            if dist_to_line2 < threshold:
                # Lines are coincident - find overlap
                t1_start = 0
                t1_end = 1
                t2_start = np.dot(p3 - p1, d1) / len1_sq
                t2_end = np.dot(p4 - p1, d1) / len1_sq

                # Find overlap interval
                overlap_start = max(t1_start, min(t2_start, t2_end))
                overlap_end = min(t1_end, max(t2_start, t2_end))

                if overlap_start <= overlap_end:
                    # Return midpoint of overlap
                    t_mid = (overlap_start + overlap_end) / 2
                    intersection_point = p1 + t_mid * d1
                    return True, intersection_point

        return False, None

    # Solve for parameters of closest approach
    s = (b * e - c * d) / denom  # Parameter for segment 1
    t = (a * e - b * d) / denom  # Parameter for segment 2

    # Check if parameters are within [0, 1] (inside segments)
    if not (0 <= s <= 1 and 0 <= t <= 1):
        return False, None

    # Calculate closest points
    point1 = p1 + s * d1
    point2 = p3 + t * d2

    # Check if points are within threshold distance
    distance = np.linalg.norm(point2 - point1)

    if distance < threshold:
        # Return midpoint as intersection
        intersection_point = (point1 + point2) / 2
        return True, Point(*intersection_point, frame=p1.frame)
    else:
        return False, None


def ray_intersection(
    ray1: OpticalRay, ray2: OpticalRay, threshold: float = VSMALL
) -> list[Point]:
    """
    Find all intersection points between two rays' path segments.

    Checks every segment of ray1 against every segment of ray2 to find
    where the actual physical ray paths intersect.

    Args:
        ray1: First optical ray
        ray2: Second optical ray
        threshold: Distance threshold for considering intersection

    Returns:
        List of intersection points (empty if no intersections found)
    """
    intersections = []

    # Check all segments of ray1 against all segments of ray2
    for i in range(len(ray1.segment_distances)):
        # Ray1 segment endpoints
        p1_start = ray1.path_positions[i]
        p1_end = ray1.path_positions[i + 1]

        for j in range(len(ray2.segment_distances)):
            # Ray2 segment endpoints
            p2_start = ray2.path_positions[j]
            p2_end = ray2.path_positions[j + 1]

            # Check intersection between these two segments
            intersects, point = line_segment_intersection(
                p1_start, p1_end, p2_start, p2_end, threshold=threshold
            )

            if intersects:
                intersections.append(point)

    return intersections
