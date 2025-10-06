"""Ray tracing module for optical simulation."""

from __future__ import annotations

import numpy as np

from laser_cross_calibration.constants import INTERSECTION_THRESHOLD, VSMALL
from laser_cross_calibration.types import POINT3, VECTOR3
from laser_cross_calibration.utils import normalize
from laser_cross_calibration.ray_tracing.materials import BaseMaterial


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

    def __init__(self, origin: POINT3, direction: VECTOR3) -> None:
        """
        Initialize optical ray.

        Args:
            origin: Starting position of the ray
            direction: Initial direction vector (will be normalized)

        Raises:
            ValueError: If direction vector is zero or origin/direction have wrong shape
        """
        # Validate and convert inputs
        origin_array = np.asarray(origin, dtype=np.float64)
        direction_array = np.asarray(direction, dtype=np.float64)

        if origin_array.shape != (3,):
            raise ValueError(f"Origin must be 3D point, got shape {origin_array.shape}")
        if direction_array.shape != (3,):
            raise ValueError(
                f"Direction must be 3D vector, got shape {direction_array.shape}"
            )

        direction_norm = np.linalg.norm(direction_array)
        if direction_norm < VSMALL:
            raise ValueError("Direction vector cannot be zero")

        # Store initial state
        self.origin: POINT3 = origin_array
        self.direction: VECTOR3 = normalize(direction_array)

        # Current state
        self.position: POINT3 = self.origin.copy()
        self.current_direction: VECTOR3 = self.direction.copy()
        self.is_alive: bool = True

        # Path history - starts with initial state
        self.path_positions: list[POINT3] = [self.origin.copy()]
        self.path_directions: list[VECTOR3] = [self.direction.copy()]
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
        self.position = self.position + distance * self.current_direction

        # Record path history
        self.path_positions.append(self.position.copy())
        self.path_directions.append(self.current_direction.copy())
        self.segment_distances.append(distance)
        self.media_history.append(medium)

    def refract(
        self,
        surface_normal: VECTOR3,
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

        normal = normalize(surface_normal)

        # Calculate incident angle
        cos_theta_i = -np.dot(self.current_direction, normal)

        # Ensure normal points toward incoming ray
        if cos_theta_i < 0:
            normal = -normal
            cos_theta_i = -cos_theta_i

        # Apply Snell's law
        n_ratio = medium_from.n() / medium_to.n()
        sin_theta_i_sq = 1.0 - cos_theta_i**2
        sin_theta_t_sq = n_ratio**2 * sin_theta_i_sq

        # Check for total internal reflection
        if sin_theta_t_sq > 1.0:
            # Total internal reflection - reflect ray
            self.current_direction = self.current_direction + 2 * cos_theta_i * normal
            return False

        # Calculate refracted direction
        cos_theta_t = np.sqrt(1.0 - sin_theta_t_sq)
        self.current_direction = (
            n_ratio * self.current_direction
            + (n_ratio * cos_theta_i - cos_theta_t) * normal
        )

        self.current_direction = normalize(self.current_direction)
        return True

    def get_point_at_distance(self, distance: float) -> POINT3:
        """
        Calculate point along current ray direction at given distance.

        Args:
            distance: Distance from current position

        Returns:
            Point at the specified distance along the ray
        """
        return self.position + distance * self.current_direction

    def __repr__(self) -> str:
        """String representation of the optical ray."""
        return (
            f"{self.__class__.__qualname__}("
            f"position={self.position}, "
            f"direction={self.current_direction}, "
            f"is_alive={self.is_alive})"
        )

    @classmethod
    def ray_x(cls, origin=np.zeros(3)):
        direction = np.zeros(3)
        direction[0] = 1.0
        return cls(origin=origin, direction=direction)

    @classmethod
    def ray_y(cls, origin=np.zeros(3)):
        direction = np.zeros(3)
        direction[1] = 1.0
        return cls(origin=origin, direction=direction)

    @classmethod
    def ray_z(cls, origin=np.zeros(3)):
        direction = np.zeros(3)
        direction[2] = 1.0
        return cls(origin=origin, direction=direction)


def line_segment_intersection(
    p1: POINT3,
    p2: POINT3,
    p3: POINT3,
    p4: POINT3,
    threshold: float = INTERSECTION_THRESHOLD,
) -> tuple[bool, POINT3]:
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
    # Convert to numpy arrays
    p1, p2, p3, p4 = [np.array(p, dtype=np.float64) for p in [p1, p2, p3, p4]]

    # Direction vectors of the segments
    d1 = p2 - p1  # Segment 1 vector
    d2 = p4 - p3  # Segment 2 vector

    # Vector between segment start points
    w = p1 - p3

    # Check if segments are degenerate (zero length)
    len1_sq = np.dot(d1, d1)
    len2_sq = np.dot(d2, d2)

    if len1_sq < threshold or len2_sq < threshold:
        return False, np.array([np.nan, np.nan, np.nan])

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

        return False, np.array([np.nan, np.nan, np.nan])

    # Solve for parameters of closest approach
    s = (b * e - c * d) / denom  # Parameter for segment 1
    t = (a * e - b * d) / denom  # Parameter for segment 2

    # Check if parameters are within [0, 1] (inside segments)
    if not (0 <= s <= 1 and 0 <= t <= 1):
        return False, np.array([np.nan, np.nan, np.nan])

    # Calculate closest points
    point1 = p1 + s * d1
    point2 = p3 + t * d2

    # Check if points are within threshold distance
    distance = np.linalg.norm(point2 - point1)

    if distance < threshold:
        # Return midpoint as intersection
        intersection_point = (point1 + point2) / 2
        return True, intersection_point
    else:
        return False, np.array([np.nan, np.nan, np.nan])


def ray_intersection(
    ray1: OpticalRay, ray2: OpticalRay, threshold: float = VSMALL
) -> list[POINT3]:
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
