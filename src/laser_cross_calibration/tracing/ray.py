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
            float(n_ratio) * local_ray.current_direction
            + float(n_ratio * cos_theta_i - cos_theta_t) * normal
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
        direction = Vector.unit_x(frame=origin.frame)
        return cls(origin=origin, direction=direction)

    @classmethod
    def ray_y(cls, origin: Point) -> OpticalRay:
        """Create a ray facing towards positive y axis."""
        direction = Vector.unit_y(frame=origin.frame)
        return cls(origin=origin, direction=direction)

    @classmethod
    def ray_z(cls, origin: Point) -> OpticalRay:
        """Create a ray facing towards positive z axis."""
        direction = Vector.unit_z(frame=origin.frame)
        return cls(origin=origin, direction=direction)
