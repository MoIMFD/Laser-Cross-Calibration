"""Single laser beam source."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np

from laser_cross_calibration.coordinate_system.utils import check_same_frame
from laser_cross_calibration.sources.base import LaserSource
from laser_cross_calibration.tracing.ray import OpticalRay

if TYPE_CHECKING:
    from laser_cross_calibration.coordinate_system import Point, Vector


class SingleLaserSource(LaserSource):
    """
    Single laser beam with fixed origin and direction.

    Simplest laser source - emits one ray from a specified position
    in a specified direction.
    """

    def __init__(self, origin: Point, direction: Vector, **kwargs):
        """
        Initialize single laser source.

        Args:
            origin: Starting position of the laser beam
            direction: Direction vector of the laser beam (will be normalized)
        """
        super().__init__(**kwargs)
        check_same_frame(origin, direction)
        self.origin = origin
        self.direction = direction

    def get_rays(self) -> list[OpticalRay]:
        """Generate single ray from this source."""
        return [OpticalRay(self.origin, self.direction)]

    def set_origin(self, origin: Point) -> Self:
        """Update laser origin position."""
        if origin.frame != self.direction.frame:
            raise TypeError(
                "Can not set direction in a different coordinate system, "
                f"expected {self.origin.frame} but got {origin.frame}"
            )
        self.origin = origin

    def get_origins(self) -> list[Point]:
        return [self.origin]

    def get_directions(self) -> list[Vector]:
        return [self.direction]

    def set_direction(self, direction: Vector) -> Self:
        """Update laser direction."""
        if direction.frame != self.direction.frame:
            raise TypeError(
                "Can not set direction in a different coordinate system, "
                f"expected {self.direction.frame} but got {direction.frame}"
            )
        self.direction = direction
        return self
