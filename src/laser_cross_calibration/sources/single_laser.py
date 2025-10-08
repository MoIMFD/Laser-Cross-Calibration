"""Single laser beam source."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from laser_cross_calibration.sources.base import LaserSource
from laser_cross_calibration.tracing.ray import OpticalRay

if TYPE_CHECKING:
    from laser_cross_calibration.types import POINT3, VECTOR3


class SingleLaserSource(LaserSource):
    """
    Single laser beam with fixed origin and direction.

    Simplest laser source - emits one ray from a specified position
    in a specified direction.
    """

    def __init__(self, origin: POINT3, direction: VECTOR3):
        """
        Initialize single laser source.

        Args:
            origin: Starting position of the laser beam
            direction: Direction vector of the laser beam (will be normalized)
        """
        self.origin = np.asarray(origin, dtype=np.float64)
        self.direction = np.asarray(direction, dtype=np.float64)

    def get_rays(self) -> list[OpticalRay]:
        """Generate single ray from this source."""
        return [OpticalRay(self.origin, self.direction)]

    def set_origin(self, origin: POINT3) -> None:
        """Update laser origin position."""
        self.origin = np.asarray(origin, dtype=np.float64)

    def get_origins(self) -> list[POINT3]:
        return [self.origin]

    def get_directions(self) -> list[VECTOR3]:
        return [self.direction]

    def set_direction(self, direction: VECTOR3) -> None:
        """Update laser direction."""
        self.direction = np.asarray(direction, dtype=np.float64)
