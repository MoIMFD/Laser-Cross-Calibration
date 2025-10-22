"""Single laser beam source."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np

from laser_cross_calibration.sources.base import LaserSource
from laser_cross_calibration.tracing.ray import OpticalRay

if TYPE_CHECKING:
    from laser_cross_calibration.types import POINT3, VECTOR3


class DualLaserStageSource(LaserSource):
    """
    Single laser beam with fixed origin and direction.

    Simplest laser source - emits one ray from a specified position
    in a specified direction.
    """

    def __init__(
        self,
        origin: POINT3,
        arm1: VECTOR3,
        arm2: VECTOR3,
        direction1: VECTOR3,
        direction2: VECTOR3,
    ):
        """
        Initialize single laser source.

        Args:
            origin: Starting position of the laser beam
            direction: Direction vector of the laser beam (will be normalized)
        """
        self.origin = np.asarray(origin, dtype=np.float64)
        self.arm1 = np.asanyarray(arm1, dtype=np.float64)
        self.arm2 = np.asanyarray(arm2, dtype=np.float64)
        self.direction1 = np.asanyarray(direction1, dtype=np.float64)
        self.direction2 = np.asanyarray(direction2, dtype=np.float64)

    @property
    def beam_origin1(self) -> POINT3:
        return self.origin + self.arm1

    @property
    def beam_origin2(self) -> POINT3:
        return self.origin + self.arm2

    def get_rays(self) -> list[OpticalRay]:
        """Generate single ray from this source."""
        return [
            OpticalRay(self.beam_origin1, self.direction1),
            OpticalRay(self.beam_origin2, self.direction2),
        ]

    def translate(self, x=0.0, y=0.0, z=0.0) -> Self:
        self.origin = self.origin + np.array((x, y, z))
        return self

    def set_origin(self, origin: POINT3) -> None:
        """Update laser origin position."""
        self.origin = np.asarray(origin, dtype=np.float64)

    def get_origins(self) -> list[POINT3]:
        return [self.beam_origin1, self.beam_origin2]

    def get_directions(self) -> list[VECTOR3]:
        return [self.direction1, self.direction2]

    def set_direction1(self, direction: VECTOR3) -> None:
        """Update laser direction."""
        self.direction1 = np.asarray(direction, dtype=np.float64)

    def set_direction2(self, direction: VECTOR3) -> None:
        """Update laser direction."""
        self.direction2 = np.asarray(direction, dtype=np.float64)
