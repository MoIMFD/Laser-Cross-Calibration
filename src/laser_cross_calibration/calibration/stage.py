"""Motorized stage positioning for laser cross-calibration."""

from dataclasses import dataclass, field

import numpy as np
import pint

from ..coordinates import CoordinateSystem, WORLD_COORDINATE_SYSTEM
from ..types import POINT3, VECTOR3
from .utils import ensure_unit, DEFAULT_UNIT, StageOutOfLimitsError


@dataclass
class Stage:
    """
    Motorized 3-axis stage for laser cross-calibration.

    Controls positioning of laser sources and provides coordinate transformations
    between stage-local coordinates and world coordinates.

    Attributes:
        position_local: Current stage position in local coordinates
        arm1: Offset vector to first laser mount point
        arm2: Offset vector to second laser mount point
        beam1_normal: Direction vector for first laser beam
        beam2_normal: Direction vector for second laser beam
        limits: (min_pos, max_pos) limits for stage positioning
        coordinate_system: Stage coordinate system for world transformations
    """

    position_local: POINT3 = field(
        default_factory=lambda: np.zeros(3, dtype=float) * DEFAULT_UNIT
    )
    arm1: VECTOR3 = field(
        default_factory=lambda: np.array([100.0, 0.0, 0.0], dtype=np.float64)
        * DEFAULT_UNIT
    )
    arm2: VECTOR3 = field(
        default_factory=lambda: np.array([0.0, 100.0, 0.0], dtype=np.float64)
        * DEFAULT_UNIT
    )
    beam1_normal: VECTOR3 = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float64)
        * DEFAULT_UNIT
    )
    beam2_normal: VECTOR3 = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float64)
        * DEFAULT_UNIT
    )
    limits: tuple[VECTOR3, VECTOR3] = (
        np.array([-50.0, -50.0, -50.0]) * DEFAULT_UNIT,
        np.array([50.0, 50.0, 50.0]) * DEFAULT_UNIT,
    )
    coordinate_system: CoordinateSystem = field(
        default_factory=lambda: CoordinateSystem(
            origin=np.zeros(3),
            basis=np.eye(3),
            parent=WORLD_COORDINATE_SYSTEM,
            name="stage",
        )
    )

    def __post_init__(self):
        """Post-initialization setup."""
        # Ensure coordinate system has correct parent
        if self.coordinate_system.parent is None:
            self.coordinate_system.parent = WORLD_COORDINATE_SYSTEM

    @property
    def arm1_local_position(self) -> POINT3:
        """Get first laser arm position in local coordinates."""
        return ensure_unit(self.position_local) + self.arm1

    @property
    def arm2_local_position(self) -> POINT3:
        """Get second laser arm position in local coordinates."""
        return ensure_unit(self.position_local) + self.arm2

    def _is_within_local_limits(self, point: POINT3) -> bool:
        """Check if point is within stage limits."""
        return all((self.limits[0] <= point) & (point <= self.limits[1]))

    def _validate_local_position(self, point: POINT3) -> POINT3:
        """Validate that position is within stage limits."""
        if self._is_within_local_limits(point=point):
            return point
        else:
            raise StageOutOfLimitsError(point, self.limits)

    def set_local_position(self, position: POINT3, unit: pint.Unit = DEFAULT_UNIT):
        """
        Set stage to new local position.

        Args:
            position: New position coordinates
            unit: Unit for position values

        Returns:
            Self for method chaining
        """
        position = ensure_unit(value=position, unit=unit)
        self._validate_local_position(position)
        self.position_local = position
        return self

    def get_world_position(self) -> POINT3:
        """Get current stage position in world coordinates."""
        if hasattr(self.position_local, "magnitude"):
            local_pos = np.array(self.position_local.magnitude)
        else:
            local_pos = np.array(self.position_local)

        if self.coordinate_system:
            return self.coordinate_system.globalize(local_pos)
        else:
            return local_pos

    def get_world_arm_positions(self) -> tuple[POINT3, POINT3]:
        """Get laser arm positions in world coordinates."""
        if hasattr(self.arm1, "magnitude"):
            arm1_local = np.array(self.arm1.magnitude)
            arm2_local = np.array(self.arm2.magnitude)
        else:
            arm1_local = np.array(self.arm1)
            arm2_local = np.array(self.arm2)

        # Arm positions relative to stage position
        local_pos = self.get_world_position()
        arm1_world = local_pos + arm1_local
        arm2_world = local_pos + arm2_local

        return arm1_world, arm2_world

    def get_world_beam_directions(self) -> tuple[VECTOR3, VECTOR3]:
        """Get laser beam directions in world coordinates."""
        if hasattr(self.beam1_normal, "magnitude"):
            beam1_local = np.array(self.beam1_normal.magnitude)
            beam2_local = np.array(self.beam2_normal.magnitude)
        else:
            beam1_local = np.array(self.beam1_normal)
            beam2_local = np.array(self.beam2_normal)

        if self.coordinate_system:
            beam1_world = self.coordinate_system.globalize_direction(beam1_local)
            beam2_world = self.coordinate_system.globalize_direction(beam2_local)
        else:
            beam1_world = beam1_local
            beam2_world = beam2_local

        return beam1_world, beam2_world
