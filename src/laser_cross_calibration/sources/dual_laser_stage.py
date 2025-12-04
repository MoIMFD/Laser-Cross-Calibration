"""Single laser beam source."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self, Unpack

import plotly.graph_objects as go
from hazy.utils import check_same_frame

from laser_cross_calibration.sources.base import LaserSource, LaserSourceTypedDict

if TYPE_CHECKING:
    from hazy import Point, Vector

    from laser_cross_calibration.tracing.ray import OpticalRay


class DualLaserStageSource(LaserSource):
    """
    Single laser beam with fixed origin and direction.

    Simplest laser source - emits one ray from a specified position
    in a specified direction.
    """

    def __init__(
        self,
        origin: Point,
        arm1: Vector,
        arm2: Vector,
        direction1: Vector,
        direction2: Vector,
        **kwargs: Unpack[LaserSourceTypedDict],
    ):
        """
        Initialize single laser source.

        Args:
            origin: Starting position of the laser beam
            direction: Direction vector of the laser beam (will be normalized)
        """
        super().__init__(**kwargs)
        check_same_frame(origin, arm1, arm2, direction1, direction2)
        self.origin = origin
        self.arm1 = arm1
        self.arm2 = arm2
        self.direction1 = direction1.normalize()
        self.direction2 = direction2.normalize()

    @property
    def beam_origin1(self) -> Point:
        result: Point = self.origin + self.arm1
        return result

    @property
    def beam_origin2(self) -> Point:
        result: Point = self.origin + self.arm2
        return result

    def get_rays(self) -> list[OpticalRay]:
        """Generate single ray from this source."""
        from laser_cross_calibration.tracing.ray import OpticalRay

        return [
            OpticalRay(self.beam_origin1, self.direction1),
            OpticalRay(self.beam_origin2, self.direction2),
        ]

    def translate(self, x=0.0, y=0.0, z=0.0) -> Self:
        self.origin = self.origin + self.origin.frame.vector(x, y, z)
        return self

    def set_origin(self, origin: Point) -> Self:
        """Update laser origin position."""
        if origin.frame != self.origin.frame:
            raise TypeError(
                "Can not set origin in a different coordinate system, "
                f"expected {self.origin.frame} but got {origin.frame}"
            )
        self.origin = origin
        return self

    def get_origins(self) -> list[Point]:
        return [self.beam_origin1, self.beam_origin2]

    def get_directions(self) -> list[Vector]:
        return [self.direction1, self.direction2]

    def set_direction1(self, direction: Vector) -> Self:
        """Update laser direction."""
        if direction.frame != self.direction1.frame:
            raise TypeError(
                "Can not set direction in a different coordinate system, "
                f"expected {self.direction1.frame} but got {direction.frame}"
            )
        self.direction1 = direction
        return self

    def set_direction2(self, direction: Vector) -> Self:
        """Update laser direction."""
        if direction.frame != self.direction2.frame:
            raise TypeError(
                "Can not set direction in a different coordinate system, "
                f"expected {self.direction2.frame} but got {direction.frame}"
            )
        self.direction2 = direction
        return self

    def to_plotly(self) -> list[go.Cone | go.Scatter3d]:
        cones = super().to_plotly()

        arms: list[go.Scatter3d] = []
        origin_global = self.origin.to_global()
        for beam_origin in self.get_origins():
            beam_global = beam_origin.to_global()
            x = [origin_global.x, beam_global.x]
            y = [origin_global.y, beam_global.y]
            z = [origin_global.z, beam_global.z]
            arms.append(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    marker={"size": 0, "color": "red"},
                    line={"color": "red", "width": 3},
                    showlegend=False,
                    name="DualLaserStage Arms",
                )
            )

        origin = go.Scatter3d(
            x=[origin_global.x],
            y=[origin_global.y],
            z=[origin_global.z],
            marker={"size": 0, "color": "red", "symbol": "square"},
            line={"color": "red", "width": 0},
            showlegend=False,
            name="DualLaserStage Origin",
        )

        return cones + arms + [origin]
