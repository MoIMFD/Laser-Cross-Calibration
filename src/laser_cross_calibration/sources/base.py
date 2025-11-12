"""Base class for laser sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NotRequired, TypedDict

import plotly.graph_objects as go

if TYPE_CHECKING:
    from hazy import Point, Vector

    from laser_cross_calibration.tracing.ray import OpticalRay


class LaserSourceTypedDict(TypedDict):
    display_scale: NotRequired[float]


class LaserSource(ABC):
    """
    Abstract base class for laser beam sources.

    All laser sources must implement get_rays() to provide
    one or more rays for tracing.
    """

    def __init__(self, display_scale: float = 1.0):
        self.display_scale = display_scale

    @abstractmethod
    def get_rays(self) -> list[OpticalRay]:
        """
        Generate rays from this source.

        Returns:
            List of OpticalRay objects
        """
        pass

    def to_plotly(self) -> list[go.Cone]:
        traces = []
        for origin, direction in zip(
            self.get_origins(), self.get_directions(), strict=False
        ):
            trace = go.Cone(
                x=[origin.to_global().x],
                y=[origin.to_global().y],
                z=[origin.to_global().z],
                u=[direction.to_global().x * self.display_scale],
                v=[direction.to_global().y * self.display_scale],
                w=[direction.to_global().z * self.display_scale],
                showscale=False,
                colorscale=[[0, "red"], [1, "red"]],
                name=f"{self.__class__.__qualname__}",
            )
            traces.append(trace)

        return traces

    @abstractmethod
    def get_origins(self) -> list[Point]:
        pass

    @abstractmethod
    def get_directions(self) -> list[Vector]:
        pass
