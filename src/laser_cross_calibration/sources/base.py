"""Base class for laser sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import plotly.graph_objects as go


if TYPE_CHECKING:
    from laser_cross_calibration.tracing.ray import OpticalRay
    from laser_cross_calibration.types import POINT3, VECTOR3


class LaserSource(ABC):
    """
    Abstract base class for laser beam sources.

    All laser sources must implement get_rays() to provide
    one or more rays for tracing.
    """

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
        for origin, direction in zip(self.get_origins(), self.get_directions()):
            trace = go.Cone(
                x=[origin[0]],
                y=[origin[1]],
                z=[origin[2]],
                u=[direction[0]],
                v=[direction[1]],
                w=[direction[2]],
                showscale=False,
                colorscale=[[0, "red"], [1, "red"]],
                name=f"{self.__class__.__qualname__}",
            )
            traces.append(trace)

        return traces

    @abstractmethod
    def get_origins(self) -> list[POINT3]:
        pass

    @abstractmethod
    def get_directions(self) -> list[VECTOR3]:
        pass
