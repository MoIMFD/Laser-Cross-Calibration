"""Base classes for optical surfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from laser_cross_calibration.constants import (
    NAN_POINT3,
    NAN_VECTOR3,
)
from laser_cross_calibration.coordinate_system import CoordinateSystem

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from laser_cross_calibration.tracing.ray import OpticalRay
    from laser_cross_calibration.types import POINT3, VECTOR3


def get_surface_color(surface_id: int) -> str:
    """
    Get a distinguishable color for a surface based on its ID.
    Uses a palette of visually distinct colors that cycle through.
    """
    color_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
    ]
    return color_palette[surface_id % len(color_palette)]


def get_colorscale(surface_id: int) -> str:
    """Get a Plotly colorscale name based on surface ID."""
    colorscales = [
        "Blues",
        "Oranges",
        "Greens",
        "Reds",
        "Purples",
        "Greys",
        "YlOrRd",
        "YlGnBu",
        "RdYlBu",
        "Spectral",
        "Viridis",
        "Plasma",
        "Inferno",
        "Magma",
        "Cividis",
    ]
    return colorscales[surface_id % len(colorscales)]


@dataclass
class IntersectionResult:
    """
    Result of ray-surface intersection calculation.

    Attributes:
        hit: Whether an intersection was found
        distance: Distance from ray origin to intersection point
        point: 3D intersection point coordinates
        normal: Surface normal vector at intersection point
        surface_id: Identifier of the intersected surface
        triangle_id: ID of intersected triangle (for triangle meshes, -1 otherwise)
        barycentric_u: Barycentric coordinate u (for triangle meshes)
        barycentric_v: Barycentric coordinate v (for triangle meshes)
        barycentric_w: Barycentric coordinate w (for triangle meshes, w = 1-u-v)
    """

    hit: bool = False
    distance: float = np.inf
    point: POINT3 = field(default_factory=lambda: NAN_POINT3.copy())
    normal: VECTOR3 = field(default_factory=lambda: NAN_VECTOR3.copy())
    surface_id: int = -1
    triangle_id: int = -1
    barycentric_u: float = float("nan")
    barycentric_v: float = float("nan")
    barycentric_w: float = float("nan")


class Surface(ABC):
    """
    Abstract base class for optical surfaces.

    All optical surfaces must implement ray intersection and normal calculation.

    Attributes:
        surface_id: Unique identifier for the surface
        material_name: Name of the surface material
    """

    _id_counter: int = 0

    def __init__(
        self,
        surface_id: int | None = None,
        info: str = "unknown",
        coordinate_system: CoordinateSystem | None = None,
    ) -> None:
        """
        Initialize surface with identification.

        Args:
            surface_id: Unique identifier for this surface (auto-assigned if None)
            material_name: Human-readable material name
        """
        if surface_id is None:
            self.surface_id = Surface._id_counter
            Surface._id_counter += 1
        elif isinstance(surface_id, int):
            if surface_id < 0:
                raise ValueError(f"Surface id must be non-negative, got {surface_id}")
            self.surface_id = surface_id
            Surface._id_counter = max(Surface._id_counter, surface_id + 1)
        else:
            raise ValueError(f"Surface id must be type int, got <{type(surface_id)}>")
        self.coordinate_system = coordinate_system or CoordinateSystem()
        self.info: str = info

    @classmethod
    def reset_id_counter(cls) -> None:
        """Reset the surface ID counter to 0. Useful for testing/notebooks."""
        cls._id_counter = 0

    @abstractmethod
    def intersect(self, ray: OpticalRay) -> IntersectionResult:
        """
        Calculate intersection between ray and surface.

        Args:
            ray: Optical ray to intersect with surface

        Returns:
            Intersection result with hit information
        """

    def get_bounds(self) -> tuple[POINT3, POINT3] | None:
        """
        Get axis-aligned bounding box for this surface.

        Returns:
            Tuple of (min_point, max_point) or None if unbounded
        """
        return None

    @abstractmethod
    def to_plotly_surface(
        self, show_normals: bool = False
    ) -> go.Surface | list[go.Scatter3d]:
        """
        Convert surface to Plotly visualization objects.

        Args:
            bounds: Optional bounds for infinite surfaces (min_coord, max_coord)
            show_normals: Whether to include normal vectors in visualization

        Returns:
            Plotly Surface object, or list of traces if show_normals=True
        """

    def __repr__(self) -> str:
        """String representation of the surface."""
        return (
            f"{self.__class__.__qualname__}("
            f"surface_id={self.surface_id}, "
            f"info='{self.info}')"
        )
