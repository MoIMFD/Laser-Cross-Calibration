"""Surface definitions for ray-surface intersection calculations."""

from laser_cross_calibration.surfaces.base import (
    IntersectionResult,
    Surface,
    get_colorscale,
    get_surface_color,
)

from laser_cross_calibration.surfaces.stl import TriSurface, StlSurface
from laser_cross_calibration.surfaces.plane import Plane
from laser_cross_calibration.surfaces.infinite_cylinder import InfiniteCylinder
from laser_cross_calibration.surfaces.finite_cylinder import FiniteCylinder
from laser_cross_calibration.surfaces.elliptic_cylinder import EllipticCylinder

__all__ = [
    "Surface",
    "IntersectionResult",
    "TriSurface",
    "StlSurface",
    "Plane",
    "InfiniteCylinder",
    "FiniteCylinder",
    "EllipticCylinder",
    "get_surface_color",
    "get_colorscale",
]
