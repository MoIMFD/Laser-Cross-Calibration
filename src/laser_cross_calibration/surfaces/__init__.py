"""Surface definitions for ray-surface intersection calculations."""

from __future__ import annotations

from laser_cross_calibration.surfaces.base import (
    IntersectionResult,
    Surface,
    get_colorscale,
    get_surface_color,
)
from laser_cross_calibration.surfaces.elliptic_cylinder import EllipticCylinder
from laser_cross_calibration.surfaces.finite_cylinder import FiniteCylinder
from laser_cross_calibration.surfaces.infinite_cylinder import InfiniteCylinder
from laser_cross_calibration.surfaces.plane import Plane
from laser_cross_calibration.surfaces.triangulated import TriSurface

__all__ = [
    "Surface",
    "IntersectionResult",
    "TriSurface",
    "Plane",
    "InfiniteCylinder",
    "FiniteCylinder",
    "EllipticCylinder",
    "get_surface_color",
    "get_colorscale",
]
