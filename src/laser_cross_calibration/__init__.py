"""Module for creating calibration grids and gcodes for the laser cross calibration."""

__version__ = "0.0.1"

from . import calibration
from . import ray_tracing
from . import gcode
from . import optimize
from . import coordinates
from .calibration import Stage, CalibrationPoint
from .ray_tracing import OpticalSystem
from .optimize.inverse import find_stage_position_for_intersection
from .coordinates import CoordinateSystem, CoordinateManager

__all__ = [
    "calibration", "ray_tracing", "gcode", "optimize", "coordinates",
    "Stage", "CalibrationPoint", "OpticalSystem", 
    "find_stage_position_for_intersection", "CoordinateSystem", "CoordinateManager"
]
