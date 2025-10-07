"""Ray tracing module for optical simulation."""

from laser_cross_calibration.tracing.intersection import (
    line_segment_intersection,
    ray_intersection,
)
from laser_cross_calibration.tracing.optical_interface import OpticalInterface
from laser_cross_calibration.tracing.optical_system import OpticalSystem
from laser_cross_calibration.tracing.ray import OpticalRay
from laser_cross_calibration.tracing.tracer import RayTracer

__all__ = [
    "OpticalRay",
    "OpticalInterface",
    "OpticalSystem",
    "RayTracer",
    "line_segment_intersection",
    "ray_intersection",
]
