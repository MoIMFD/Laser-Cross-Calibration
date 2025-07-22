"""Calibration point data structures for PIV applications."""

from dataclasses import dataclass

from ..types import POINT3


@dataclass
class CalibrationPoint:
    """
    A calibration point for PIV camera calibration.
    
    Represents the fundamental data unit: a known 3D intersection point
    and the corresponding stage position that generated it.
    
    Attributes:
        intersection_position: 3D position where laser beams intersect
        stage_position: Stage position that generated this intersection
    """
    intersection_position: POINT3
    stage_position: POINT3