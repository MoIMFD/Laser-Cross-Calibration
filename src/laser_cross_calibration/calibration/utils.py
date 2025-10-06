"""Utility functions and exceptions for calibration module."""

import pint

from laser_cross_calibration.types import POINT3

# Unit registry and defaults
ureg = pint.UnitRegistry()
DEFAULT_UNIT = ureg.mm


def ensure_unit(value, unit=None):
    """Ensure a value has units, adding default if needed."""
    if hasattr(value, "units"):
        return value
    else:
        return value * (unit or DEFAULT_UNIT)


class StageOutOfLimitsError(ValueError):
    """Exception raised when stage position is outside allowed limits."""

    def __init__(self, point: POINT3, limits: tuple, message: str = None):
        self.point = point
        self.limits = limits

        if message is None:
            message = f"Point {point} is outside of limits {limits}"

        super().__init__(message)
