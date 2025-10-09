from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from laser_cross_calibration.types import POINT3, VECTOR3

VSMALL: float = 1e-8
VVSMALL: float = 1e-12  # For preventing division by zero
INTERSECTION_THRESHOLD: float = 1e-4

NAN_POINT3: POINT3 = np.array((np.nan, np.nan, np.nan))
ORIGIN_POINT3: POINT3 = np.zeros(3)

NAN_VECTOR3: VECTOR3 = np.array((np.nan, np.nan, np.nan))
ZERO_VECTOR3: VECTOR3 = np.zeros(3)
UNIT_X_VECTOR3: VECTOR3 = np.array((1, 0, 0))
UNIT_Y_VECTOR3: VECTOR3 = np.array((0, 1, 0))
UNIT_Z_VECTOR3: VECTOR3 = np.array((0, 0, 1))
