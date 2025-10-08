from __future__ import annotations

from typing import Annotated

import numpy as np
from numpy import typing as npt

# 3D point coordinates (x, y, z)
POINT3 = Annotated[npt.NDArray[np.float64], "Shape[3]"]

# 3D vector (dx, dy, dz)
VECTOR3 = Annotated[npt.NDArray[np.float64], "Shape[3]"]

# 3x3 matrix
MATRIX3x3 = Annotated[npt.NDArray[np.float64], "Shape[3, 3]"]
