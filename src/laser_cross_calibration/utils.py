from __future__ import annotations

import numpy as np


def normalize(array: np.typing.NDArray) -> np.typing.NDArray:
    magnitude = np.linalg.norm(array)
    return array / magnitude
