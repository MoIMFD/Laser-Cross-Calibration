from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from laser_cross_calibration.coordinate_system.primitives import GeometricPrimitive


def check_same_frame(*objects: Iterable[GeometricPrimitive]):
    """Verify that two geometric primitives are in the same reference frame.

    Args:
        object1: First geometric primitive
        object2: Second geometric primitive

    Raises:
        RuntimeError: If objects are in different frames
    """
    if len(objects) <= 1:
        return
    if not all(objects[0].frame == obj.frame for obj in objects[1:]):
        mixed_systems = set([obj.frame for obj in objects])
        raise RuntimeError(
            "Expected all objects to be in the same coordinate system, "
            f"got {mixed_systems}"
        )
