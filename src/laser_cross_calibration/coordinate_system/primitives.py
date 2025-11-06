"""Geometric primitives with frame awareness for coordinate system transformations.

This module provides Point and Vector classes that carry frame information and
support arithmetic operations with proper type semantics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self, TypeVar

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from laser_cross_calibration.coordinate_system.frame import Frame


class GeometricPrimitive:
    """Base class for geometric primitives (Point and Vector) with frame awareness.

    Uses homogeneous coordinates (x, y, z, w) for unified transformation handling.
    Points have w=1, Vectors have w=0.
    """
    def __init__(self, x: float, y: float, z: float, w: float, frame: Frame):
        """Initialize geometric primitive in homogeneous coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            w: Homogeneous coordinate (1 for Point, 0 for Vector)
            frame: Reference frame for this primitive
        """
        self._homogeneous = np.array([x, y, z, w], dtype=float)
        self.frame = frame

    @property
    def coords(self) -> NDArray[np.floating]:
        """Return Cartesian coordinates (x, y, z) as numpy array."""
        return self._homogeneous[:3]

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Return Cartesian coordinates for numpy operations.

        Enables usage like: np.array(point) or np.add(point1, point2)

        Args:
            dtype: Desired array data type
            copy: Whether to copy the data

        Returns:
            Numpy array of Cartesian coordinates
        """
        coords = self._homogeneous[:3]
        if dtype is not None:
            coords = coords.astype(dtype)
        return coords if copy is False else coords.copy()

    @property
    def x(self) -> float:
        """X coordinate."""
        return self._homogeneous[0]

    @property
    def y(self) -> float:
        """Y coordinate."""
        return self._homogeneous[1]

    @property
    def z(self) -> float:
        """Z coordinate."""
        return self._homogeneous[2]

    def __getitem__(self, index: int) -> float:
        """Access coordinates by index: primitive[0] for x, primitive[1] for y, etc."""
        return self.coords[index]

    def __iter__(self):
        """Iterate over Cartesian coordinates."""
        return iter(self.coords)

    def to_frame(self, target_frame: Frame):
        """Transform this primitive to a different reference frame.

        Args:
            target_frame: Target reference frame

        Returns:
            New primitive of same type in target frame
        """
        transformation = self.frame.transform_to(target=target_frame)
        x, y, z, w = transformation @ self._homogeneous

        if isinstance(self, Point):
            # Points (w=1): normalize by w after transformation
            return type(self)(x=x / w, y=y / w, z=z / w, w=1.0, frame=target_frame)
        else:
            # Vectors (w=0): do not normalize, w stays 0
            return type(self)(x=x, y=y, z=z, w=0.0, frame=target_frame)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"x={self.x}, "
            f"y={self.y}, "
            f"z={self.z}, "
            f"frame={self.frame.name})"
        )


class Vector(GeometricPrimitive):
    """Geometric vector representing direction and magnitude.

    Vectors have homogeneous coordinate w=0, making them invariant to translation.

    Arithmetic semantics:
        - Vector + Vector = Vector (combine displacements)
        - Vector - Vector = Vector (difference of displacements)
        - Vector + Point = Point (displace position)
        - Vector - Point = ERROR (undefined operation)
    """

    def __init__(self, x: float, y: float, z: float, frame: Frame, *, w=0.0):
        """Initialize vector in given frame.

        Args:
            x: X component
            y: Y component
            z: Z component
            frame: Reference frame
            w: Homogeneous coordinate (should be 0 for vectors)
        """
        super().__init__(x=x, y=y, z=z, w=w, frame=frame)

    def __add__(
        self, other: Self | Vector | NDArray[np.floating]
    ) -> Point | Vector | NDArray[np.floating]:
        """Add vector to another vector or point.

        Args:
            other: Vector, Point, or numpy array

        Returns:
            Vector if adding to Vector, Point if adding to Point

        Raises:
            RuntimeError: If frames don't match
        """
        if isinstance(other, Point):
            check_same_frame(self, other)
            x, y, z = np.add(self, other)
            return Point(x, y, z, frame=self.frame)
        elif isinstance(other, Vector):
            check_same_frame(self, other)
            x, y, z = np.add(self, other)
            return Vector(x, y, z, frame=self.frame)
        else:
            return other.__add__(self)

    def __sub__(
        self, other: Self | Vector | NDArray[np.floating]
    ) -> Vector | NDArray[np.floating]:
        """Subtract vector from this vector.

        Args:
            other: Vector or numpy array

        Returns:
            Resulting vector

        Raises:
            TypeError: If attempting to subtract Point from Vector
            RuntimeError: If frames don't match
        """
        if isinstance(other, Point):
            raise TypeError("Cannot subtract Point from Vector")
        elif isinstance(other, Vector):
            check_same_frame(self, other)
            x, y, z = np.subtract(self, other)
            return Vector(x, y, z, frame=self.frame)
        else:
            return other.__sub__(self)


class Point(GeometricPrimitive):
    """Geometric point representing position in space.

    Points have homogeneous coordinate w=1, making them affected by translation.

    Arithmetic semantics:
        - Point - Point = Vector (displacement between positions)
        - Point + Vector = Point (displace position)
        - Point - Vector = Point (displace position backwards)
        - Point + Point = ERROR (undefined operation)
    """

    def __init__(self, x: float, y: float, z: float, frame: Frame, *, w=1.0):
        """Initialize point in given frame.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            frame: Reference frame
            w: Homogeneous coordinate (should be 1 for points)
        """
        super().__init__(x=x, y=y, z=z, w=w, frame=frame)

    def __sub__(
        self, other: Self | Vector | NDArray[np.floating]
    ) -> Point | Vector | NDArray[np.floating]:
        """Subtract point or vector from this point.

        Args:
            other: Point, Vector, or numpy array

        Returns:
            Vector if subtracting Point, Point if subtracting Vector

        Raises:
            RuntimeError: If frames don't match
        """
        if isinstance(other, Point):
            check_same_frame(self, other)
            x, y, z = np.subtract(self, other)
            return Vector(x, y, z, frame=self.frame)
        elif isinstance(other, Vector):
            check_same_frame(self, other)
            x, y, z = np.subtract(self, other)
            return Point(x, y, z, frame=self.frame)
        else:
            return other.__sub__(self)

    def __add__(
        self, other: Self | Vector | NDArray[np.floating]
    ) -> Point | Vector | NDArray[np.floating]:
        """Add vector to this point.

        Args:
            other: Vector or numpy array

        Returns:
            Resulting point

        Raises:
            TypeError: If attempting to add two Points
            RuntimeError: If frames don't match
        """
        if isinstance(other, Point):
            raise TypeError("Can not add 2 Points.")
        elif isinstance(other, Vector):
            check_same_frame(self, other)
            x, y, z = np.add(self, other)
            return Point(x, y, z, frame=self.frame)
        else:
            return other.__add__(self)


def check_same_frame(object1: GeometricPrimitive, object2: GeometricPrimitive):
    """Verify that two geometric primitives are in the same reference frame.

    Args:
        object1: First geometric primitive
        object2: Second geometric primitive

    Raises:
        RuntimeError: If objects are in different frames
    """
    if object1.frame != object2.frame:
        raise RuntimeError(
            "Can only process objects in same coordinate system, "
            f"got frame 1 {object1.frame.name} "
            f"and frame 2 {object2.frame.name}. "
            "Consider transforming prior to operation."
        )
