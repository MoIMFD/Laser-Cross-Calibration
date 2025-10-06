"""Coordinate system management inspired by Optiland."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Union

import numpy as np

from laser_cross_calibration.types import POINT3, VECTOR3, MATRIX3x3
from laser_cross_calibration.utils import normalize


@dataclass
class CoordinateSystem:
    """
    3D coordinate system with origin and orthonormal basis.

    Inspired by Optiland's coordinate system management for robust
    transformations between different reference frames.

    Attributes:
        origin: Origin point of the coordinate system
        basis: 3x3 orthonormal basis matrix [e1, e2, e3] as rows
        parent: Parent coordinate system (for hierarchical transforms)
        name: Optional identifier for the coordinate system
    """

    origin: POINT3 = field(default_factory=lambda: np.zeros(3))
    basis: MATRIX3x3 = field(default_factory=lambda: np.eye(3))
    parent: Optional["CoordinateSystem"] = None
    name: str = "local"

    def __post_init__(self):
        """Validate coordinate system after initialization."""
        self.origin = np.asarray(self.origin, dtype=np.float64)
        self.basis = np.asarray(self.basis, dtype=np.float64)

        # Validate basis is orthonormal
        if not self._is_orthonormal(self.basis, tolerance=1e-10):
            # Try to orthonormalize
            self.basis = self._orthonormalize(self.basis)

        if not self._is_orthonormal(self.basis, tolerance=1e-8):
            raise ValueError(
                f"Coordinate system '{self.name}' basis is not orthonormal"
            )

    @staticmethod
    def _is_orthonormal(basis: MATRIX3x3, tolerance: float = 1e-10) -> bool:
        """Check if basis matrix is orthonormal."""
        # Check if each row is unit length
        norms = np.linalg.norm(basis, axis=1)
        if not np.allclose(norms, 1.0, atol=tolerance):
            return False

        # Check if rows are orthogonal
        dot_products = [
            np.dot(basis[0], basis[1]),
            np.dot(basis[0], basis[2]),
            np.dot(basis[1], basis[2]),
        ]
        if not np.allclose(dot_products, 0.0, atol=tolerance):
            return False

        # Check right-handed system
        cross_product = np.cross(basis[0], basis[1])
        if not np.allclose(cross_product, basis[2], atol=tolerance):
            return False

        return True

    @staticmethod
    def _orthonormalize(basis: MATRIX3x3) -> MATRIX3x3:
        """Orthonormalize basis using Gram-Schmidt process."""
        # Gram-Schmidt orthogonalization
        e1 = normalize(basis[0])

        e2 = basis[1] - np.dot(basis[1], e1) * e1
        e2 = normalize(e2)

        e3 = basis[2] - np.dot(basis[2], e1) * e1 - np.dot(basis[2], e2) * e2
        e3 = normalize(e3)

        return np.array([e1, e2, e3])

    @property
    def e1(self) -> VECTOR3:
        """First basis vector (X-axis)."""
        return self.basis[0]

    @property
    def e2(self) -> VECTOR3:
        """Second basis vector (Y-axis)."""
        return self.basis[1]

    @property
    def e3(self) -> VECTOR3:
        """Third basis vector (Z-axis)."""
        return self.basis[2]

    def localize(self, global_point: POINT3) -> POINT3:
        """
        Transform point from global coordinates to this local system.

        Args:
            global_point: Point in global coordinates

        Returns:
            Point in local coordinates
        """
        global_point = np.asarray(global_point)

        # If we have a parent, first transform to parent coordinates
        if self.parent is not None:
            parent_point = self.parent.localize(global_point)
        else:
            parent_point = global_point

        # Transform from parent to local coordinates
        relative_point = parent_point - self.origin
        local_point = self.basis @ relative_point  # basis rows as transformation matrix

        return local_point

    def globalize(self, local_point: POINT3) -> POINT3:
        """
        Transform point from this local system to global coordinates.

        Args:
            local_point: Point in local coordinates

        Returns:
            Point in global coordinates
        """
        local_point = np.asarray(local_point)

        # Transform from local to parent coordinates
        parent_point = (
            self.basis.T @ local_point + self.origin
        )  # basis.T for inverse transform

        # If we have a parent, transform further to global
        if self.parent is not None:
            global_point = self.parent.globalize(parent_point)
        else:
            global_point = parent_point

        return global_point

    def localize_direction(self, global_direction: VECTOR3) -> VECTOR3:
        """
        Transform direction vector from global to local coordinates.

        Note: Direction vectors don't include origin translation.

        Args:
            global_direction: Direction in global coordinates

        Returns:
            Direction in local coordinates
        """
        global_direction = np.asarray(global_direction)

        # If we have a parent, first transform to parent coordinates
        if self.parent is not None:
            parent_direction = self.parent.localize_direction(global_direction)
        else:
            parent_direction = global_direction

        # Transform direction (no origin translation)
        local_direction = self.basis @ parent_direction

        return local_direction

    def globalize_direction(self, local_direction: VECTOR3) -> VECTOR3:
        """
        Transform direction vector from local to global coordinates.

        Args:
            local_direction: Direction in local coordinates

        Returns:
            Direction in global coordinates
        """
        local_direction = np.asarray(local_direction)

        # Transform from local to parent coordinates (no origin translation)
        parent_direction = self.basis.T @ local_direction

        # If we have a parent, transform further to global
        if self.parent is not None:
            global_direction = self.parent.globalize_direction(parent_direction)
        else:
            global_direction = parent_direction

        return global_direction

    def transform_to(self, other: "CoordinateSystem", point: POINT3) -> POINT3:
        """
        Transform point from this coordinate system to another.

        Args:
            other: Target coordinate system
            point: Point in this coordinate system

        Returns:
            Point in target coordinate system
        """
        # Transform to global, then to target
        global_point = self.globalize(point)
        target_point = other.localize(global_point)
        return target_point

    def get_transformation_matrix(self) -> np.ndarray:
        """
        Get 4x4 homogeneous transformation matrix to global coordinates.

        Returns:
            4x4 transformation matrix
        """
        transform = np.eye(4)
        transform[:3, :3] = self.basis.T  # Rotation part
        transform[:3, 3] = self.origin  # Translation part

        # Chain with parent transformation if exists
        if self.parent is not None:
            parent_transform = self.parent.get_transformation_matrix()
            transform = parent_transform @ transform

        return transform

    @classmethod
    def from_points(
        cls,
        origin: POINT3,
        x_point: POINT3,
        xy_plane_point: POINT3,
        name: str = "custom",
    ) -> "CoordinateSystem":
        """
        Create coordinate system from three points.

        Args:
            origin: Origin point
            x_point: Point defining +X direction
            xy_plane_point: Point in XY plane (defines Y direction)
            name: Name for the coordinate system

        Returns:
            New coordinate system
        """
        origin = np.asarray(origin)
        x_point = np.asarray(x_point)
        xy_plane_point = np.asarray(xy_plane_point)

        # X-axis from origin to x_point
        e1 = normalize(x_point - origin)

        # Y-axis in XY plane, orthogonal to X
        xy_vector = xy_plane_point - origin
        e2 = xy_vector - np.dot(xy_vector, e1) * e1
        e2 = normalize(e2)

        # Z-axis from cross product
        e3 = np.cross(e1, e2)
        e3 = normalize(e3)

        basis = np.array([e1, e2, e3])

        return cls(origin=origin, basis=basis, name=name)

    @classmethod
    def from_euler_angles(
        cls,
        origin: POINT3,
        roll: float,
        pitch: float,
        yaw: float,
        name: str = "euler",
        degrees: bool = True,
    ) -> "CoordinateSystem":
        """
        Create coordinate system from Euler angles.

        Args:
            origin: Origin point
            roll: Rotation around X-axis
            pitch: Rotation around Y-axis
            yaw: Rotation around Z-axis
            name: Name for the coordinate system
            degrees: Whether angles are in degrees (True) or radians (False)

        Returns:
            New coordinate system
        """
        if degrees:
            roll = np.radians(roll)
            pitch = np.radians(pitch)
            yaw = np.radians(yaw)

        # Rotation matrices
        R_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ]
        )

        R_y = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ]
        )

        R_z = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )

        # Combined rotation (ZYX order)
        rotation = R_z @ R_y @ R_x

        return cls(origin=np.asarray(origin), basis=rotation, name=name)

    def __repr__(self) -> str:
        """String representation of coordinate system."""
        return f"CoordinateSystem(name='{self.name}', origin={self.origin}, parent={self.parent.name if self.parent else None})"


# Global coordinate system (world reference)
WORLD_COORDINATE_SYSTEM = CoordinateSystem(
    origin=np.zeros(3), basis=np.eye(3), name="world"
)


class CoordinateManager:
    """
    Manager for multiple coordinate systems with automatic tracking.

    Provides centralized management of coordinate systems and their relationships,
    similar to Optiland's coordinate system hierarchy.
    """

    def __init__(self):
        """Initialize coordinate manager."""
        self.systems: dict[str, CoordinateSystem] = {}
        self.register_system(WORLD_COORDINATE_SYSTEM)

    def register_system(self, system: CoordinateSystem) -> None:
        """Register a coordinate system."""
        self.systems[system.name] = system

    def get_system(self, name: str) -> CoordinateSystem:
        """Get coordinate system by name."""
        if name not in self.systems:
            raise ValueError(f"Coordinate system '{name}' not found")
        return self.systems[name]

    def create_child_system(
        self,
        parent_name: str,
        child_name: str,
        local_origin: POINT3,
        local_basis: Optional[MATRIX3x3] = None,
    ) -> CoordinateSystem:
        """
        Create child coordinate system relative to parent.

        Args:
            parent_name: Name of parent coordinate system
            child_name: Name for new child system
            local_origin: Origin in parent coordinates
            local_basis: Basis in parent coordinates (identity if None)

        Returns:
            New child coordinate system
        """
        parent = self.get_system(parent_name)

        if local_basis is None:
            local_basis = np.eye(3)

        child = CoordinateSystem(
            origin=local_origin, basis=local_basis, parent=parent, name=child_name
        )

        self.register_system(child)
        return child

    def transform_point(
        self, point: POINT3, from_system: str, to_system: str
    ) -> POINT3:
        """
        Transform point between coordinate systems.

        Args:
            point: Point to transform
            from_system: Source coordinate system name
            to_system: Target coordinate system name

        Returns:
            Transformed point
        """
        source = self.get_system(from_system)
        target = self.get_system(to_system)

        return source.transform_to(target, point)

    def list_systems(self) -> List[str]:
        """Get list of registered coordinate system names."""
        return list(self.systems.keys())

    def get_hierarchy(self) -> dict:
        """Get hierarchy of coordinate systems."""
        hierarchy = {}
        for name, system in self.systems.items():
            parent_name = system.parent.name if system.parent else None
            hierarchy[name] = {
                "parent": parent_name,
                "origin": system.origin.tolist(),
                "basis": system.basis.tolist(),
            }
        return hierarchy
