from __future__ import annotations

from typing import TYPE_CHECKING, Self, TypedDict

from laser_cross_calibration.tracing import OpticalRay

if TYPE_CHECKING:
    from laser_cross_calibration.tracing import OpticalRay


class CoordinateSystem:
    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        rx: float = 0.0,
        ry: float = 0.0,
        rz: float = 0.0,
        parent_cs: CoordinateSystem | None = None,
    ):
        self.x = x
        self.y = y
        self.z = z

        self.rx = rx
        self.ry = ry
        self.rz = rz

        self.parent_cs = parent_cs

    def localize(self, rays):
        if self.parent_cs:
            self.parent_cs.localize(rays)

        rays.translate(-self.x, -self.y, -self.z)
        if self.rz:
            rays.rotate_z(-self.rz)
        if self.ry:
            rays.rotate_y(-self.ry)
        if self.rx:
            rays.rotate_x(-self.rx)

    def globalize(self):
        raise NotImplementedError()

    def position_in_global_cs(self):
        raise NotImplementedError()

    def get_rotation_matrix(self):
        raise NotImplementedError()

    def get_effective_transform(self):
        raise NotImplementedError()

    def get_effective_rotation_euler(self):
        raise NotImplementedError()

    def to_dict(self) -> CoordinateSystemDict:
        """Create a dictionary description of a CoordinateSystem instance from a dict."""

        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "rx": self.rx,
            "ry": self.ry,
            "rz": self.rz,
            "parent_cs": self.parent_cs.to_dict() if self.parent_cs else None,
        }

    @classmethod
    def from_dict(cls, dictionary: CoordinateSystemDict) -> CoordinateSystem:
        """Create a CoordinateSystem instance from a dict."""
        parent_dict = dictionary.get("parent_cs", None)
        return CoordinateSystem(
            x=dictionary.get("x", 0.0),
            y=dictionary.get("y", 0.0),
            z=dictionary.get("z", 0.0),
            rx=dictionary.get("rx", 0.0),
            ry=dictionary.get("ry", 0.0),
            rz=dictionary.get("rz", 0.0),
            parent_cs=cls.from_dict(parent_dict) if parent_dict else None,
        )


class CoordinateSystemDict(TypedDict):
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    parent_cs: CoordinateSystemDict | None
