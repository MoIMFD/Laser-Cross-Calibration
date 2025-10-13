from __future__ import annotations

from typing import TYPE_CHECKING, Self, TypedDict

from laser_cross_calibration.tracing import OpticalRay

if TYPE_CHECKING:
    from laser_cross_calibration.types import POINT3


class CoordinateSystem:
    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        rx: float = 0.0,
        ry: float = 0.0,
        rz: float = 0.0,
        parent_cs: Self | None = None,
    ):
        self.x = x
        self.y = y
        self.z = z

        self.rx = rx
        self.ry = ry
        self.rz = rz

        self.parent_cs = parent_cs

    def localize(self, ray: OpticalRay) -> OpticalRay:
        """Transform a ray from world coordinates to the local coordinate system."""
        if self.parent_cs:
            local_ray = self.parent_cs.localize(ray)
        else:
            local_ray = ray.copy()
        local_ray.translate(-self.x, -self.y, -self.z)
        local_ray.rotate(self.rx, self.ry, self.rz)

        return local_ray

    def globalize(self, ray: OpticalRay) -> OpticalRay:
        """Transform a ray from the local coordinate system to world coordinates."""

        if self.parent_cs:
            global_ray = self.parent_cs.globalize(ray)
        else:
            global_ray = ray.copy()
        global_ray.translate(self.x, self.y, self.z)
        global_ray.rotate(self.rx, self.ry, self.rz)
        return global_ray

    @property
    def position_in_global_cs(self) -> POINT3:
        """Position in global coordinate system.

        Creates a local ray and utilizes the ray transformation methods.
        """
        local_ray = OpticalRay.ray_x()
        global_ray = self.globalize(local_ray)
        return global_ray.origin

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
