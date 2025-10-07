from __future__ import annotations

from scipy.spatial.transform import Rotation as R

import numpy as np


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

    def to_dict(self) -> dict:
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, dictionary: dict):
        raise NotImplementedError()
