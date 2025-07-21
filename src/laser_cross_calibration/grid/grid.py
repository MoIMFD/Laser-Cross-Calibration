from dataclasses import dataclass, field

import numpy as np
import numpy.typing as nptyping

import pint

from ..types import POINT3, VECTOR3, MATRIX3x3


ureg = pint.UnitRegistry()

DEFAULT_UNIT = ureg.mm


def ensure_unit(value, unit=None):
    if hasattr(value, "units"):
        return value
    else:
        return value * (unit or DEFAULT_UNIT)


class StageOutOfLimitsError(ValueError):
    def __init__(self, point, limits, message=None):
        self.point = point
        self.limits = limits

        if message is None:
            message = f"Point {point} is outside of limits {limits}"

        super().__init__(message)


@dataclass
class OrthogonalCoordinateSystem:
    origin: VECTOR3 = field(default_factory=lambda: np.zeros(3) * ureg.mm)
    base: MATRIX3x3 = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        self.base = np.array(self.base).astype(float)
        assert np.allclose(np.linalg.norm(self.base, axis=1), 1.0), (
            "Base must be build from unit vectors with lengt 1"
        )
        assert np.allclose(np.cross(self.e1, self.e2), self.e3), (
            "Base must be orthogonal"
        )

    @property
    def e1(self) -> VECTOR3:
        return self.base[0]

    @property
    def e2(self) -> VECTOR3:
        return self.base[1]

    @property
    def e3(self) -> VECTOR3:
        return self.base[2]

    @classmethod
    def from_unnormed_base(cls, base):
        base = np.asarray(base)
        base[0] = base[0] / np.linalg.norm(base[0])
        base[1] = base[1] / np.linalg.norm(base[1])
        base[2] = base[2] / np.linalg.norm(base[2])
        return cls(base)


@dataclass
class CalibrationPoint:
    intersection_position: POINT3
    stage_position: POINT3


@dataclass
class Stage:
    origin_world: OrthogonalCoordinateSystem = field(
        default_factory=OrthogonalCoordinateSystem
    )
    position_local: POINT3 = field(
        default_factory=lambda: np.zeros(3, dtype=float) * ureg.mm
    )
    arm1: VECTOR3 = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float64) * ureg.mm
    )
    arm2: VECTOR3 = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float64) * ureg.mm
    )
    limits: tuple[VECTOR3, VECTOR3] = (
        np.array([0.0, 0.0, 0.0]) * ureg.mm,
        np.array([30.0, 30.0, 30.0]) * ureg.mm,
    )

    @property
    def arm1_local_position(self) -> POINT3:
        return self.position_local + self.arm1

    @property
    def arm2_local_position(self) -> POINT3:
        return self.position_local + self.arm1

    def _is_within_local_limits(self, point: POINT3) -> bool:
        return all((self.limits[0] <= point) & (point <= self.limits[1]))

    def _validate_local_position(self, point: POINT3) -> POINT3:
        if self._is_within_local_limits(point=point):
            return point
        else:
            raise StageOutOfLimitsError(point, self.limits)

    def set_local_position(self, position: POINT3, unit: pint.Unit = DEFAULT_UNIT):
        position = ensure_unit(value=position, unit=unit)
        self._validate_local_position(position)
        self.position_local = position
        return self
