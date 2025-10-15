from __future__ import annotations

from math import isclose
from math import pi as PI
from typing import TYPE_CHECKING

import pytest

from laser_cross_calibration.constants import (
    ORIGIN_POINT3,
    UNIT_X_VECTOR3,
    UNIT_Y_VECTOR3,
    UNIT_Z_VECTOR3,
)
from laser_cross_calibration.coordinate_system import CoordinateSystem
from laser_cross_calibration.tracing import OpticalRay
from tests.utils import assert_vectors_close

if TYPE_CHECKING:
    from laser_cross_calibration.materials.constant import ConstantMaterial


@pytest.mark.unit
class TestCoordinateSystem:
    def test_create(self):
        """Test the creation"""
        parent = CoordinateSystem()
        cs = CoordinateSystem(x=1, y=2, z=3, rx=4, ry=5, rz=6, parent_cs=parent)

        assert isclose(cs.x, 1)
        assert isclose(cs.y, 2)
        assert isclose(cs.z, 3)

        assert isclose(cs.rx, 4)
        assert isclose(cs.ry, 5)
        assert isclose(cs.rz, 6)

        assert cs.parent_cs is parent

    def test_default(self):
        cs = CoordinateSystem()

        assert isclose(cs.x, 0)
        assert isclose(cs.y, 0)
        assert isclose(cs.z, 0)

        assert isclose(cs.rx, 0)
        assert isclose(cs.ry, 0)
        assert isclose(cs.rz, 0)

        assert cs.parent_cs is None

    def test_to_dict(self):
        parent1 = CoordinateSystem(x=-10)
        parent2 = CoordinateSystem(x=10, parent_cs=parent1)
        cs = CoordinateSystem(x=1, y=2, z=3, rx=4, ry=5, rz=6, parent_cs=parent2)

        dictionary = cs.to_dict()

        assert isclose(dictionary["x"], cs.x)
        assert isclose(dictionary["y"], cs.y)
        assert isclose(dictionary["z"], cs.z)

        assert isclose(dictionary["rx"], cs.rx)
        assert isclose(dictionary["ry"], cs.ry)
        assert isclose(dictionary["rz"], cs.rz)

        # extract parents
        dict_parent = dictionary["parent_cs"]
        assert isinstance(dict_parent, dict)
        dict_grandparent = dict_parent["parent_cs"]
        assert isinstance(dict_grandparent, dict)
        assert isclose(dict_parent["x"], parent2.x)
        assert isclose(dict_grandparent["x"], parent1.x)

    def test_from_dict(self):
        parent1 = CoordinateSystem(x=-10)
        parent2 = CoordinateSystem(x=10, parent_cs=parent1)
        original_cs = CoordinateSystem(
            x=1, y=2, z=3, rx=4, ry=5, rz=6, parent_cs=parent2
        )

        dictionary = original_cs.to_dict()

        rebuild_cs = CoordinateSystem.from_dict(dictionary)

        assert isclose(original_cs.x, rebuild_cs.x)
        assert isclose(original_cs.y, rebuild_cs.y)
        assert isclose(original_cs.z, rebuild_cs.z)

        assert isclose(original_cs.rx, rebuild_cs.rx)
        assert isclose(original_cs.ry, rebuild_cs.ry)
        assert isclose(original_cs.rz, rebuild_cs.rz)

        assert isinstance(rebuild_cs.parent_cs, CoordinateSystem)
        assert isinstance(original_cs.parent_cs, CoordinateSystem)
        assert isclose(original_cs.parent_cs.x, rebuild_cs.parent_cs.x)

        assert isinstance(rebuild_cs.parent_cs.parent_cs, CoordinateSystem)
        assert isinstance(original_cs.parent_cs.parent_cs, CoordinateSystem)
        assert isclose(
            original_cs.parent_cs.parent_cs.x, rebuild_cs.parent_cs.parent_cs.x
        )


@pytest.mark.integration
class TestCoordinateSystemWithRay:
    def test_localize_ray_translation(self, air: ConstantMaterial):
        """Test localization of a global ray to a shifted coordinate system."""
        global_ray = OpticalRay.ray_x()
        cs = CoordinateSystem(x=5, y=10, z=-5)

        local_ray = cs.localize(global_ray)
        assert_vectors_close(local_ray.origin, (-5, -10, 5))
        assert_vectors_close(local_ray.current_direction, UNIT_X_VECTOR3)

        global_ray.propagate(10, air)
        local_ray = cs.localize(global_ray)
        assert_vectors_close(local_ray.origin, (-5, -10, 5))
        assert_vectors_close(local_ray.current_direction, UNIT_X_VECTOR3)
        assert_vectors_close(local_ray.current_position, (5, -10, 5))
        assert local_ray.segment_distances[0] == global_ray.segment_distances[0]
        assert local_ray.media_history[0].name == global_ray.media_history[0].name

    def test_globalize_ray_translation(self, air: ConstantMaterial):
        """Test globalization of a local ray from a shifted coordinate system."""

        local_ray = OpticalRay.ray_y()
        cs = CoordinateSystem(x=-5, y=-10, z=5)

        global_ray = cs.globalize(local_ray)
        assert_vectors_close(global_ray.origin, (-5, -10, 5))
        assert_vectors_close(global_ray.current_direction, UNIT_Y_VECTOR3)

        local_ray.propagate(10, air)
        global_ray = cs.globalize(local_ray)
        assert_vectors_close(global_ray.origin, (-5, -10, 5))
        assert_vectors_close(global_ray.current_direction, UNIT_Y_VECTOR3)
        assert_vectors_close(global_ray.current_position, (-5, 0, 5))
        assert local_ray.segment_distances[0] == global_ray.segment_distances[0]
        assert local_ray.media_history[0].name == global_ray.media_history[0].name

    def test_localize_ray_rotation(self, air: ConstantMaterial):
        """test localization of rotated coordinate systems."""
        global_ray = OpticalRay.ray_x()
        cs = CoordinateSystem(ry=PI / 2)

        local_ray = cs.localize(global_ray)
        assert_vectors_close(local_ray.origin, ORIGIN_POINT3)
        assert_vectors_close(local_ray.current_direction, -UNIT_Z_VECTOR3)

        global_ray = OpticalRay.ray_x(origin=UNIT_X_VECTOR3 + UNIT_Z_VECTOR3)
        cs = CoordinateSystem(ry=PI / 2)

        local_ray = cs.localize(global_ray)
        assert_vectors_close(local_ray.origin, -UNIT_Z_VECTOR3 + UNIT_X_VECTOR3)
        assert_vectors_close(local_ray.current_direction, -UNIT_Z_VECTOR3)

        global_ray.propagate(10, air)
        local_ray = cs.localize(global_ray)
        assert_vectors_close(
            local_ray.current_position, -11 * UNIT_Z_VECTOR3 + UNIT_X_VECTOR3
        )
        assert local_ray.media_history[-1].name == global_ray.media_history[-1].name

    def test_neasted_coordinate_transform(self):
        """Test nested coordinate system transformation."""
        global_ray = OpticalRay.ray_x(
            origin=UNIT_X_VECTOR3 + UNIT_Y_VECTOR3 + UNIT_Z_VECTOR3
        )

        cs1 = CoordinateSystem(x=5)
        cs2 = CoordinateSystem(y=10, parent_cs=cs1)

        local_ray = cs2.localize(global_ray)
        assert_vectors_close(local_ray.origin, (-4, -9, 1))
        assert_vectors_close(local_ray.initial_direction, UNIT_X_VECTOR3)

        cs3 = CoordinateSystem(rz=3 * PI / 2, parent_cs=cs2)
        local_ray = cs3.localize(global_ray)
        assert_vectors_close(local_ray.origin, (-9, 4, 1))
        assert_vectors_close(local_ray.initial_direction, -UNIT_Y_VECTOR3)
