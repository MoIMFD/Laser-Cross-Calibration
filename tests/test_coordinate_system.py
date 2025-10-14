from __future__ import annotations

from math import isclose

import pytest

from laser_cross_calibration.coordinate_system import CoordinateSystem


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
