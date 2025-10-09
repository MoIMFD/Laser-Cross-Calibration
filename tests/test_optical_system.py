from __future__ import annotations

import pytest

from laser_cross_calibration.surfaces import Plane
from laser_cross_calibration.tracing import OpticalInterface, OpticalSystem


@pytest.mark.unit
class TestOpticalSystem:
    def test_creation(self):
        system = OpticalSystem()

        assert system.interfaces == []

    def test_add_interface(self, air, water):
        system = OpticalSystem()
        assert system.interfaces == []

        geometries = [Plane.create_xy(), Plane.create_xz(), Plane.create_yz()]
        interfaces = [
            OpticalInterface(geometry=geometry, material_pre=air, material_post=water)
            for geometry in geometries
        ]

        for i, interface in enumerate(interfaces):
            system.add_interface(interface)

            assert system.interfaces[i] == interface
