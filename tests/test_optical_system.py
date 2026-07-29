from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from laser_cross_calibration.surfaces import Plane
from laser_cross_calibration.tracing import OpticalInterface, OpticalSystem

if TYPE_CHECKING:
    from hazy import Frame


@pytest.mark.unit
class TestOpticalSystem:
    def test_creation(self):
        system = OpticalSystem()

        assert system.interfaces == []

    def test_add_interface(self, air, water, frame: Frame):
        system = OpticalSystem()
        assert system.interfaces == []

        geometries = [
            Plane.create_xy(frame=frame),
            Plane.create_xz(frame=frame),
            Plane.create_yz(frame=frame),
        ]
        interfaces = [
            OpticalInterface(geometry=geometry, material_pre=air, material_post=water)
            for geometry in geometries
        ]

        for i, interface in enumerate(interfaces):
            system.add_interface(interface)

            assert system.interfaces[i] == interface
