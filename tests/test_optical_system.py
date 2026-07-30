from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from laser_cross_calibration.surfaces import Plane
from laser_cross_calibration.tracing import OpticalInterface, OpticalRay, OpticalSystem

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

    def test_trace_ray_does_not_duplicate_final_segment(
        self, air, glass_bk7, water, frame: Frame
    ):
        """A ray that runs out of interfaces must travel its final leg once.

        Regression test: trace_ray propagated final_propagation_distance both
        inside the `interface is None` branch and again, unconditionally,
        after the loop -- doubling the last leg of every ray that exits the
        system normally and duplicating the last media_history entry. Easy to
        miss when the duplicated medium is the same as the one before it
        (e.g. "water -> water"), but it happens for every ray, regardless of
        geometry.
        """
        final_propagation_distance = 10.0
        system = OpticalSystem(final_propagation_distance=final_propagation_distance)
        system.add_interface(
            OpticalInterface(
                geometry=Plane(point=frame.origin, normal=-frame.y_axis),
                material_pre=air,
                material_post=glass_bk7,
            )
        )
        system.add_interface(
            OpticalInterface(
                geometry=Plane(
                    point=frame.origin + frame.y_axis * 0.05, normal=frame.y_axis
                ),
                material_pre=glass_bk7,
                material_post=water,
            )
        )

        ray = OpticalRay(
            origin=frame.origin - frame.y_axis * 0.1, direction=frame.y_axis
        )
        traced = system.trace_ray(ray)

        # air segment + glass segment + one final tail segment in water
        assert traced.segment_distances == [
            pytest.approx(0.1),
            pytest.approx(0.05),
            final_propagation_distance,
        ]
        assert [medium.name for medium in traced.media_history] == [
            "air",
            "BK7 glass",
            "water",
        ]
