"""Regression test for Optimizer.find_source_origin frame handling.

find_source_origin optimizes a stage origin expressed in source.origin.frame
(see the `objective` closure, which builds candidates via
`source.origin.frame.point(new_origin)`). It used to return that raw,
numerically-correct array re-tagged with `target`'s original frame instead
of `source.origin.frame` -- a mislabeling, not a transform. This was
invisible whenever target and source happened to share a frame (e.g. both
already in the world frame, as in the simple Gunady setup), but produced a
Point with the right numbers under the wrong frame as soon as the stage sits
in a rotated/translated frame of its own, which is the normal case for a
real, physically-mounted dual-laser stage.
"""

from __future__ import annotations

from math import cos, radians, sin
from typing import TYPE_CHECKING

import pytest

from laser_cross_calibration import materials, surfaces, tracing
from laser_cross_calibration.optimization import Optimizer
from laser_cross_calibration.sources import DualLaserStageSource
from tests.utils import assert_vectors_close

if TYPE_CHECKING:
    from hazy import Frame, Point


@pytest.fixture
def rotated_stage_frame(frame: Frame) -> Frame:
    """A stage mount that is rotated and offset relative to the world frame.

    Mirrors a real physically-mounted dual-laser stage, which is never
    conveniently axis-aligned with the world/optical-system frame.
    """
    return (
        frame.make_child(name="stage")
        .rotate_euler(z=30.0, degrees=True)
        .translate(x=0.2, y=-0.3)
    )


@pytest.fixture
def flat_plate_system(frame: Frame) -> tracing.OpticalSystem:
    """Air -> glass -> water, same setup as the Gunady fixtures."""
    system = tracing.OpticalSystem(final_propagation_distance=10)
    system.add_interface(
        tracing.OpticalInterface(
            geometry=surfaces.Plane(point=frame.origin, normal=-frame.y_axis),
            material_pre=materials.AIR,
            material_post=materials.GLASS_FUSED_SILICA,
        )
    )
    system.add_interface(
        tracing.OpticalInterface(
            geometry=surfaces.Plane(
                point=frame.origin + frame.y_axis * 0.05, normal=frame.y_axis
            ),
            material_pre=materials.GLASS_FUSED_SILICA,
            material_post=materials.WATER,
        )
    )
    return system


def _make_source(stage_frame: Frame) -> DualLaserStageSource:
    angle_1 = radians(11.5)
    angle_2 = radians(12.6)
    return DualLaserStageSource(
        origin=stage_frame.origin - stage_frame.y_axis * 0.3,
        arm1=stage_frame.x_axis * 0.1,
        arm2=-stage_frame.x_axis * 0.1,
        direction1=stage_frame.vector(-sin(angle_1), cos(angle_1), 0.0),
        direction2=stage_frame.vector(sin(angle_2), cos(angle_2), 0.0),
    )


class _FixedEstimator:
    """Estimator stub returning a known-good initial guess, regardless of target.

    find_source_origin's frame handling is what's under test here, not the
    quality of the ML estimator, so a trivial stand-in is enough.
    """

    def __init__(self, guess: Point):
        self._guess = guess

    def __call__(self, point: Point) -> Point:  # noqa: ARG002
        return self._guess


@pytest.mark.unit
def test_find_source_origin_result_usable_with_set_origin(
    flat_plate_system: tracing.OpticalSystem, rotated_stage_frame: Frame
):
    """result.x must be directly settable on the source without a frame error."""
    source = _make_source(rotated_stage_frame)
    tracer = tracing.RayTracer(optical_system=flat_plate_system)

    known_good_origin = source.origin.copy()
    optimizer = Optimizer(tracer=tracer, estimator=_FixedEstimator(known_good_origin))

    _, intersections = tracer.trace_and_find_crossings(sources=[source])
    assert len(intersections) == 1

    # Target deliberately expressed in the world frame, not source.origin.frame,
    # to reproduce the mismatch a real caller hits (targets are usually defined
    # in world/optical-system coordinates, stages are not).
    target_world = intersections[0].to_frame(rotated_stage_frame.parent)

    result = optimizer.find_source_origin(target=target_world, source=source)

    source.set_origin(result.x)


@pytest.mark.unit
def test_find_source_origin_reproduces_target(
    flat_plate_system: tracing.OpticalSystem, rotated_stage_frame: Frame
):
    """The stage origin found must actually reproduce the requested target."""
    source = _make_source(rotated_stage_frame)
    tracer = tracing.RayTracer(optical_system=flat_plate_system)

    known_good_origin = source.origin.copy()
    optimizer = Optimizer(tracer=tracer, estimator=_FixedEstimator(known_good_origin))

    _, intersections = tracer.trace_and_find_crossings(sources=[source])
    target_world = intersections[0].to_frame(rotated_stage_frame.parent)

    # Nudge the true answer away from the initial guess so the optimizer has
    # to move the stage, rather than trivially confirming its starting point.
    target_world = target_world + rotated_stage_frame.parent.vector(0.002, 0.0, 0.0)

    result = optimizer.find_source_origin(target=target_world, source=source)

    source.set_origin(result.x)
    _, verify_intersections = tracer.trace_and_find_crossings(sources=[source])

    assert len(verify_intersections) == 1
    assert_vectors_close(
        verify_intersections[0].to_frame(rotated_stage_frame.parent),
        target_world,
        atol=1e-5,
    )
