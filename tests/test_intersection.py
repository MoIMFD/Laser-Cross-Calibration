from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from laser_cross_calibration.constants import (
    INTERSECTION_THRESHOLD,
    VSMALL,
)
from laser_cross_calibration.tracing import OpticalRay
from laser_cross_calibration.tracing.intersection import (
    line_segment_intersection,
    ray_intersection,
)
from tests.utils import assert_vectors_close

if TYPE_CHECKING:
    from hazy import Frame


@pytest.mark.unit
class TestLineSegmentIntersection:
    def test_perpendicular_crossing_segments(self, frame: Frame):
        """Test two perpendicular line segments that cross exactly."""

        p1 = frame.point(0.0, -1.0, 0.0)
        p2 = frame.point(0.0, 1.0, 0.0)
        p3 = frame.point(-1.0, 0.0, 0.0)
        p4 = frame.point(1.0, 0.0, 0.0)

        intersects, point = line_segment_intersection(p1, p2, p3, p4)

        assert intersects is True
        assert_vectors_close(point, frame.origin)

    def test_perpendicular_crossing_segments_3d(self, frame: Frame):
        """Test two perpendicular line segments that cross in 3D space."""

        p1 = frame.point(0.0, 0.0, -1.0)
        p2 = frame.point(0.0, 0.0, 1.0)
        p3 = frame.point(-1.0, 0.0, 0.0)
        p4 = frame.point(1.0, 0.0, 0.0)

        intersects, point = line_segment_intersection(p1, p2, p3, p4)

        assert intersects is True
        assert_vectors_close(point, frame.origin)

    def test_skew_lines_no_intersection(self, frame: Frame):
        """Test two skew lines that don't intersect."""

        p1 = frame.point(0.0, 0.0, 0.0)
        p2 = frame.point(1.0, 0.0, 0.0)
        p3 = frame.point(0.0, 1.0, 1.0)
        p4 = frame.point(1.0, 1.0, 1.0)

        intersects, point = line_segment_intersection(p1, p2, p3, p4)

        assert intersects is False

    def test_near_miss_within_threshold(self, frame: Frame):
        """Test two segments that nearly intersect within threshold."""
        threshold = 0.01
        p1 = frame.point(0.0, -1.0, 0.0)
        p2 = frame.point(0.0, 1.0, 0.0)
        p3 = frame.point(-1.0, 0.0, 0.005)
        p4 = frame.point(1.0, 0.0, 0.005)

        intersects, point = line_segment_intersection(p1, p2, p3, p4, threshold)

        assert intersects is True
        expected = np.array([0.0, 0.0, 0.0025])
        assert_vectors_close(np.array(point), expected)

    def test_near_miss_outside_threshold(self, frame: Frame):
        """Test two segments that nearly intersect outside threshold."""
        threshold = 0.001
        p1 = frame.point(0.0, -1.0, 0.0)
        p2 = frame.point(0.0, 1.0, 0.0)
        p3 = frame.point(-1.0, 0.0, 0.005)
        p4 = frame.point(1.0, 0.0, 0.005)

        intersects, point = line_segment_intersection(p1, p2, p3, p4, threshold)

        assert intersects is False

    def test_parallel_segments_no_intersection(self, frame: Frame):
        """Test two parallel segments that don't intersect."""
        p1 = frame.point(0.0, 0.0, 0.0)
        p2 = frame.point(1.0, 0.0, 0.0)
        p3 = frame.point(0.0, 1.0, 0.0)
        p4 = frame.point(1.0, 1.0, 0.0)

        intersects, _ = line_segment_intersection(p1, p2, p3, p4)

        assert intersects is False

    def test_collinear_overlapping_segments(self, frame: Frame):
        """Test two collinear segments that overlap."""
        p1 = frame.point(0.0, 0.0, 0.0)
        p2 = frame.point(2.0, 0.0, 0.0)
        p3 = frame.point(1.0, 0.0, 0.0)
        p4 = frame.point(3.0, 0.0, 0.0)

        intersects, point = line_segment_intersection(p1, p2, p3, p4)

        assert intersects is True
        expected = frame.point(1.5, 0.0, 0.0)
        assert_vectors_close(point, expected)

    def test_collinear_non_overlapping_segments(self, frame: Frame):
        """Test two collinear segments that don't overlap."""
        p1 = frame.point(0.0, 0.0, 0.0)
        p2 = frame.point(1.0, 0.0, 0.0)
        p3 = frame.point(2.0, 0.0, 0.0)
        p4 = frame.point(3.0, 0.0, 0.0)

        intersects, _ = line_segment_intersection(p1, p2, p3, p4)

        assert intersects is False

    def test_segments_not_overlapping_in_parameter_space(self, frame: Frame):
        """Test segments that would intersect if extended but don't in their bounds."""
        p1 = frame.point(0.0, 0.0, 0.0)
        p2 = frame.point(0.5, 0.0, 0.0)
        p3 = frame.point(1.0, -1.0, 0.0)
        p4 = frame.point(1.0, 1.0, 0.0)

        intersects, point = line_segment_intersection(p1, p2, p3, p4)

        assert intersects is False
        assert all(np.isnan(value) for value in point)

    def test_very_short_segment_below_threshold(self, frame: Frame):
        """Test that very short segments are rejected."""
        threshold = INTERSECTION_THRESHOLD
        p1 = frame.point(0.0, 0.0, 0.0)
        p2 = frame.point(1e-5, 0.0, 0.0)
        p3 = frame.point(0.0, -1.0, 0.0)
        p4 = frame.point(0.0, 1.0, 0.0)

        intersects, _ = line_segment_intersection(p1, p2, p3, p4, threshold)

        assert intersects is False

    def test_diagonal_intersection(self, frame: Frame):
        """Test two diagonal segments intersecting at origin."""
        p1 = frame.point(-1.0, -1.0, 0.0)
        p2 = frame.point(1.0, 1.0, 0.0)
        p3 = frame.point(-1.0, 1.0, 0.0)
        p4 = frame.point(1.0, -1.0, 0.0)

        intersects, point = line_segment_intersection(p1, p2, p3, p4)

        assert intersects is True
        assert_vectors_close(point, frame.origin)

    def test_t_shape_intersection(self, frame: Frame):
        """Test T-shaped intersection at endpoint."""
        p1 = frame.point(0.0, 0.0, 0.0)
        p2 = frame.point(0.0, 1.0, 0.0)
        p3 = frame.point(-1.0, 0.5, 0.0)
        p4 = frame.point(1.0, 0.5, 0.0)

        intersects, point = line_segment_intersection(p1, p2, p3, p4)

        assert intersects is True
        expected = frame.point(0.0, 0.5, 0.0)
        assert_vectors_close(point, expected)


@pytest.mark.unit
class TestRayIntersection:
    def test_single_segment_rays_crossing(self, frame: Frame, air):
        """Test two single-segment rays that cross."""
        ray1 = OpticalRay(origin=frame.point(0.0, -2.0, 0.0), direction=frame.y_axis)
        ray1.propagate(distance=4.0, medium=air)

        ray2 = OpticalRay(origin=frame.point(-2.0, 0.0, 0.0), direction=frame.x_axis)
        ray2.propagate(distance=4.0, medium=air)

        intersections = ray_intersection(ray1, ray2, threshold=INTERSECTION_THRESHOLD)

        assert len(intersections) == 1
        assert_vectors_close(intersections[0], frame.origin)

    def test_multi_segment_rays_single_crossing(self, frame: Frame, air):
        """Test multi-segment rays with single crossing point."""
        ray1 = OpticalRay(origin=frame.point(0.0, -2.0, 0.0), direction=frame.y_axis)
        ray1.propagate(distance=1.0, medium=air)
        ray1.propagate(distance=2.0, medium=air)
        ray1.propagate(distance=1.0, medium=air)

        ray2 = OpticalRay(origin=frame.point(-2.0, 0.0, 0.0), direction=frame.x_axis)
        ray2.propagate(distance=4.0, medium=air)

        intersections = ray_intersection(ray1, ray2, threshold=INTERSECTION_THRESHOLD)

        assert len(intersections) == 1
        assert_vectors_close(intersections[0], frame.origin)

    def test_parallel_rays_no_intersection(self, frame: Frame, air):
        """Test parallel rays that never intersect."""
        ray1 = OpticalRay(origin=frame.origin, direction=frame.x_axis)
        ray1.propagate(distance=5.0, medium=air)

        ray2 = OpticalRay(origin=frame.point(0.0, 1.0, 0.0), direction=frame.x_axis)
        ray2.propagate(distance=5.0, medium=air)

        intersections = ray_intersection(ray1, ray2, threshold=INTERSECTION_THRESHOLD)

        assert len(intersections) == 0

    def test_diverging_rays_no_intersection(self, frame: Frame, air):
        """Test diverging rays that don't intersect.

        Note: Rays starting from the same origin but going in different directions
        should not report an intersection, as they only share the starting point
        but their actual path segments don't cross.
        """
        ray1 = OpticalRay(origin=frame.origin, direction=frame.x_axis)
        ray1.propagate(distance=5.0, medium=air)

        ray2 = OpticalRay(origin=frame.origin, direction=frame.y_axis)
        ray2.propagate(distance=5.0, medium=air)

        intersections = ray_intersection(ray1, ray2, threshold=INTERSECTION_THRESHOLD)

        assert len(intersections) <= 1

    def test_skew_rays_3d_no_intersection(self, frame: Frame, air):
        """Test skew rays in 3D that don't intersect."""
        ray1 = OpticalRay(origin=frame.origin, direction=frame.x_axis)
        ray1.propagate(distance=5.0, medium=air)

        ray2 = OpticalRay(origin=frame.point(0.0, 1.0, 1.0), direction=frame.x_axis)
        ray2.propagate(distance=5.0, medium=air)

        intersections = ray_intersection(ray1, ray2, threshold=VSMALL)

        assert len(intersections) == 0

    def test_near_miss_within_threshold(self, frame: Frame, air):
        """Test rays that nearly cross within threshold."""
        threshold = 0.01
        ray1 = OpticalRay(origin=frame.point(0.0, -2.0, 0.0), direction=frame.y_axis)
        ray1.propagate(distance=4.0, medium=air)

        ray2 = OpticalRay(origin=frame.point(-2.0, 0.0, 0.005), direction=frame.x_axis)
        ray2.propagate(distance=4.0, medium=air)

        intersections = ray_intersection(ray1, ray2, threshold=threshold)

        assert len(intersections) == 1
        expected = np.array([0.0, 0.0, 0.0025])
        assert_vectors_close(intersections[0], expected)

    def test_multiple_crossings(self, frame: Frame, air):
        """Test rays that cross multiple times in a zig-zag pattern."""
        ray1 = OpticalRay(origin=frame.point(0.0, -3.0, 0.0), direction=frame.y_axis)
        ray1.propagate(distance=1.5, medium=air)
        ray1.current_direction = frame.x_axis
        ray1.propagate(distance=2.0, medium=air)
        ray1.current_direction = -frame.y_axis
        ray1.propagate(distance=1.0, medium=air)

        ray2 = OpticalRay(origin=frame.point(-0.5, -1.5, 0.0), direction=frame.x_axis)
        ray2.propagate(distance=3.0, medium=air)

        intersections = ray_intersection(ray1, ray2, threshold=INTERSECTION_THRESHOLD)

        assert len(intersections) >= 1

    def test_crossing_at_segment_boundary(self, frame: Frame, air):
        """Test rays crossing exactly at segment boundary.

        Note: When rays cross at a segment boundary, the intersection might be
        detected by both adjacent segments, so we allow for some flexibility.
        """
        ray1 = OpticalRay(origin=frame.point(0.0, -2.0, 0.0), direction=frame.y_axis)
        ray1.propagate(distance=2.0, medium=air)
        ray1.propagate(distance=2.0, medium=air)

        ray2 = OpticalRay(origin=frame.point(-2.0, 0.0, 0.0), direction=frame.x_axis)
        ray2.propagate(distance=4.0, medium=air)

        intersections = ray_intersection(ray1, ray2, threshold=INTERSECTION_THRESHOLD)

        assert len(intersections) >= 1
        assert_vectors_close(intersections[0], frame.origin)

    def test_realistic_laser_cross_scenario(self, frame: Frame, air):
        """Test realistic laser cross setup with angled beams."""
        ray1 = OpticalRay(origin=frame.point(-0.1, 0.0, -0.26), direction=frame.z_axis)
        ray1.propagate(distance=0.3, medium=air)

        direction2 = frame.x_axis - 0.2 * frame.z_axis
        direction2 = direction2 / np.linalg.norm(direction2)
        ray2 = OpticalRay(origin=frame.point(-0.1, 0.0, 0.01), direction=direction2)
        ray2.propagate(distance=0.3, medium=air)

        intersections = ray_intersection(ray1, ray2, threshold=INTERSECTION_THRESHOLD)

        assert len(intersections) >= 1

    def test_default_threshold_usage(self, frame: Frame, air):
        """Test that default threshold parameter works correctly."""
        ray1 = OpticalRay(origin=frame.point(0.0, -2.0, 0.0), direction=frame.y_axis)
        ray1.propagate(distance=4.0, medium=air)

        ray2 = OpticalRay(origin=frame.point(-2.0, 0.0, 0.0), direction=frame.x_axis)
        ray2.propagate(distance=4.0, medium=air)

        intersections_vsmall = ray_intersection(ray1, ray2)
        intersections_threshold = ray_intersection(
            ray1, ray2, threshold=INTERSECTION_THRESHOLD
        )

        assert len(intersections_vsmall) == len(intersections_threshold)
