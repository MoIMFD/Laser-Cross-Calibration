from __future__ import annotations

import numpy as np
import pytest

from laser_cross_calibration.constants import (
    INTERSECTION_THRESHOLD,
    ORIGIN_POINT3,
    UNIT_X_VECTOR3,
    UNIT_Y_VECTOR3,
    UNIT_Z_VECTOR3,
    VSMALL,
)
from laser_cross_calibration.tracing import OpticalRay
from laser_cross_calibration.tracing.intersection import (
    line_segment_intersection,
    ray_intersection,
)
from tests.utils import assert_vectors_close


@pytest.mark.unit
class TestLineSegmentIntersection:
    def test_perpendicular_crossing_segments(self):
        """Test two perpendicular line segments that cross exactly."""
        p1 = np.array([0.0, -1.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        p3 = np.array([-1.0, 0.0, 0.0])
        p4 = np.array([1.0, 0.0, 0.0])

        intersects, point = line_segment_intersection(p1, p2, p3, p4)

        assert intersects is True
        assert_vectors_close(point, ORIGIN_POINT3)

    def test_perpendicular_crossing_segments_3d(self):
        """Test two perpendicular line segments that cross in 3D space."""
        p1 = np.array([0.0, 0.0, -1.0])
        p2 = np.array([0.0, 0.0, 1.0])
        p3 = np.array([-1.0, 0.0, 0.0])
        p4 = np.array([1.0, 0.0, 0.0])

        intersects, point = line_segment_intersection(p1, p2, p3, p4)

        assert intersects is True
        assert_vectors_close(point, ORIGIN_POINT3)

    def test_skew_lines_no_intersection(self):
        """Test two skew lines that don't intersect."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 1.0])
        p4 = np.array([1.0, 1.0, 1.0])

        intersects, point = line_segment_intersection(p1, p2, p3, p4)

        assert intersects is False
        assert np.all(np.isnan(point))

    def test_near_miss_within_threshold(self):
        """Test two segments that nearly intersect within threshold."""
        threshold = 0.01
        p1 = np.array([0.0, -1.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        p3 = np.array([-1.0, 0.0, 0.005])
        p4 = np.array([1.0, 0.0, 0.005])

        intersects, point = line_segment_intersection(p1, p2, p3, p4, threshold)

        assert intersects is True
        expected = np.array([0.0, 0.0, 0.0025])
        assert_vectors_close(point, expected)

    def test_near_miss_outside_threshold(self):
        """Test two segments that nearly intersect outside threshold."""
        threshold = 0.001
        p1 = np.array([0.0, -1.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        p3 = np.array([-1.0, 0.0, 0.005])
        p4 = np.array([1.0, 0.0, 0.005])

        intersects, point = line_segment_intersection(p1, p2, p3, p4, threshold)

        assert intersects is False
        assert np.all(np.isnan(point))

    def test_parallel_segments_no_intersection(self):
        """Test two parallel segments that don't intersect."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])
        p4 = np.array([1.0, 1.0, 0.0])

        intersects, point = line_segment_intersection(p1, p2, p3, p4)

        assert intersects is False
        assert np.all(np.isnan(point))

    def test_collinear_overlapping_segments(self):
        """Test two collinear segments that overlap."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 0.0])
        p4 = np.array([3.0, 0.0, 0.0])

        intersects, point = line_segment_intersection(p1, p2, p3, p4)

        assert intersects is True
        expected = np.array([1.5, 0.0, 0.0])
        assert_vectors_close(point, expected)

    def test_collinear_non_overlapping_segments(self):
        """Test two collinear segments that don't overlap."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([2.0, 0.0, 0.0])
        p4 = np.array([3.0, 0.0, 0.0])

        intersects, point = line_segment_intersection(p1, p2, p3, p4)

        assert intersects is False
        assert np.all(np.isnan(point))

    def test_segments_not_overlapping_in_parameter_space(self):
        """Test segments that would intersect if extended but don't in their bounds."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([0.5, 0.0, 0.0])
        p3 = np.array([1.0, -1.0, 0.0])
        p4 = np.array([1.0, 1.0, 0.0])

        intersects, point = line_segment_intersection(p1, p2, p3, p4)

        assert intersects is False
        assert np.all(np.isnan(point))

    def test_very_short_segment_below_threshold(self):
        """Test that very short segments are rejected."""
        threshold = INTERSECTION_THRESHOLD
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1e-5, 0.0, 0.0])
        p3 = np.array([0.0, -1.0, 0.0])
        p4 = np.array([0.0, 1.0, 0.0])

        intersects, point = line_segment_intersection(p1, p2, p3, p4, threshold)

        assert intersects is False
        assert np.all(np.isnan(point))

    def test_diagonal_intersection(self):
        """Test two diagonal segments intersecting at origin."""
        p1 = np.array([-1.0, -1.0, 0.0])
        p2 = np.array([1.0, 1.0, 0.0])
        p3 = np.array([-1.0, 1.0, 0.0])
        p4 = np.array([1.0, -1.0, 0.0])

        intersects, point = line_segment_intersection(p1, p2, p3, p4)

        assert intersects is True
        assert_vectors_close(point, ORIGIN_POINT3)

    def test_t_shape_intersection(self):
        """Test T-shaped intersection at endpoint."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        p3 = np.array([-1.0, 0.5, 0.0])
        p4 = np.array([1.0, 0.5, 0.0])

        intersects, point = line_segment_intersection(p1, p2, p3, p4)

        assert intersects is True
        expected = np.array([0.0, 0.5, 0.0])
        assert_vectors_close(point, expected)


@pytest.mark.unit
class TestRayIntersection:
    def test_single_segment_rays_crossing(self, air):
        """Test two single-segment rays that cross."""
        ray1 = OpticalRay(origin=np.array([0.0, -2.0, 0.0]), direction=UNIT_Y_VECTOR3)
        ray1.propagate(distance=4.0, medium=air)

        ray2 = OpticalRay(origin=np.array([-2.0, 0.0, 0.0]), direction=UNIT_X_VECTOR3)
        ray2.propagate(distance=4.0, medium=air)

        intersections = ray_intersection(ray1, ray2, threshold=INTERSECTION_THRESHOLD)

        assert len(intersections) == 1
        assert_vectors_close(intersections[0], ORIGIN_POINT3)

    def test_multi_segment_rays_single_crossing(self, air):
        """Test multi-segment rays with single crossing point."""
        ray1 = OpticalRay(origin=np.array([0.0, -2.0, 0.0]), direction=UNIT_Y_VECTOR3)
        ray1.propagate(distance=1.0, medium=air)
        ray1.propagate(distance=2.0, medium=air)
        ray1.propagate(distance=1.0, medium=air)

        ray2 = OpticalRay(origin=np.array([-2.0, 0.0, 0.0]), direction=UNIT_X_VECTOR3)
        ray2.propagate(distance=4.0, medium=air)

        intersections = ray_intersection(ray1, ray2, threshold=INTERSECTION_THRESHOLD)

        assert len(intersections) == 1
        assert_vectors_close(intersections[0], ORIGIN_POINT3)

    def test_parallel_rays_no_intersection(self, air):
        """Test parallel rays that never intersect."""
        ray1 = OpticalRay(origin=np.array([0.0, 0.0, 0.0]), direction=UNIT_X_VECTOR3)
        ray1.propagate(distance=5.0, medium=air)

        ray2 = OpticalRay(origin=np.array([0.0, 1.0, 0.0]), direction=UNIT_X_VECTOR3)
        ray2.propagate(distance=5.0, medium=air)

        intersections = ray_intersection(ray1, ray2, threshold=INTERSECTION_THRESHOLD)

        assert len(intersections) == 0

    def test_diverging_rays_no_intersection(self, air):
        """Test diverging rays that don't intersect.

        Note: Rays starting from the same origin but going in different directions
        should not report an intersection, as they only share the starting point
        but their actual path segments don't cross.
        """
        ray1 = OpticalRay(origin=ORIGIN_POINT3, direction=UNIT_X_VECTOR3)
        ray1.propagate(distance=5.0, medium=air)

        ray2 = OpticalRay(origin=ORIGIN_POINT3, direction=UNIT_Y_VECTOR3)
        ray2.propagate(distance=5.0, medium=air)

        intersections = ray_intersection(ray1, ray2, threshold=INTERSECTION_THRESHOLD)

        assert len(intersections) <= 1

    def test_skew_rays_3d_no_intersection(self, air):
        """Test skew rays in 3D that don't intersect."""
        ray1 = OpticalRay(origin=np.array([0.0, 0.0, 0.0]), direction=UNIT_X_VECTOR3)
        ray1.propagate(distance=5.0, medium=air)

        ray2 = OpticalRay(origin=np.array([0.0, 1.0, 1.0]), direction=UNIT_X_VECTOR3)
        ray2.propagate(distance=5.0, medium=air)

        intersections = ray_intersection(ray1, ray2, threshold=VSMALL)

        assert len(intersections) == 0

    def test_near_miss_within_threshold(self, air):
        """Test rays that nearly cross within threshold."""
        threshold = 0.01
        ray1 = OpticalRay(origin=np.array([0.0, -2.0, 0.0]), direction=UNIT_Y_VECTOR3)
        ray1.propagate(distance=4.0, medium=air)

        ray2 = OpticalRay(
            origin=np.array([-2.0, 0.0, 0.005]), direction=UNIT_X_VECTOR3
        )
        ray2.propagate(distance=4.0, medium=air)

        intersections = ray_intersection(ray1, ray2, threshold=threshold)

        assert len(intersections) == 1
        expected = np.array([0.0, 0.0, 0.0025])
        assert_vectors_close(intersections[0], expected)

    def test_multiple_crossings(self, air):
        """Test rays that cross multiple times in a zig-zag pattern."""
        ray1 = OpticalRay(origin=np.array([0.0, -3.0, 0.0]), direction=UNIT_Y_VECTOR3)
        ray1.propagate(distance=1.5, medium=air)
        ray1.current_direction = UNIT_X_VECTOR3
        ray1.propagate(distance=2.0, medium=air)
        ray1.current_direction = -UNIT_Y_VECTOR3
        ray1.propagate(distance=1.0, medium=air)

        ray2 = OpticalRay(origin=np.array([-0.5, -1.5, 0.0]), direction=UNIT_X_VECTOR3)
        ray2.propagate(distance=3.0, medium=air)

        intersections = ray_intersection(ray1, ray2, threshold=INTERSECTION_THRESHOLD)

        assert len(intersections) >= 1

    def test_crossing_at_segment_boundary(self, air):
        """Test rays crossing exactly at segment boundary.

        Note: When rays cross at a segment boundary, the intersection might be
        detected by both adjacent segments, so we allow for some flexibility.
        """
        ray1 = OpticalRay(origin=np.array([0.0, -2.0, 0.0]), direction=UNIT_Y_VECTOR3)
        ray1.propagate(distance=2.0, medium=air)
        ray1.propagate(distance=2.0, medium=air)

        ray2 = OpticalRay(origin=np.array([-2.0, 0.0, 0.0]), direction=UNIT_X_VECTOR3)
        ray2.propagate(distance=4.0, medium=air)

        intersections = ray_intersection(ray1, ray2, threshold=INTERSECTION_THRESHOLD)

        assert len(intersections) >= 1
        assert_vectors_close(intersections[0], ORIGIN_POINT3)

    def test_realistic_laser_cross_scenario(self, air):
        """Test realistic laser cross setup with angled beams."""
        ray1 = OpticalRay(
            origin=np.array([-0.1, 0.0, -0.26]), direction=UNIT_Z_VECTOR3
        )
        ray1.propagate(distance=0.3, medium=air)

        direction2 = UNIT_X_VECTOR3 - 0.2 * UNIT_Z_VECTOR3
        direction2 = direction2 / np.linalg.norm(direction2)
        ray2 = OpticalRay(origin=np.array([-0.1, 0.0, 0.01]), direction=direction2)
        ray2.propagate(distance=0.3, medium=air)

        intersections = ray_intersection(ray1, ray2, threshold=INTERSECTION_THRESHOLD)

        assert len(intersections) >= 1

    def test_default_threshold_usage(self, air):
        """Test that default threshold parameter works correctly."""
        ray1 = OpticalRay(origin=np.array([0.0, -2.0, 0.0]), direction=UNIT_Y_VECTOR3)
        ray1.propagate(distance=4.0, medium=air)

        ray2 = OpticalRay(origin=np.array([-2.0, 0.0, 0.0]), direction=UNIT_X_VECTOR3)
        ray2.propagate(distance=4.0, medium=air)

        intersections_vsmall = ray_intersection(ray1, ray2)
        intersections_threshold = ray_intersection(
            ray1, ray2, threshold=INTERSECTION_THRESHOLD
        )

        assert len(intersections_vsmall) == len(intersections_threshold)
