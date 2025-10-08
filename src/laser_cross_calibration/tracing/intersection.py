"""Intersection utilities for ray path analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from laser_cross_calibration.constants import INTERSECTION_THRESHOLD, VSMALL

if TYPE_CHECKING:
    from laser_cross_calibration.tracing.ray import OpticalRay
    from laser_cross_calibration.types import POINT3


def line_segment_intersection(
    p1: POINT3,
    p2: POINT3,
    p3: POINT3,
    p4: POINT3,
    threshold: float = INTERSECTION_THRESHOLD,
) -> tuple[bool, POINT3]:
    """
    Find intersection between two 3D line segments.

    Uses the closest approach method for two line segments in 3D space.

    Args:
        p1, p2: Start and end points of first line segment
        p3, p4: Start and end points of second line segment
        threshold: Distance threshold for considering intersection

    Returns:
        Tuple of (intersection_found, intersection_point)
    """
    p1, p2, p3, p4 = [np.array(p, dtype=np.float64) for p in [p1, p2, p3, p4]]

    d1 = p2 - p1
    d2 = p4 - p3
    w = p1 - p3

    len1_sq = np.dot(d1, d1)
    len2_sq = np.dot(d2, d2)

    if len1_sq < threshold or len2_sq < threshold:
        return False, np.array([np.nan, np.nan, np.nan])

    a = len1_sq
    b = np.dot(d1, d2)
    c = len2_sq
    d = np.dot(d1, w)
    e = np.dot(d2, w)

    denom = a * c - b * b

    if abs(denom) < threshold:
        if a > threshold:
            t = -d / a
            closest_on_1 = p1 + t * d1
            dist_to_line2 = np.linalg.norm(closest_on_1 - p3)

            if dist_to_line2 < threshold:
                t1_start = 0
                t1_end = 1
                t2_start = np.dot(p3 - p1, d1) / len1_sq
                t2_end = np.dot(p4 - p1, d1) / len1_sq

                overlap_start = max(t1_start, min(t2_start, t2_end))
                overlap_end = min(t1_end, max(t2_start, t2_end))

                if overlap_start <= overlap_end:
                    t_mid = (overlap_start + overlap_end) / 2
                    intersection_point = p1 + t_mid * d1
                    return True, intersection_point

        return False, np.array([np.nan, np.nan, np.nan])

    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom

    if not (0 <= s <= 1 and 0 <= t <= 1):
        return False, np.array([np.nan, np.nan, np.nan])

    point1 = p1 + s * d1
    point2 = p3 + t * d2

    distance = np.linalg.norm(point2 - point1)

    if distance < threshold:
        intersection_point = (point1 + point2) / 2
        return True, intersection_point
    else:
        return False, np.array([np.nan, np.nan, np.nan])


def ray_intersection(
    ray1: OpticalRay, ray2: OpticalRay, threshold: float = VSMALL
) -> list[POINT3]:
    """
    Find all intersection points between two rays' path segments.

    Checks every segment of ray1 against every segment of ray2 to find
    where the actual physical ray paths intersect.

    Args:
        ray1: First optical ray
        ray2: Second optical ray
        threshold: Distance threshold for considering intersection

    Returns:
        List of intersection points (empty if no intersections found)
    """
    intersections = []

    for i in range(len(ray1.segment_distances)):
        p1_start = ray1.path_positions[i]
        p1_end = ray1.path_positions[i + 1]

        for j in range(len(ray2.segment_distances)):
            p2_start = ray2.path_positions[j]
            p2_end = ray2.path_positions[j + 1]

            intersects, point = line_segment_intersection(
                p1_start, p1_end, p2_start, p2_end, threshold=threshold
            )

            if intersects:
                intersections.append(point)

    return intersections
