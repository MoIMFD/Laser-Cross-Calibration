import math

import numpy as np
import pytest

from laser_cross_calibration.constants import (
    ORIGIN_POINT3,
    UNIT_X_VECTOR3,
    UNIT_Z_VECTOR3,
)
from laser_cross_calibration.tracing import OpticalRay
from laser_cross_calibration.utils import normalize


@pytest.mark.unit
class TestOpticalRay:
    def test_creation(self):
        """Test basic ray creation with valid inputs."""
        ray = OpticalRay(origin=ORIGIN_POINT3, direction=UNIT_X_VECTOR3)
        assert np.allclose(ray.origin, np.zeros(3))
        assert np.allclose(ray.direction, np.array((1, 0, 0)))

    def test_direction_is_normalized(self):
        """Test that direction vector is automatically normalized."""
        ray = OpticalRay(origin=ORIGIN_POINT3, direction=np.array([3.0, 4.0, 0.0]))
        assert np.allclose(ray.direction, np.array([0.6, 0.8, 0.0]))
        assert np.isclose(np.linalg.norm(ray.direction), 1.0)

    def test_initial_state(self):
        """Test that initial state is correctly set."""
        origin = np.array([1.0, 2.0, 3.0])
        direction = UNIT_Z_VECTOR3
        ray = OpticalRay(origin=origin, direction=direction)

        assert np.allclose(ray.position, origin)
        assert np.allclose(ray.current_direction, direction)
        assert ray.is_alive is True
        assert len(ray.path_positions) == 1
        assert len(ray.path_directions) == 1
        assert len(ray.segment_distances) == 0

    def test_invalid_origin_shape(self):
        """Test that invalid origin shape raises ValueError."""
        with pytest.raises(ValueError, match="Origin must be 3D point"):
            OpticalRay(origin=np.array([1.0, 2.0]), direction=UNIT_X_VECTOR3)

    def test_invalid_direction_shape(self):
        """Test that invalid direction shape raises ValueError."""
        with pytest.raises(ValueError, match="Direction must be 3D vector"):
            OpticalRay(origin=ORIGIN_POINT3, direction=np.array([1.0, 0.0]))

    def test_zero_direction_raises_error(self):
        """Test that zero direction vector raises ValueError."""
        with pytest.raises(ValueError, match="Direction vector cannot be zero"):
            OpticalRay(origin=ORIGIN_POINT3, direction=np.array([0.0, 0.0, 0.0]))

    def test_propagate(self, air):
        """Test ray propagation through medium."""
        ray = OpticalRay(origin=ORIGIN_POINT3, direction=UNIT_X_VECTOR3)
        ray.propagate(distance=10.0, medium=air)

        assert np.allclose(ray.position, np.array([10.0, 0.0, 0.0]))
        assert len(ray.path_positions) == 2
        assert len(ray.segment_distances) == 1
        assert ray.segment_distances[0] == 10.0

    def test_propagate_multiple_segments(self, air, water):
        """Test multiple propagation segments."""
        ray = OpticalRay(origin=ORIGIN_POINT3, direction=UNIT_Z_VECTOR3)

        ray.propagate(distance=5.0, medium=air)
        ray.propagate(distance=3.0, medium=water)

        assert np.allclose(ray.position, np.array([0.0, 0.0, 8.0]))
        assert len(ray.segment_distances) == 2
        assert ray.segment_distances == [5.0, 3.0]
        assert len(ray.media_history) == 2

    def test_refract_rare_to_dense(self, air, water):
        """Test refraction on a surface from optic rare to optic dense medium"""
        ray = OpticalRay(origin=ORIGIN_POINT3, direction=UNIT_X_VECTOR3)
        surface_normal = normalize(np.array((0.5, 1, 0)))

        # angle prior to refraction
        theta_1 = math.acos(abs(surface_normal.dot(ray.current_direction)))

        # perform refraction operation
        is_refracted = ray.refract(
            surface_normal=surface_normal, medium_from=air, medium_to=water
        )
        assert is_refracted is True

        # angle after refraction
        theta_2 = math.acos(abs(surface_normal.dot(ray.current_direction)))
        assert theta_2 < theta_1
        # test if Snell's law is full filled
        assert math.isclose(math.sin(theta_1) / math.sin(theta_2), water.n() / air.n())

    def test_refract_dense_to_rare(self, air, water):
        """Test refraction on a surface from optic dense to optic rare medium"""
        ray = OpticalRay(origin=ORIGIN_POINT3, direction=UNIT_X_VECTOR3)
        surface_normal = normalize(np.array((0.9, 1.0, 0)))

        # angle prior to refraction
        theta_1 = math.acos(abs(surface_normal.dot(ray.current_direction)))

        # perform refraction operation
        is_refracted = ray.refract(
            surface_normal=surface_normal, medium_from=water, medium_to=air
        )
        assert is_refracted is True

        # angle after refraction
        theta_2 = math.acos(abs(surface_normal.dot(ray.current_direction)))
        assert theta_2 > theta_1
        # test if Snell's law is full filled
        assert math.isclose(math.sin(theta_1) / math.sin(theta_2), air.n() / water.n())

    def test_total_internal_reflection_above_critical_angle(self, air, water):
        """Test total internal reflection above critical angle.

        Critical angle for water -> air: theta_c = arcsin(n_air/n_water) ~ 48.6 deg
        At angles > theta_c, light should reflect instead of refract.
        """
        critical_angle = math.asin(air.n() / water.n())
        incident_angle = critical_angle + np.radians(5)

        ray_direction = np.array(
            [
                np.sin(incident_angle),
                0.0,
                np.cos(incident_angle),
            ]
        )
        ray = OpticalRay(origin=ORIGIN_POINT3, direction=ray_direction)
        surface_normal = UNIT_Z_VECTOR3

        theta_before = math.acos(abs(surface_normal.dot(ray.current_direction)))
        assert math.isclose(theta_before, incident_angle, abs_tol=1e-6)

        is_refracted = ray.refract(
            surface_normal=surface_normal, medium_from=water, medium_to=air
        )

        assert is_refracted is False

        theta_after = math.acos(abs(surface_normal.dot(ray.current_direction)))
        assert math.isclose(theta_after, theta_before)

    def test_refraction_below_critical_angle(self, air, water):
        """Test normal refraction below critical angle.

        At angles < theta_c, light should refract normally.
        """
        critical_angle = math.asin(air.n() / water.n())
        incident_angle = critical_angle - np.radians(5)

        ray_direction = np.array(
            [
                np.sin(incident_angle),
                0.0,
                np.cos(incident_angle),
            ]
        )
        ray = OpticalRay(origin=ORIGIN_POINT3, direction=ray_direction)
        surface_normal = UNIT_Z_VECTOR3

        theta_before = math.acos(abs(surface_normal.dot(ray.current_direction)))
        assert math.isclose(theta_before, incident_angle, abs_tol=1e-6)

        is_refracted = ray.refract(
            surface_normal=surface_normal, medium_from=water, medium_to=air
        )

        assert is_refracted is True

        theta_after = math.acos(abs(surface_normal.dot(ray.current_direction)))
        assert theta_after > theta_before
