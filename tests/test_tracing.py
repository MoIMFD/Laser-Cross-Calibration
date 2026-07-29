from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from laser_cross_calibration.tracing import OpticalRay
from tests.utils import assert_vectors_close

if TYPE_CHECKING:
    from hazy import Frame


@pytest.mark.unit
class TestOpticalRay:
    def test_creation(self, frame: Frame):
        """Test basic ray creation with valid inputs."""
        ray = OpticalRay(
            origin=frame.origin,
            direction=frame.x_axis,
        )
        assert_vectors_close(ray.origin, frame.origin)
        assert_vectors_close(ray.initial_direction, frame.x_axis)

    def test_direction_is_normalized(self, frame: Frame):
        """Test that direction vector is automatically normalized."""
        ray = OpticalRay(origin=frame.origin, direction=frame.vector(3.0, 4.0, 0.0))
        assert_vectors_close(ray.initial_direction, np.array([0.6, 0.8, 0.0]))
        assert np.isclose(ray.initial_direction.magnitude, 1.0)

    def test_initial_state(self, frame: Frame):
        """Test that initial state is correctly set."""
        origin = frame.point(1.0, 2.0, 3.0)
        direction = frame.z_axis
        ray = OpticalRay(origin=origin, direction=direction)

        assert_vectors_close(ray.current_position, origin)
        assert_vectors_close(ray.current_direction, direction)
        assert ray.is_alive is True
        assert len(ray.path_positions) == 1
        assert len(ray.path_directions) == 1
        assert len(ray.segment_distances) == 0

    def test_invalid_origin_type(self, frame: Frame):
        """Test that invalid origin type raises ValueError."""
        with pytest.raises(RuntimeError, match="Expected object with frame attribute"):
            OpticalRay(origin=np.array([1.0, 2.0, 3.0]), direction=frame.x_axis)

    def test_invalid_direction_type(self, frame: Frame):
        """Test that invalid direction shape raises ValueError."""
        with pytest.raises(RuntimeError, match="Expected object with frame attribute"):
            OpticalRay(origin=frame.origin, direction=np.array([1.0, 0.0, 2.0]))

    def test_zero_direction_raises_error(self, frame: Frame):
        """Test that zero direction vector raises ValueError."""
        with pytest.raises(
            RuntimeError, match="Can not normalize vector with zero length Vector"
        ):
            OpticalRay(origin=frame.origin, direction=frame.vector(0.0, 0.0, 0.0))

    def test_propagate(self, frame: Frame, air):
        """Test ray propagation through medium."""
        ray = OpticalRay(origin=frame.origin, direction=frame.x_axis)
        ray.propagate(distance=10.0, medium=air)

        assert_vectors_close(ray.current_position, frame.point(10.0, 0.0, 0.0))
        assert len(ray.path_positions) == 2
        assert len(ray.segment_distances) == 1
        assert ray.segment_distances[0] == 10.0

    def test_propagate_multiple_segments(self, air, water, frame: Frame):
        """Test multiple propagation segments."""
        ray = OpticalRay(origin=frame.origin, direction=frame.z_axis)

        ray.propagate(distance=5.0, medium=air)
        ray.propagate(distance=3.0, medium=water)

        assert_vectors_close(ray.current_position, np.array([0.0, 0.0, 8.0]))
        assert len(ray.segment_distances) == 2
        assert ray.segment_distances == [5.0, 3.0]
        assert len(ray.media_history) == 2

    def test_refract_rare_to_dense(self, air, water, frame: Frame):
        """Test refraction on a surface from optic rare to optic dense medium"""
        ray = OpticalRay(origin=frame.origin, direction=frame.x_axis)
        surface_normal = frame.vector((0.5, 1, 0)).normalize()

        # angle prior to refraction
        theta_1 = np.acos(abs(surface_normal.dot(ray.current_direction)))

        # perform refraction operation
        is_refracted = ray.refract(
            surface_normal=surface_normal, medium_from=air, medium_to=water
        )
        assert is_refracted is True

        # angle after refraction
        theta_2 = np.acos(abs(surface_normal.dot(ray.current_direction)))
        assert theta_2 < theta_1
        # test if Snell's law is full filled
        assert np.isclose(np.sin(theta_1) / np.sin(theta_2), water.n() / air.n())

    def test_refract_dense_to_rare(self, air, water, frame: Frame):
        """Test refraction on a surface from optic dense to optic rare medium"""
        ray = OpticalRay(origin=frame.origin, direction=frame.x_axis)
        surface_normal = frame.vector(0.9, 1.0, 0).normalize()

        # angle prior to refraction
        theta_1 = np.acos(abs(surface_normal.dot(ray.current_direction)))

        # perform refraction operation
        is_refracted = ray.refract(
            surface_normal=surface_normal, medium_from=water, medium_to=air
        )
        assert is_refracted is True

        # angle after refraction
        theta_2 = np.acos(abs(surface_normal.dot(ray.current_direction)))
        assert theta_2 > theta_1
        # test if Snell's law is full filled
        assert np.isclose(np.sin(theta_1) / np.sin(theta_2), air.n() / water.n())

    def test_total_internal_reflection_above_critical_angle(
        self, air, water, frame: Frame
    ):
        """Test total internal reflection above critical angle.

        Critical angle for water -> air: theta_c = arcsin(n_air/n_water) ~ 48.6 deg
        At angles > theta_c, light should reflect instead of refract.
        """
        critical_angle = math.asin(air.n() / water.n())
        incident_angle = critical_angle + np.radians(5)

        ray_direction = frame.vector(
            np.sin(incident_angle),
            0.0,
            np.cos(incident_angle),
        )
        ray = OpticalRay(origin=frame.origin, direction=ray_direction)
        surface_normal = frame.z_axis

        theta_before = math.acos(abs(surface_normal.dot(ray.current_direction)))
        assert math.isclose(theta_before, incident_angle, abs_tol=1e-6)

        is_refracted = ray.refract(
            surface_normal=surface_normal, medium_from=water, medium_to=air
        )

        assert is_refracted is False

        theta_after = math.acos(abs(surface_normal.dot(ray.current_direction)))
        assert math.isclose(theta_after, theta_before)

    def test_refraction_below_critical_angle(self, air, water, frame: Frame):
        """Test normal refraction below critical angle.

        At angles < theta_c, light should refract normally.
        """
        critical_angle = math.asin(air.n() / water.n())
        incident_angle = critical_angle - np.radians(5)

        ray_direction = frame.vector(
            np.sin(incident_angle),
            0.0,
            np.cos(incident_angle),
        )
        ray = OpticalRay(origin=frame.origin, direction=ray_direction)
        surface_normal = frame.z_axis

        theta_before = math.acos(abs(surface_normal.dot(ray.current_direction)))
        assert math.isclose(theta_before, incident_angle, abs_tol=1e-6)

        is_refracted = ray.refract(
            surface_normal=surface_normal, medium_from=water, medium_to=air
        )

        assert is_refracted is True

        theta_after = math.acos(abs(surface_normal.dot(ray.current_direction)))
        assert theta_after > theta_before

    def test_copy(self, air, water, frame: Frame):
        """Test copying a ray and verifying independence."""
        original_ray = OpticalRay(origin=frame.origin, direction=frame.x_axis)
        original_ray.propagate(5, air)

        copied_ray = original_ray.copy()

        assert original_ray is not copied_ray

        assert_vectors_close(original_ray.origin, copied_ray.origin)
        assert_vectors_close(
            original_ray.initial_direction, copied_ray.initial_direction
        )
        assert_vectors_close(original_ray.current_position, copied_ray.current_position)
        assert_vectors_close(
            original_ray.current_direction, copied_ray.current_direction
        )
        assert original_ray.is_alive == copied_ray.is_alive
        assert len(original_ray.path_positions) == len(copied_ray.path_positions)
        assert len(original_ray.path_directions) == len(copied_ray.path_directions)
        assert all(
            mat1.name == mat2.name and math.isclose(mat1.n(), mat2.n())
            for mat1, mat2 in zip(
                original_ray.media_history, copied_ray.media_history, strict=False
            )
        )
        assert original_ray.segment_distances == copied_ray.segment_distances

        original_ray.propagate(distance=1.0, medium=air)
        copied_ray.current_direction = frame.z_axis
        copied_ray.propagate(distance=1.0, medium=water)

        assert not np.allclose(
            np.array(original_ray.current_direction),
            np.array(copied_ray.current_direction),
        )

        assert not all(
            mat1.name == mat2.name and math.isclose(mat1.n(), mat2.n())
            for mat1, mat2 in zip(
                original_ray.media_history, copied_ray.media_history, strict=False
            )
        )

        original_ray.propagate(distance=1.0, medium=air)
        assert len(original_ray.media_history) != len(copied_ray.media_history)

    def test_copy_fresh_ray(self, frame: Frame):
        """Test copying a ray that hasn't been propagated."""
        original_ray = OpticalRay(origin=frame.origin, direction=frame.x_axis)
        copied_ray = original_ray.copy()

        assert original_ray is not copied_ray
        assert_vectors_close(original_ray.origin, copied_ray.origin)
        assert len(copied_ray.media_history) == 0
        assert len(copied_ray.path_positions) == 1

    def test_copy_independence_of_nested_structures(self, air, frame: Frame):
        """Verify modifying nested structures doesn't affect the original."""
        original_ray = OpticalRay(origin=frame.origin, direction=frame.x_axis)
        original_ray.propagate(5, air)

        copied_ray = original_ray.copy()

        if len(copied_ray.path_positions) > 0:
            copied_ray.path_positions[0] = frame.point(999, 999, 999)

        assert not np.allclose(
            np.array(original_ray.path_positions[0]),
            np.array(copied_ray.path_positions[0]),
        )

    def test_rotation_x(self, frame: Frame):
        """Test rotation of ray along x-axis."""
        ray = OpticalRay(origin=frame.origin, direction=frame.x_axis)

        ray.rotate(rx=np.pi / 2)
        assert_vectors_close(ray.current_direction, frame.x_axis)

        ray.rotate(ry=np.pi / 2)
        assert_vectors_close(ray.current_direction, -frame.z_axis)

        ray.rotate(ry=np.pi / 2)
        assert_vectors_close(ray.current_direction, -frame.x_axis)

        ray.rotate(ry=np.pi)
        assert_vectors_close(ray.current_direction, frame.x_axis)

    def test_rotation_y(self, frame: Frame):
        """Test rotation of ray along y-axis."""
        ray = OpticalRay(origin=frame.origin, direction=frame.y_axis)

        ray.rotate(ry=np.pi / 2)
        assert_vectors_close(ray.current_direction, frame.y_axis)

        ray.rotate(rx=np.pi / 2)
        assert_vectors_close(ray.current_direction, frame.z_axis)

        ray.rotate(rx=np.pi / 2)
        assert_vectors_close(ray.current_direction, -frame.y_axis)

        ray.rotate(rx=np.pi)
        assert_vectors_close(ray.current_direction, frame.y_axis)

    def test_rotation_z(self, frame: Frame):
        """Test rotation of ray along z-axis."""
        ray = OpticalRay(origin=frame.origin, direction=frame.z_axis)

        ray.rotate(rz=np.pi / 2)
        assert_vectors_close(ray.current_direction, frame.z_axis)

        ray.rotate(rx=np.pi / 2)
        assert_vectors_close(ray.current_direction, -frame.y_axis)

        ray.rotate(rx=np.pi / 2)
        assert_vectors_close(ray.current_direction, -frame.z_axis)

        ray.rotate(rx=np.pi)
        assert_vectors_close(ray.current_direction, frame.z_axis)

        assert_vectors_close(ray.origin, frame.origin)

    def test_rotation_chaining(self, frame: Frame):
        """Test chaining of rotation operations."""
        ray = OpticalRay(origin=frame.origin, direction=frame.x_axis)

        assert_vectors_close(ray.initial_direction, frame.x_axis)

        ray.rotate(rx=-np.pi).rotate(rz=np.pi / 2).rotate(ry=np.pi)
        assert_vectors_close(ray.initial_direction, frame.y_axis)

        ray.rotate(ry=-np.pi).rotate(rz=-np.pi / 2).rotate(rx=np.pi)
        assert_vectors_close(ray.initial_direction, frame.x_axis)

    def test_non_origin_rotation(self, frame: Frame):
        """Test rotation for rays with origin different than (0, 0, 0)."""
        ray = OpticalRay(origin=frame.origin + frame.x_axis, direction=frame.x_axis)
        assert_vectors_close(ray.origin, frame.origin + frame.x_axis)
        assert_vectors_close(ray.initial_direction, frame.x_axis)

        ray.rotate(ry=np.pi / 2)
        assert_vectors_close(ray.origin, frame.origin - frame.z_axis)
        assert_vectors_close(ray.initial_direction, -frame.z_axis)

        ray = OpticalRay(
            origin=frame.origin + 2 * frame.y_axis + 3 * frame.x_axis,
            direction=frame.z_axis,
        )
        assert_vectors_close(ray.origin, [3, 2, 0])
        assert_vectors_close(ray.initial_direction, frame.z_axis)

        ray.rotate(rz=3 * np.pi / 2)
        assert_vectors_close(ray.origin, [2, -3, 0])
        assert_vectors_close(ray.initial_direction, frame.z_axis)

    def test_translation(self, frame: Frame):
        """Test translation operation of a ray."""
        ray = OpticalRay(origin=frame.origin, direction=frame.x_axis)
        assert_vectors_close(ray.origin, frame.origin)

        ray.translate()
        assert_vectors_close(ray.origin, frame.origin)

        ray.translate(x=5)
        assert_vectors_close(ray.origin, frame.origin + 5 * frame.x_axis)

        ray.translate(x=-5)
        assert_vectors_close(ray.origin, frame.origin)

        ray.translate(y=3, z=2)
        assert_vectors_close(
            ray.origin, frame.origin + 3 * frame.y_axis + 2 * frame.z_axis
        )

        ray.translate(x=-10)
        assert_vectors_close(
            ray.origin,
            frame.origin + 3 * frame.y_axis + 2 * frame.z_axis - 10 * frame.x_axis,
        )

    def test_translation_chaining(self, frame: Frame):
        """Test chaining of translation operations."""
        ray = OpticalRay(origin=frame.origin, direction=frame.x_axis)
        ray.translate(x=10).translate(y=-3).translate(z=-2)

        assert_vectors_close(
            ray.origin,
            frame.origin + 10 * frame.x_axis - 3 * frame.y_axis - 2 * frame.z_axis,
        )
