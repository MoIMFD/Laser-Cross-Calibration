"""Test for laser cross-calibration setup based on Gunady et al. paper.

This test validates the core calibration relationship where the ratio of
intersection point displacement to stage displacement follows:
    Δy_intersection / Δy_stage = (tan α₁ + tan β₁) / (tan α₂ + tan β₂)

For the specific angles used (11.5° and 12.6°) and the material configuration
(air -> fused silica -> water), this ratio should be approximately 1.34.

Reference: Gunady et al., "Laser Cross-Calibration for Dimensional Metrology"
"""

from math import cos, isclose, sin, tan

import numpy as np
import pytest
from scipy.stats import linregress

import laser_cross_calibration as lcc


def calculate_expected_calibration_ratio(
    angle_1_deg: float, angle_2_deg: float, n_air: float, n_glass: float, n_water: float
) -> float:
    """Calculate theoretical calibration ratio from system parameters.

    Args:
        angle_1_deg: First laser angle in degrees
        angle_2_deg: Second laser angle in degrees
        n_air: Refractive index of air
        n_glass: Refractive index of glass
        n_water: Refractive index of water

    Returns:
        Expected ratio Δy_intersection / Δy_stage
    """
    angle_1 = np.deg2rad(angle_1_deg)
    angle_2 = np.deg2rad(angle_2_deg)

    refracted_angle_1_glass = np.arcsin(n_air * np.sin(angle_1) / n_glass)
    refracted_angle_1_water = np.arcsin(n_glass * np.sin(refracted_angle_1_glass) / n_water)

    refracted_angle_2_glass = np.arcsin(n_air * np.sin(angle_2) / n_glass)
    refracted_angle_2_water = np.arcsin(n_glass * np.sin(refracted_angle_2_glass) / n_water)

    alpha_1 = angle_1
    beta_1 = angle_2
    alpha_2 = refracted_angle_1_water
    beta_2 = refracted_angle_2_water

    ratio = (tan(alpha_1) + tan(beta_1)) / (tan(alpha_2) + tan(beta_2))

    return ratio


@pytest.fixture
def gunady_optical_system():
    """Create optical system matching Gunady et al. experimental setup.

    Setup:
        - Air region: y < 0
        - Fused silica plate: 0 <= y <= 0.05
        - Water region: y > 0.05
    """
    system = lcc.tracing.OpticalSystem(max_propagation_distance=10)

    front_interface = lcc.tracing.OpticalInterface(
        geometry=lcc.surfaces.Plane(
            point=lcc.constants.ORIGIN_POINT3,
            normal=-lcc.constants.UNIT_Y_VECTOR3,
        ),
        material_pre=lcc.materials.AIR,
        material_post=lcc.materials.GLASS_FUSED_SILICA,
    )
    system.add_interface(front_interface)

    back_interface = lcc.tracing.OpticalInterface(
        geometry=lcc.surfaces.Plane(
            point=lcc.constants.UNIT_Y_VECTOR3 * 0.05,
            normal=lcc.constants.UNIT_Y_VECTOR3,
        ),
        material_pre=lcc.materials.GLASS_FUSED_SILICA,
        material_post=lcc.materials.WATER,
    )
    system.add_interface(back_interface)

    return system


@pytest.fixture
def gunady_laser_source():
    """Create dual laser source matching Gunady et al. configuration.

    Configuration:
        - Source position: y = -0.3 (below optical interfaces)
        - Arm separation: 0.2 (0.1 on each side)
        - Laser 1 angle: 11.5° (pointing inward and upward)
        - Laser 2 angle: 12.6° (pointing inward and upward)
    """
    arm_length = 0.1
    angle_1 = np.deg2rad(11.5)
    angle_2 = np.deg2rad(12.6)

    source = lcc.sources.DualLaserStageSource(
        origin=lcc.constants.ORIGIN_POINT3 - lcc.constants.UNIT_Y_VECTOR3 * 0.3,
        arm1=lcc.constants.UNIT_X_VECTOR3 * arm_length,
        arm2=-lcc.constants.UNIT_X_VECTOR3 * arm_length,
        direction1=np.array([-sin(angle_1), cos(angle_1), 0.0]) * 0.1,
        direction2=np.array([sin(angle_2), cos(angle_2), 0.0]) * 0.1,
    )

    return source


@pytest.mark.integration
def test_gunady_calibration_ratio(gunady_optical_system, gunady_laser_source):
    """Test that stage displacement produces correct intersection displacement ratio.

    This validates the fundamental calibration relationship from the Gunady paper.
    The test:
    1. Calculates the theoretical ratio from physics (Snell's law + geometry)
    2. Measures the actual ratio by tracing rays at multiple stage positions
    3. Verifies they match within measurement tolerance

    The tolerance of 1% accounts for numerical precision in ray tracing and
    intersection calculations.
    """
    angle_1_deg = 11.5
    angle_2_deg = 12.6
    n_air = lcc.materials.AIR.n()
    n_glass = lcc.materials.GLASS_FUSED_SILICA.n()
    n_water = lcc.materials.WATER.n()

    expected_ratio = calculate_expected_calibration_ratio(
        angle_1_deg, angle_2_deg, n_air, n_glass, n_water
    )

    system = gunady_optical_system
    source = gunady_laser_source
    tracer = lcc.tracing.RayTracer(optical_system=system)

    stage_shifts = []
    intersection_positions = []

    stage_shift_increment = 0.05
    num_positions = 5

    for i in range(num_positions):
        stage_shift = i * stage_shift_increment
        stage_shifts.append(stage_shift)

        rays, intersections = tracer.trace_and_find_crossings(sources=[source])

        assert len(intersections) == 1, "Expected exactly one intersection point"
        intersection_positions.append(intersections[0])

        source.translate(y=stage_shift_increment)

    intersection_y_coords = np.array(intersection_positions)[:, 1]
    intersection_displacements = intersection_y_coords - intersection_y_coords[0]

    regression = linregress(stage_shifts, intersection_displacements)

    relative_tolerance = 0.01

    assert isclose(
        regression.slope, expected_ratio, rel_tol=relative_tolerance
    ), (
        f"Calibration ratio mismatch:\n"
        f"  Expected (theory): {expected_ratio:.6f}\n"
        f"  Measured (simulation): {regression.slope:.6f}\n"
        f"  Difference: {abs(regression.slope - expected_ratio):.6f}\n"
        f"  Tolerance: {relative_tolerance * 100:.1f}%"
    )

    assert (
        regression.rvalue > 0.999
    ), f"Expected linear relationship (R > 0.999), got R = {regression.rvalue:.6f}"


@pytest.mark.unit
def test_ray_propagates_through_correct_media(gunady_optical_system, gunady_laser_source):
    """Test that rays propagate through the correct sequence of media.

    This is a regression test for the bug where rays would incorrectly return to
    air after passing through the glass, instead of entering water. The bug was
    caused by accepting negative intersection distances in plane.py.

    Expected media sequence: air -> fused silica -> water
    """
    system = gunady_optical_system
    source = gunady_laser_source
    tracer = lcc.tracing.RayTracer(optical_system=system)

    rays, _ = tracer.trace_and_find_crossings(sources=[source])

    assert len(rays) == 2, "Expected two rays from dual laser source"

    for i, ray in enumerate(rays):
        media_names = [medium.name for medium in ray.media_history]

        assert len(media_names) >= 3, (
            f"Ray {i} should pass through at least 3 media, got {len(media_names)}"
        )

        assert media_names[0] == "air", f"Ray {i} should start in air, got {media_names[0]}"

        assert media_names[1] == "fused silica", (
            f"Ray {i} should pass through fused silica, got {media_names[1]}"
        )

        assert media_names[2] == "water", (
            f"Ray {i} should enter water after glass, got {media_names[2]} "
            f"(full sequence: {media_names})"
        )

        assert "air" not in media_names[2:], (
            f"Ray {i} should not return to air after entering glass "
            f"(full sequence: {media_names})"
        )
