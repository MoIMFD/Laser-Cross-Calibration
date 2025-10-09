from __future__ import annotations

import numpy as np
import pytest

from laser_cross_calibration import materials, sources, surfaces, tracing
from laser_cross_calibration.constants import (
    ORIGIN_POINT3,
    UNIT_X_VECTOR3,
    UNIT_Y_VECTOR3,
    UNIT_Z_VECTOR3,
)


@pytest.fixture(autouse=True)
def reset_surface_counter():
    """Automatically reset surface ID counter after each test."""
    yield
    surfaces.Surface.reset_id_counter()


@pytest.fixture
def air():
    return materials.ConstantMaterial(name="air", ior=1.0)


@pytest.fixture
def water():
    return materials.ConstantMaterial(name="water", ior=1.333)


@pytest.fixture
def glass_bk7():
    return materials.GLASS_BK7


@pytest.fixture(scope="function")
def simple_ray_100():
    return tracing.OpticalRay(
        origin=ORIGIN_POINT3,
        direction=UNIT_X_VECTOR3,
    )


@pytest.fixture(scope="function")
def simple_ray_010():
    return tracing.OpticalRay(
        origin=ORIGIN_POINT3,
        direction=UNIT_Y_VECTOR3,
    )


@pytest.fixture(scope="function")
def simple_ray_001():
    return tracing.OpticalRay(
        origin=ORIGIN_POINT3,
        direction=UNIT_Z_VECTOR3,
    )


@pytest.fixture(scope="function")
def simple_ray_110():
    return tracing.OpticalRay(
        origin=ORIGIN_POINT3,
        direction=UNIT_X_VECTOR3 + UNIT_Y_VECTOR3,
    )


@pytest.fixture(scope="function")
def simple_ray_101():
    return tracing.OpticalRay(
        origin=ORIGIN_POINT3,
        direction=UNIT_X_VECTOR3 + UNIT_Z_VECTOR3,
    )


@pytest.fixture(scope="function")
def simple_ray_011():
    return tracing.OpticalRay(
        origin=ORIGIN_POINT3,
        direction=UNIT_Y_VECTOR3 + UNIT_Z_VECTOR3,
    )


@pytest.fixture(scope="function")
def single_triangle_surface_xy() -> surfaces.TriSurface:
    """Single triangle in XY plane at Z=0"""
    vertices = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    return surfaces.TriSurface(vertices=vertices, faces=faces)


@pytest.fixture(scope="function")
def single_triangle_surface_xz() -> surfaces.TriSurface:
    """Single triangle in XZ plane at Y=0"""
    vertices = np.array([[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    faces = np.array([[2, 1, 0]])
    return surfaces.TriSurface(vertices=vertices, faces=faces)


@pytest.fixture(scope="function")
def single_triangle_surface_yz() -> surfaces.TriSurface:
    """Single triangle in YZ plane at X=0"""
    vertices = np.asarray([[0.0, 1.0, -1.0], [0.0, 0.0, 1.0], [0.0, -1.0, -1.0]])
    faces = np.asarray([[0, 1, 2]])
    return surfaces.TriSurface(vertices=vertices, faces=faces)


@pytest.fixture
def triangle_box_surface() -> surfaces.TriSurface:
    """Simple box (12 triangles, 2 per face) for testing multi-surface intersections."""
    vertices = np.asarray(
        [
            # Bottom face (Z=0)
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            # Top face (Z=1)
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    faces = np.asarray(
        [
            # Bottom (Z=0)
            [0, 1, 2],
            [0, 2, 3],
            # Top (Z=1)
            [4, 6, 5],
            [4, 7, 6],
            # Front (Y=0)
            [0, 5, 1],
            [0, 4, 5],
            # Back (Y=1)
            [2, 7, 3],
            [2, 6, 7],
            # Left (X=0)
            [0, 3, 7],
            [0, 7, 4],
            # Right (X=1)
            [1, 5, 6],
            [1, 6, 2],
        ]
    )
    return surfaces.TriSurface(vertices=vertices, faces=faces)


@pytest.fixture
def single_laser_source_x() -> sources.SingleLaserSource:
    return sources.SingleLaserSource(origin=ORIGIN_POINT3, direction=UNIT_X_VECTOR3)


@pytest.fixture
def dual_laser_stage_source_xy() -> sources.DualLaserStageSource:
    return sources.DualLaserStageSource(
        origin=ORIGIN_POINT3,
        arm1=UNIT_X_VECTOR3,
        arm2=UNIT_Y_VECTOR3,
        direction1=UNIT_Y_VECTOR3,
        direction2=UNIT_X_VECTOR3,
    )


@pytest.fixture(scope="session")
def stl_cube(tmp_path_factory):
    """Generate a STL file for a cube with edge length 2."""
    tmp_path = tmp_path_factory.mktemp("stl_data")
    stl_content = """solid cube
facet normal 0 0 1
 outer loop
  vertex 1 1 1
  vertex -1 1 1
  vertex -1 -1 1
 endloop
endfacet
facet normal 0 -0 1
 outer loop
  vertex 1 1 1
  vertex -1 -1 1
  vertex 1 -1 1
 endloop
endfacet
facet normal 0 -1 0
 outer loop
  vertex 1 -1 -1
  vertex 1 -1 1
  vertex -1 -1 1
 endloop
endfacet
facet normal 0 -1 0
 outer loop
  vertex 1 -1 -1
  vertex -1 -1 1
  vertex -1 -1 -1
 endloop
endfacet
facet normal -1 -0 -0
 outer loop
  vertex -1 -1 -1
  vertex -1 -1 1
  vertex -1 1 1
 endloop
endfacet
facet normal -1 -0 0
 outer loop
  vertex -1 -1 -1
  vertex -1 1 1
  vertex -1 1 -1
 endloop
endfacet
facet normal 0 0 -1
 outer loop
  vertex -1 1 -1
  vertex 1 1 -1
  vertex 1 -1 -1
 endloop
endfacet
facet normal 0 0 -1
 outer loop
  vertex -1 1 -1
  vertex 1 -1 -1
  vertex -1 -1 -1
 endloop
endfacet
facet normal 1 -0 0
 outer loop
  vertex 1 1 -1
  vertex 1 1 1
  vertex 1 -1 1
 endloop
endfacet
facet normal 1 -0 0
 outer loop
  vertex 1 1 -1
  vertex 1 -1 1
  vertex 1 -1 -1
 endloop
endfacet
facet normal 0 1 0
 outer loop
  vertex -1 1 -1
  vertex -1 1 1
  vertex 1 1 1
 endloop
endfacet
facet normal 0 1 -0
 outer loop
  vertex -1 1 -1
  vertex 1 1 1
  vertex 1 1 -1
 endloop
endfacet
endsolid cube
"""
    stl_file_path = tmp_path / "cube.stl"
    stl_file_path.write_text(stl_content)
    return stl_file_path


@pytest.fixture(scope="session")
def stl_uv_sphere(tmp_path_factory):
    """Generate a STL file for a uv sphere with radius 1."""
    tmp_path = tmp_path_factory.mktemp("stl_data")
    stl_content = """solid uvsphere
facet normal 0.37147036 0.89680886 0.24029912
 outer loop
  vertex 0 0.9999999 0
  vertex 0 0.8660253 0.5
  vertex 0.6123724 0.6123724 0.5
 endloop
endfacet
facet normal 0.37147027 0.89680886 0.24029924
 outer loop
  vertex 0 0.9999999 0
  vertex 0.6123724 0.6123724 0.5
  vertex 0.70710677 0.70710677 0
 endloop
endfacet
facet normal 0.10659552 0.25734445 0.96042234
 outer loop
  vertex 0 0.49999994 0.8660254
  vertex 0 0 1
  vertex 0.35355338 0.35355338 0.8660254
 endloop
endfacet
facet normal 0.10659552 0.25734445 -0.96042234
 outer loop
  vertex 0 0 -1
  vertex 0 0.49999994 -0.8660254
  vertex 0.35355338 0.35355338 -0.8660254
 endloop
endfacet
facet normal 0.37147027 0.89680886 -0.24029914
 outer loop
  vertex 0 0.8660253 -0.5
  vertex 0 0.9999999 0
  vertex 0.70710677 0.70710677 0
 endloop
endfacet
facet normal 0.37147036 0.89680886 -0.24029927
 outer loop
  vertex 0 0.8660253 -0.5
  vertex 0.70710677 0.70710677 0
  vertex 0.6123724 0.6123724 -0.5
 endloop
endfacet
facet normal 0.28108454 0.67859834 0.67859834
 outer loop
  vertex 0 0.8660253 0.5
  vertex 0 0.49999994 0.8660254
  vertex 0.35355338 0.35355338 0.8660254
 endloop
endfacet
facet normal 0.28108457 0.67859846 0.6785983
 outer loop
  vertex 0 0.8660253 0.5
  vertex 0.35355338 0.35355338 0.8660254
  vertex 0.6123724 0.6123724 0.5
 endloop
endfacet
facet normal 0.28108463 0.67859834 -0.67859834
 outer loop
  vertex 0 0.49999994 -0.8660254
  vertex 0 0.8660253 -0.5
  vertex 0.6123724 0.6123724 -0.5
 endloop
endfacet
facet normal 0.2810846 0.67859846 -0.6785983
 outer loop
  vertex 0 0.49999994 -0.8660254
  vertex 0.6123724 0.6123724 -0.5
  vertex 0.35355338 0.35355338 -0.8660254
 endloop
endfacet
facet normal 0.25734445 0.106595546 -0.96042234
 outer loop
  vertex 0 0 -1
  vertex 0.35355338 0.35355338 -0.8660254
  vertex 0.49999997 0 -0.8660254
 endloop
endfacet
facet normal 0.8968088 0.37147033 -0.24029925
 outer loop
  vertex 0.6123724 0.6123724 -0.5
  vertex 0.70710677 0.70710677 0
  vertex 0.99999994 0 0
 endloop
endfacet
facet normal 0.89680886 0.37147036 -0.24029922
 outer loop
  vertex 0.6123724 0.6123724 -0.5
  vertex 0.99999994 0 0
  vertex 0.8660253 0 -0.5
 endloop
endfacet
facet normal 0.67859846 0.28108463 0.6785983
 outer loop
  vertex 0.6123724 0.6123724 0.5
  vertex 0.35355338 0.35355338 0.8660254
  vertex 0.49999997 0 0.8660254
 endloop
endfacet
facet normal 0.6785984 0.28108463 0.67859834
 outer loop
  vertex 0.6123724 0.6123724 0.5
  vertex 0.49999997 0 0.8660254
  vertex 0.8660253 0 0.5
 endloop
endfacet
facet normal 0.6785984 0.28108463 -0.67859834
 outer loop
  vertex 0.35355338 0.35355338 -0.8660254
  vertex 0.6123724 0.6123724 -0.5
  vertex 0.8660253 0 -0.5
 endloop
endfacet
facet normal 0.6785984 0.28108454 -0.67859834
 outer loop
  vertex 0.35355338 0.35355338 -0.8660254
  vertex 0.8660253 0 -0.5
  vertex 0.49999997 0 -0.8660254
 endloop
endfacet
facet normal 0.89680886 0.37147036 0.24029927
 outer loop
  vertex 0.70710677 0.70710677 0
  vertex 0.6123724 0.6123724 0.5
  vertex 0.8660253 0 0.5
 endloop
endfacet
facet normal 0.8968088 0.37147033 0.24029922
 outer loop
  vertex 0.70710677 0.70710677 0
  vertex 0.8660253 0 0.5
  vertex 0.99999994 0 0
 endloop
endfacet
facet normal 0.25734445 0.10659552 0.96042234
 outer loop
  vertex 0.35355338 0.35355338 0.8660254
  vertex 0 0 1
  vertex 0.49999997 0 0.8660254
 endloop
endfacet
facet normal 0.25734448 -0.10659558 -0.96042246
 outer loop
  vertex 0 0 -1
  vertex 0.49999997 0 -0.8660254
  vertex 0.35355335 -0.35355335 -0.8660254
 endloop
endfacet
facet normal 0.8968088 -0.37147042 -0.24029922
 outer loop
  vertex 0.8660253 0 -0.5
  vertex 0.99999994 0 0
  vertex 0.7071067 -0.7071067 0
 endloop
endfacet
facet normal 0.89680886 -0.37147036 -0.24029912
 outer loop
  vertex 0.8660253 0 -0.5
  vertex 0.7071067 -0.7071067 0
  vertex 0.6123724 -0.6123724 -0.5
 endloop
endfacet
facet normal 0.6785984 -0.2810847 0.6785983
 outer loop
  vertex 0.8660253 0 0.5
  vertex 0.49999997 0 0.8660254
  vertex 0.35355335 -0.35355335 0.8660254
 endloop
endfacet
facet normal 0.6785984 -0.28108463 0.67859834
 outer loop
  vertex 0.8660253 0 0.5
  vertex 0.35355335 -0.35355335 0.8660254
  vertex 0.6123724 -0.6123724 0.5
 endloop
endfacet
facet normal 0.6785984 -0.28108463 -0.67859834
 outer loop
  vertex 0.49999997 0 -0.8660254
  vertex 0.8660253 0 -0.5
  vertex 0.6123724 -0.6123724 -0.5
 endloop
endfacet
facet normal 0.6785983 -0.28108466 -0.67859834
 outer loop
  vertex 0.49999997 0 -0.8660254
  vertex 0.6123724 -0.6123724 -0.5
  vertex 0.35355335 -0.35355335 -0.8660254
 endloop
endfacet
facet normal 0.89680886 -0.37147036 0.24029922
 outer loop
  vertex 0.99999994 0 0
  vertex 0.8660253 0 0.5
  vertex 0.6123724 -0.6123724 0.5
 endloop
endfacet
facet normal 0.8968088 -0.37147042 0.24029912
 outer loop
  vertex 0.99999994 0 0
  vertex 0.6123724 -0.6123724 0.5
  vertex 0.7071067 -0.7071067 0
 endloop
endfacet
facet normal 0.25734448 -0.10659556 0.96042246
 outer loop
  vertex 0.49999997 0 0.8660254
  vertex 0 0 1
  vertex 0.35355335 -0.35355335 0.8660254
 endloop
endfacet
facet normal 0.37147036 -0.8968088 -0.24029912
 outer loop
  vertex 0.6123724 -0.6123724 -0.5
  vertex 0.7071067 -0.7071067 0
  vertex 0 -0.9999999 0
 endloop
endfacet
facet normal 0.37147036 -0.89680886 -0.24029912
 outer loop
  vertex 0.6123724 -0.6123724 -0.5
  vertex 0 -0.9999999 0
  vertex 0 -0.8660253 -0.5
 endloop
endfacet
facet normal 0.28108463 -0.6785984 0.6785984
 outer loop
  vertex 0.6123724 -0.6123724 0.5
  vertex 0.35355335 -0.35355335 0.8660254
  vertex 0 -0.49999994 0.8660254
 endloop
endfacet
facet normal 0.28108463 -0.67859834 0.67859834
 outer loop
  vertex 0.6123724 -0.6123724 0.5
  vertex 0 -0.49999994 0.8660254
  vertex 0 -0.8660253 0.5
 endloop
endfacet
facet normal 0.28108463 -0.67859834 -0.67859834
 outer loop
  vertex 0.35355335 -0.35355335 -0.8660254
  vertex 0.6123724 -0.6123724 -0.5
  vertex 0 -0.8660253 -0.5
 endloop
endfacet
facet normal 0.28108466 -0.6785984 -0.6785984
 outer loop
  vertex 0.35355335 -0.35355335 -0.8660254
  vertex 0 -0.8660253 -0.5
  vertex 0 -0.49999994 -0.8660254
 endloop
endfacet
facet normal 0.37147036 -0.89680886 0.24029909
 outer loop
  vertex 0.7071067 -0.7071067 0
  vertex 0.6123724 -0.6123724 0.5
  vertex 0 -0.8660253 0.5
 endloop
endfacet
facet normal 0.37147036 -0.8968088 0.24029912
 outer loop
  vertex 0.7071067 -0.7071067 0
  vertex 0 -0.8660253 0.5
  vertex 0 -0.9999999 0
 endloop
endfacet
facet normal 0.10659556 -0.25734448 0.96042246
 outer loop
  vertex 0.35355335 -0.35355335 0.8660254
  vertex 0 0 1
  vertex 0 -0.49999994 0.8660254
 endloop
endfacet
facet normal 0.10659556 -0.25734448 -0.96042246
 outer loop
  vertex 0 0 -1
  vertex 0.35355335 -0.35355335 -0.8660254
  vertex 0 -0.49999994 -0.8660254
 endloop
endfacet
facet normal -0.28108463 -0.6785984 0.6785984
 outer loop
  vertex 0 -0.8660253 0.5
  vertex 0 -0.49999994 0.8660254
  vertex -0.35355335 -0.35355335 0.8660254
 endloop
endfacet
facet normal -0.28108463 -0.6785984 0.67859834
 outer loop
  vertex 0 -0.8660253 0.5
  vertex -0.35355335 -0.35355335 0.8660254
  vertex -0.6123724 -0.6123724 0.5
 endloop
endfacet
facet normal -0.28108463 -0.67859834 -0.67859834
 outer loop
  vertex 0 -0.49999994 -0.8660254
  vertex 0 -0.8660253 -0.5
  vertex -0.6123724 -0.6123724 -0.5
 endloop
endfacet
facet normal -0.28108463 -0.6785984 -0.6785984
 outer loop
  vertex 0 -0.49999994 -0.8660254
  vertex -0.6123724 -0.6123724 -0.5
  vertex -0.35355335 -0.35355335 -0.8660254
 endloop
endfacet
facet normal -0.37147036 -0.89680886 0.24029912
 outer loop
  vertex 0 -0.9999999 0
  vertex 0 -0.8660253 0.5
  vertex -0.6123724 -0.6123724 0.5
 endloop
endfacet
facet normal -0.37147036 -0.8968088 0.24029909
 outer loop
  vertex 0 -0.9999999 0
  vertex -0.6123724 -0.6123724 0.5
  vertex -0.7071067 -0.7071067 0
 endloop
endfacet
facet normal -0.10659556 -0.25734448 0.96042246
 outer loop
  vertex 0 -0.49999994 0.8660254
  vertex 0 0 1
  vertex -0.35355335 -0.35355335 0.8660254
 endloop
endfacet
facet normal -0.10659556 -0.25734448 -0.96042246
 outer loop
  vertex 0 0 -1
  vertex 0 -0.49999994 -0.8660254
  vertex -0.35355335 -0.35355335 -0.8660254
 endloop
endfacet
facet normal -0.37147036 -0.8968088 -0.24029912
 outer loop
  vertex 0 -0.8660253 -0.5
  vertex 0 -0.9999999 0
  vertex -0.7071067 -0.7071067 0
 endloop
endfacet
facet normal -0.37147036 -0.89680886 -0.24029912
 outer loop
  vertex 0 -0.8660253 -0.5
  vertex -0.7071067 -0.7071067 0
  vertex -0.6123724 -0.6123724 -0.5
 endloop
endfacet
facet normal -0.67859834 -0.28108463 -0.67859834
 outer loop
  vertex -0.35355335 -0.35355335 -0.8660254
  vertex -0.6123724 -0.6123724 -0.5
  vertex -0.8660253 0 -0.5
 endloop
endfacet
facet normal -0.6785984 -0.28108466 -0.6785984
 outer loop
  vertex -0.35355335 -0.35355335 -0.8660254
  vertex -0.8660253 0 -0.5
  vertex -0.49999994 0 -0.8660254
 endloop
endfacet
facet normal -0.89680886 -0.37147036 0.24029909
 outer loop
  vertex -0.7071067 -0.7071067 0
  vertex -0.6123724 -0.6123724 0.5
  vertex -0.8660253 0 0.5
 endloop
endfacet
facet normal -0.8968088 -0.37147036 0.24029912
 outer loop
  vertex -0.7071067 -0.7071067 0
  vertex -0.8660253 0 0.5
  vertex -0.9999999 0 0
 endloop
endfacet
facet normal -0.25734448 -0.10659556 0.96042246
 outer loop
  vertex -0.35355335 -0.35355335 0.8660254
  vertex 0 0 1
  vertex -0.49999994 0 0.8660254
 endloop
endfacet
facet normal -0.25734448 -0.10659556 -0.96042246
 outer loop
  vertex 0 0 -1
  vertex -0.35355335 -0.35355335 -0.8660254
  vertex -0.49999994 0 -0.8660254
 endloop
endfacet
facet normal -0.8968088 -0.37147036 -0.24029912
 outer loop
  vertex -0.6123724 -0.6123724 -0.5
  vertex -0.7071067 -0.7071067 0
  vertex -0.9999999 0 0
 endloop
endfacet
facet normal -0.89680886 -0.37147036 -0.24029912
 outer loop
  vertex -0.6123724 -0.6123724 -0.5
  vertex -0.9999999 0 0
  vertex -0.8660253 0 -0.5
 endloop
endfacet
facet normal -0.6785984 -0.28108463 0.6785984
 outer loop
  vertex -0.6123724 -0.6123724 0.5
  vertex -0.35355335 -0.35355335 0.8660254
  vertex -0.49999994 0 0.8660254
 endloop
endfacet
facet normal -0.67859834 -0.28108463 0.67859834
 outer loop
  vertex -0.6123724 -0.6123724 0.5
  vertex -0.49999994 0 0.8660254
  vertex -0.8660253 0 0.5
 endloop
endfacet
facet normal -0.67859834 0.28108463 -0.67859834
 outer loop
  vertex -0.49999994 0 -0.8660254
  vertex -0.8660253 0 -0.5
  vertex -0.6123724 0.6123724 -0.5
 endloop
endfacet
facet normal -0.6785984 0.28108463 -0.6785984
 outer loop
  vertex -0.49999994 0 -0.8660254
  vertex -0.6123724 0.6123724 -0.5
  vertex -0.35355335 0.35355335 -0.8660254
 endloop
endfacet
facet normal -0.89680886 0.37147036 0.24029912
 outer loop
  vertex -0.9999999 0 0
  vertex -0.8660253 0 0.5
  vertex -0.6123724 0.6123724 0.5
 endloop
endfacet
facet normal -0.8968088 0.37147036 0.24029909
 outer loop
  vertex -0.9999999 0 0
  vertex -0.6123724 0.6123724 0.5
  vertex -0.7071067 0.7071067 0
 endloop
endfacet
facet normal -0.25734448 0.10659556 0.96042246
 outer loop
  vertex -0.49999994 0 0.8660254
  vertex 0 0 1
  vertex -0.35355335 0.35355335 0.8660254
 endloop
endfacet
facet normal -0.25734448 0.10659556 -0.96042246
 outer loop
  vertex 0 0 -1
  vertex -0.49999994 0 -0.8660254
  vertex -0.35355335 0.35355335 -0.8660254
 endloop
endfacet
facet normal -0.8968088 0.37147036 -0.24029912
 outer loop
  vertex -0.8660253 0 -0.5
  vertex -0.9999999 0 0
  vertex -0.7071067 0.7071067 0
 endloop
endfacet
facet normal -0.89680886 0.37147036 -0.24029912
 outer loop
  vertex -0.8660253 0 -0.5
  vertex -0.7071067 0.7071067 0
  vertex -0.6123724 0.6123724 -0.5
 endloop
endfacet
facet normal -0.6785984 0.28108463 0.6785984
 outer loop
  vertex -0.8660253 0 0.5
  vertex -0.49999994 0 0.8660254
  vertex -0.35355335 0.35355335 0.8660254
 endloop
endfacet
facet normal -0.6785984 0.28108463 0.67859834
 outer loop
  vertex -0.8660253 0 0.5
  vertex -0.35355335 0.35355335 0.8660254
  vertex -0.6123724 0.6123724 0.5
 endloop
endfacet
facet normal -0.28108463 0.67859834 -0.67859834
 outer loop
  vertex -0.35355335 0.35355335 -0.8660254
  vertex -0.6123724 0.6123724 -0.5
  vertex 0 0.8660253 -0.5
 endloop
endfacet
facet normal -0.28108466 0.6785984 -0.6785984
 outer loop
  vertex -0.35355335 0.35355335 -0.8660254
  vertex 0 0.8660253 -0.5
  vertex 0 0.49999994 -0.8660254
 endloop
endfacet
facet normal -0.37147036 0.89680886 0.24029909
 outer loop
  vertex -0.7071067 0.7071067 0
  vertex -0.6123724 0.6123724 0.5
  vertex 0 0.8660253 0.5
 endloop
endfacet
facet normal -0.37147036 0.8968088 0.24029912
 outer loop
  vertex -0.7071067 0.7071067 0
  vertex 0 0.8660253 0.5
  vertex 0 0.9999999 0
 endloop
endfacet
facet normal -0.10659556 0.25734448 0.96042246
 outer loop
  vertex -0.35355335 0.35355335 0.8660254
  vertex 0 0 1
  vertex 0 0.49999994 0.8660254
 endloop
endfacet
facet normal -0.10659556 0.25734448 -0.96042246
 outer loop
  vertex 0 0 -1
  vertex -0.35355335 0.35355335 -0.8660254
  vertex 0 0.49999994 -0.8660254
 endloop
endfacet
facet normal -0.37147036 0.8968088 -0.24029912
 outer loop
  vertex -0.6123724 0.6123724 -0.5
  vertex -0.7071067 0.7071067 0
  vertex 0 0.9999999 0
 endloop
endfacet
facet normal -0.37147036 0.89680886 -0.24029912
 outer loop
  vertex -0.6123724 0.6123724 -0.5
  vertex 0 0.9999999 0
  vertex 0 0.8660253 -0.5
 endloop
endfacet
facet normal -0.28108463 0.6785984 0.6785984
 outer loop
  vertex -0.6123724 0.6123724 0.5
  vertex -0.35355335 0.35355335 0.8660254
  vertex 0 0.49999994 0.8660254
 endloop
endfacet
facet normal -0.28108463 0.67859834 0.67859834
 outer loop
  vertex -0.6123724 0.6123724 0.5
  vertex 0 0.49999994 0.8660254
  vertex 0 0.8660253 0.5
 endloop
endfacet
endsolid uvsphere
"""

    stl_file_path = tmp_path / "uv_sphere.stl"
    stl_file_path.write_text(stl_content)
    return stl_file_path


@pytest.fixture(scope="session")
def stl_ico_sphere(tmp_path_factory):
    """Generate a STL file for a ico sphere with radius 1."""
    tmp_path = tmp_path_factory.mktemp("stl_data")
    stl_content = """solid icosphere
facet normal 0.18759656 -0.57735366 -0.7946511
 outer loop
  vertex 0 0 -1
  vertex 0.7236 -0.52572 -0.447215
  vertex -0.276385 -0.85064 -0.447215
 endloop
endfacet
facet normal 0.6070647 0 -0.7946524
 outer loop
  vertex 0.7236 -0.52572 -0.447215
  vertex 0 0 -1
  vertex 0.7236 0.52572 -0.447215
 endloop
endfacet
facet normal -0.49112207 -0.35682905 -0.7946522
 outer loop
  vertex 0 0 -1
  vertex -0.276385 -0.85064 -0.447215
  vertex -0.894425 0 -0.447215
 endloop
endfacet
facet normal -0.49112207 0.35682905 -0.7946522
 outer loop
  vertex 0 0 -1
  vertex -0.894425 0 -0.447215
  vertex -0.276385 0.85064 -0.447215
 endloop
endfacet
facet normal 0.18759656 0.57735366 -0.7946511
 outer loop
  vertex 0 0 -1
  vertex -0.276385 0.85064 -0.447215
  vertex 0.7236 0.52572 -0.447215
 endloop
endfacet
facet normal 0.9822461 0 -0.1875968
 outer loop
  vertex 0.7236 -0.52572 -0.447215
  vertex 0.7236 0.52572 -0.447215
  vertex 0.894425 0 0.447215
 endloop
endfacet
facet normal 0.30353555 -0.93417156 -0.18758914
 outer loop
  vertex -0.276385 -0.85064 -0.447215
  vertex 0.7236 -0.52572 -0.447215
  vertex 0.276385 -0.85064 0.447215
 endloop
endfacet
facet normal -0.7946492 -0.5773594 -0.18758695
 outer loop
  vertex -0.894425 0 -0.447215
  vertex -0.276385 -0.85064 -0.447215
  vertex -0.7236 -0.52572 0.447215
 endloop
endfacet
facet normal -0.7946492 0.5773594 -0.18758696
 outer loop
  vertex -0.276385 0.85064 -0.447215
  vertex -0.894425 0 -0.447215
  vertex -0.7236 0.52572 0.447215
 endloop
endfacet
facet normal 0.30353555 0.93417156 -0.18758915
 outer loop
  vertex 0.7236 0.52572 -0.447215
  vertex -0.276385 0.85064 -0.447215
  vertex 0.276385 0.85064 0.447215
 endloop
endfacet
facet normal 0.7946492 -0.5773594 0.18758696
 outer loop
  vertex 0.7236 -0.52572 -0.447215
  vertex 0.894425 0 0.447215
  vertex 0.276385 -0.85064 0.447215
 endloop
endfacet
facet normal -0.30353555 -0.93417156 0.18758915
 outer loop
  vertex -0.276385 -0.85064 -0.447215
  vertex 0.276385 -0.85064 0.447215
  vertex -0.7236 -0.52572 0.447215
 endloop
endfacet
facet normal -0.9822461 0 0.1875968
 outer loop
  vertex -0.894425 0 -0.447215
  vertex -0.7236 -0.52572 0.447215
  vertex -0.7236 0.52572 0.447215
 endloop
endfacet
facet normal -0.30353555 0.93417156 0.18758914
 outer loop
  vertex -0.276385 0.85064 -0.447215
  vertex -0.7236 0.52572 0.447215
  vertex 0.276385 0.85064 0.447215
 endloop
endfacet
facet normal 0.7946492 0.5773594 0.18758695
 outer loop
  vertex 0.7236 0.52572 -0.447215
  vertex 0.276385 0.85064 0.447215
  vertex 0.894425 0 0.447215
 endloop
endfacet
facet normal 0.49112207 -0.35682905 0.7946522
 outer loop
  vertex 0.276385 -0.85064 0.447215
  vertex 0.894425 0 0.447215
  vertex 0 0 1
 endloop
endfacet
facet normal -0.18759656 -0.57735366 0.7946511
 outer loop
  vertex -0.7236 -0.52572 0.447215
  vertex 0.276385 -0.85064 0.447215
  vertex 0 0 1
 endloop
endfacet
facet normal -0.6070647 0 0.7946524
 outer loop
  vertex -0.7236 0.52572 0.447215
  vertex -0.7236 -0.52572 0.447215
  vertex 0 0 1
 endloop
endfacet
facet normal -0.18759656 0.57735366 0.7946511
 outer loop
  vertex 0.276385 0.85064 0.447215
  vertex -0.7236 0.52572 0.447215
  vertex 0 0 1
 endloop
endfacet
facet normal 0.49112207 0.35682905 0.7946522
 outer loop
  vertex 0.894425 0 0.447215
  vertex 0.276385 0.85064 0.447215
  vertex 0 0 1
 endloop
endfacet
endsolid icosphere
"""
    stl_file_path = tmp_path / "ico_sphere.stl"
    stl_file_path.write_text(stl_content)
    return stl_file_path
