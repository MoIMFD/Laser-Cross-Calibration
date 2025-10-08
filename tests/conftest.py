import numpy as np
import pytest

from laser_cross_calibration import materials, sources, surfaces, tracing

@pytest.fixture
def air():
    return materials.ConstantMaterial(name="air", ior=1.0)


@pytest.fixture
def water():
    return materials.ConstantMaterial(name="water", ior=1.333)


@pytest.fixture
def glass_bk7():
    return materials.GLASS_BK7


@pytest.fixture
def simple_ray():
    return tracing.OpticalRay(
        origin=np.array([0.0, 0.0, 0.0]),
        direction=np.array([0.0, 0.0, 1.0]),
        # wavelength=632.8e-9,
    )


@pytest.fixture
def ray_at_angle():
    direction = np.array([1.0, 0.0, 1.0])
    direction = direction / np.linalg.norm(direction)
    return tracing.OpticalRay(
        origin=np.array([0.0, 0.0, 0.0]),
        direction=direction,
        # wavelength=632.8e-9,
    )

@pytest.fixture
def create_horizontal_single_triangle_surface() -> surfaces.TriSurface:
    """Single triangle in XY plane at Z=0"""
    vertices = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    return surfaces.TriSurface(vertices=vertices, faces=faces)

@pytest.fixture
def create_vertical_single_triangle_surface() -> surfaces.TriSurface:
    """Single triangle in YZ plane at X=0"""
    vertices = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    faces = np.asarray([[0, 1, 2]])
    return surfaces.TriSurface(vertices=vertices, faces=faces)

@pytest.fixture
def create_triangle_box_surface() -> surfaces.TriSurface:
      """Simple box (12 triangles, 2 per face) for testing multi-surface intersections."""
      vertices = np.asarray([
          # Bottom face (Z=0)
          [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
          # Top face (Z=1)
          [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
      ])
      faces = np.asarray([
          # Bottom (Z=0)
          [0, 1, 2], [0, 2, 3],
          # Top (Z=1)
          [4, 6, 5], [4, 7, 6],
          # Front (Y=0)
          [0, 5, 1], [0, 4, 5],
          # Back (Y=1)
          [2, 7, 3], [2, 6, 7],
          # Left (X=0)
          [0, 3, 7], [0, 7, 4],
          # Right (X=1)
          [1, 5, 6], [1, 6, 2],
      ])
      return surfaces.TriSurface(vertices=vertices, faces=faces)