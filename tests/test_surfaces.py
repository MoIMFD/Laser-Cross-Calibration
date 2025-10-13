from __future__ import annotations

from math import isclose

import numpy as np
import pytest

from laser_cross_calibration.constants import (
    ORIGIN_POINT3,
    UNIT_X_VECTOR3,
    UNIT_Y_VECTOR3,
    UNIT_Z_VECTOR3,
    VSMALL,
)
from laser_cross_calibration.surfaces import (
    IntersectionResult,
    Plane,
    Surface,
    TriSurface,
)
from laser_cross_calibration.tracing.ray import OpticalRay
from tests.utils import assert_vectors_close


@pytest.mark.unit
class TestSurface:
    def test_counter(self):
        for i in range(10):
            assert Surface._id_counter == i

            plane = Plane(point=ORIGIN_POINT3, normal=UNIT_X_VECTOR3)
            assert Surface._id_counter == i + 1
            assert plane.surface_id == Surface._id_counter - 1

        plane = Plane(point=ORIGIN_POINT3, normal=UNIT_X_VECTOR3, surface_id=100)

        assert plane.surface_id == 100
        assert Surface._id_counter == 101

        # test wrong surface id type
        wrong_types = ["some string", 1.234, 1 + 2j, [1, 2, 3], {"some": 1, "dict": 2}]

        for wrong_arg in wrong_types:
            with pytest.raises(ValueError, match="Surface id must be type int"):
                plane = Plane(
                    point=ORIGIN_POINT3, normal=UNIT_X_VECTOR3, surface_id=wrong_arg
                )

        # test negative ints
        with pytest.raises(ValueError, match="Surface id must be non-negative"):
            plane = Plane(point=ORIGIN_POINT3, normal=UNIT_X_VECTOR3, surface_id=-1)


@pytest.mark.unit
class TestTriangleSurface:
    def test_creation(self):
        tri_surface = TriSurface(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            faces=np.array([[0, 1, 2]]),
        )

        assert isinstance(tri_surface, TriSurface | Surface)
        assert tri_surface.vertices.shape == (3, 3)
        assert tri_surface.faces.shape == (1, 3)
        assert len(tri_surface.triangle_normals) == 1

    def test_normal_calculation(
        self,
        single_triangle_surface_xy: TriSurface,
        single_triangle_surface_yz: TriSurface,
    ):
        # check horizontal tri element with vertical normal
        normal = single_triangle_surface_xy.triangle_normals[0]
        assert_vectors_close(normal, (0, 0, 1))
        assert isclose(np.linalg.norm(normal), 1.0)

        # check vertical tri element with horizontal normal
        normal = single_triangle_surface_yz.triangle_normals[0]
        assert_vectors_close(normal, (1, 0, 0))
        assert isclose(np.linalg.norm(normal), 1.0)

    def test_degenerate_triangle_handling(self):
        """Degenerate triangle (all same point) shouldn't crash, but normal will be
        zero.
        """
        tri_surface = TriSurface(
            vertices=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            faces=np.array([[0, 1, 2]]),
        )
        normal = tri_surface.triangle_normals[0]
        # Should be zero or near-zero for degenerate triangle
        assert np.linalg.norm(normal) < VSMALL

    def test_normal_direction_right_hand_rule(self):
        """Normal should follow right-hand rule: (v1-v0) x (v2-v0)."""
        vertices = np.array(
            [
                [0, 0, 0],  # v0
                [1, 0, 0],  # v1
                [0, 1, 0],  # v2
            ]
        )
        faces = np.array([[0, 1, 2]])
        tri_surface = TriSurface(vertices=vertices, faces=faces)

        normal_right = tri_surface.triangle_normals[0]
        # (v1-v0) = (1,0,0), (v2-v0) = (0,1,0), cross = (0,0,1)
        assert_vectors_close(normal_right, (0, 0, 1))

        faces = np.array([[2, 1, 0]])
        tri_surface = TriSurface(vertices=vertices, faces=faces)

        normal_left = tri_surface.triangle_normals[0]
        # (v1-v0) = (1,0,0), (v2-v0) = (0,1,0), cross = (0,0,1)
        assert_vectors_close(normal_left, (0, 0, -1))

        # check if normals are opposed
        assert_vectors_close(normal_right, -normal_left)

    def test_mesh_validation_invalid_shape(self):
        """Invalid vertex/face shapes should raise ValueError."""
        with pytest.raises(ValueError, match="Vertices must have shape"):
            TriSurface(
                vertices=np.array([[0, 0]]),  # Only 2D
                faces=np.array([[0, 1, 2]]),
            )

        with pytest.raises(ValueError, match="Faces must have shape"):
            TriSurface(
                vertices=np.array([[0, 0, 0], [1, 0, 0]]),
                faces=np.array([[0, 1]]),  # Only 2 indices
            )

    def test_mesh_validation_invalid_indices(self):
        """Face indices out of bounds should raise ValueError."""
        with pytest.raises(ValueError, match="Face indices exceed"):
            TriSurface(
                vertices=np.array([[0, 0, 0], [1, 0, 0]]),
                faces=np.array([[0, 1, 5]]),  # Index 5 doesn't exist
            )

    def test_scale_transformation(self, single_triangle_surface_yz: TriSurface):
        """Test scaling modifies vertices correctly."""
        surface = single_triangle_surface_yz
        original_vertices = surface.vertices.copy()

        surface.scale(x=2.0, z=0.5)

        assert_vectors_close(surface.vertices[:, 0], original_vertices[:, 0] * 2.0)
        assert_vectors_close(surface.vertices[:, 2], original_vertices[:, 2] * 0.5)

    def test_translate_transformation(self, single_triangle_surface_yz: TriSurface):
        """Test translation modifies vertices correctly."""
        surface = single_triangle_surface_yz
        original_vertices = surface.vertices.copy()

        surface.translate(x=1.0, y=2.0, z=3.0)

        expected = original_vertices + np.array([1.0, 2.0, 3.0])
        assert_vectors_close(surface.vertices, expected)

    def test_transformation_chaining(self):
        """Test that transformations can be chained."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        surface = TriSurface(vertices=vertices, faces=np.array([[0, 1, 2]]))

        surface.scale(x=2.0).translate(z=5.0)

        assert_vectors_close(surface.vertices[0], [0, 0, 5])
        assert_vectors_close(surface.vertices[1], [2, 0, 5])

    def test_bounding_box(self):
        """Test bounding box calculation."""
        vertices = np.array(
            [
                [0, 0, 0],
                [2, 3, 1],
                [-1, 1, 5],
            ]
        )
        surface = TriSurface(vertices=vertices, faces=np.array([[0, 1, 2]]))

        min_bounds, max_bounds = surface.get_bounds()

        assert_vectors_close(min_bounds, [-1, 0, 0])
        assert_vectors_close(max_bounds, [2, 3, 5])

    def test_smooth_vs_flat_shading(self):
        """Verify smooth=True creates vertex normals, smooth=False doesn't."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])

        smooth_surface = TriSurface(vertices=vertices, faces=faces, is_smooth=True)
        flat_surface = TriSurface(vertices=vertices, faces=faces, is_smooth=False)

        assert smooth_surface.vertex_normals is not None
        assert flat_surface.vertex_normals is None

    def test_vertex_normals_for_smooth_surface(self):
        """Verify vertex normals are computed for smooth shading."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])

        smooth_surface = TriSurface(vertices=vertices, faces=faces, is_smooth=True)

        assert smooth_surface.vertex_normals is not None
        assert smooth_surface.vertex_normals.shape == (3, 3)  # 3 vertices, 3D normals
        # For single triangle, all vertex normals should equal triangle normal
        for vertex_normal in smooth_surface.vertex_normals:
            assert_vectors_close(vertex_normal, smooth_surface.triangle_normals[0])

    def test_vertex_normals_shared_vertices(self):
        """Vertex normals should average adjacent triangle normals."""
        # Two triangles forming a right-angled corner
        vertices = np.array(
            [
                [0, 0, 0],  # Shared edge vertices
                [1, 0, 0],
                [0.5, 1, 0],  # Top of first triangle (horizontal)
                [0.5, 0, 1],  # Top of second triangle (vertical)
            ]
        )
        faces = np.array(
            [
                [0, 1, 2],  # Horizontal triangle (normal pointing up)
                [0, 1, 3],  # Vertical triangle (normal pointing forward)
            ]
        )

        smooth_surface = TriSurface(vertices=vertices, faces=faces, is_smooth=True)

        assert smooth_surface.vertex_normals is not None
        # Vertices 0 and 1 are shared - their normals should be averaged
        assert smooth_surface.vertex_normals[0].shape == (3,)
        # Should not be zero (averaging two different triangle normals)
        assert np.linalg.norm(smooth_surface.vertex_normals[0]) > 0.5


@pytest.mark.integration
class TestTriangleRayInteraction:
    def test_valid_triangle_ray_intersection(
        self,
        simple_ray_100: OpticalRay,
        simple_ray_010: OpticalRay,
        simple_ray_001: OpticalRay,
        single_triangle_surface_yz: TriSurface,
        single_triangle_surface_xz: TriSurface,
        single_triangle_surface_xy: TriSurface,
    ):
        """Test the intersection logic between triangles and rays. For a correct
        passing of the surface_id test, the order of the fixtures in the function
        declaration is impotatnt! First fixture gets 0, second fixture 1 and third
        fixture 2. If the order is changed, the assertions must be adapted accordingly.
        """
        surface = single_triangle_surface_yz
        ray = simple_ray_100

        x_offset = 1
        for i in range(1, 10):
            surface.translate(x=x_offset)

            intersection = surface.intersect(ray)

            assert isinstance(intersection, IntersectionResult)
            assert intersection.hit is True

            assert isinstance(intersection.point, np.ndarray)
            assert_vectors_close(intersection.point, (i * x_offset, 0, 0))

            assert isinstance(intersection.normal, np.ndarray)
            assert_vectors_close(intersection.normal, (1, 0, 0))

            assert isinstance(intersection.triangle_id, int)
            assert intersection.triangle_id == 0

            assert isinstance(intersection.surface_id, int)
            assert intersection.surface_id == 0

            # a visual validation for the barycentric coordinates can be found
            # at: https://en.wikipedia.org/wiki/Barycentric_coordinate_system
            assert isclose(intersection.barycentric_u, 0.5)
            assert isclose(intersection.barycentric_v, 0.25)
            assert isclose(intersection.barycentric_w, 0.25)

        surface = single_triangle_surface_xz
        ray = simple_ray_010

        y_offset = 1
        for i in range(1, 10):
            surface.translate(y=y_offset)

            intersection = surface.intersect(ray)

            assert isinstance(intersection, IntersectionResult)
            assert intersection.hit is True

            assert isinstance(intersection.point, np.ndarray)
            assert_vectors_close(intersection.point, (0, i * y_offset, 0))

            assert isinstance(intersection.normal, np.ndarray)
            assert_vectors_close(intersection.normal, (0, 1, 0))

            assert isinstance(intersection.triangle_id, int)
            assert intersection.triangle_id == 0

            assert isinstance(intersection.surface_id, int)
            assert intersection.surface_id == 1

            # a visual validation for the barycentric coordinates can be found
            # at: https://en.wikipedia.org/wiki/Barycentric_coordinate_system
            assert isclose(intersection.barycentric_u, 0.25)
            assert isclose(intersection.barycentric_v, 0.25)
            assert isclose(intersection.barycentric_w, 0.5)

        surface = single_triangle_surface_xy
        ray = simple_ray_001

        z_offset = 1
        for i in range(1, 10):
            surface.translate(z=z_offset)

            intersection = surface.intersect(ray)

            assert isinstance(intersection, IntersectionResult)
            assert intersection.hit is True

            assert isinstance(intersection.point, np.ndarray)
            assert_vectors_close(intersection.point, (0, 0, i * z_offset))

            assert isinstance(intersection.normal, np.ndarray)
            assert_vectors_close(intersection.normal, (0, 0, 1))

            assert isinstance(intersection.triangle_id, int)
            assert intersection.triangle_id == 0

            assert isinstance(intersection.surface_id, int)
            assert intersection.surface_id == 2

            # a visual validation for the barycentric coordinates can be found
            # at: https://en.wikipedia.org/wiki/Barycentric_coordinate_system
            assert isclose(intersection.barycentric_u, 0.25)
            assert isclose(intersection.barycentric_v, 0.5)
            assert isclose(intersection.barycentric_w, 0.25)


@pytest.mark.unit
class TestTriSurfaceStlHandling:
    def test_load_stl(self, stl_cube: str, stl_uv_sphere: str, stl_ico_sphere: str):
        # test the cube
        surface = TriSurface.from_stl_file(stl_path=stl_cube)

        assert isinstance(surface, Surface | TriSurface)

        bounds = surface.get_bounds()
        assert_vectors_close(bounds[0], (-1, -1, -1))
        assert_vectors_close(bounds[1], (1, 1, 1))

        assert len(surface.faces) == 12

        # check if surface normals match unit vectors, 2 per cube side
        expected_normals = [
            (UNIT_X_VECTOR3, 2),  # +X face
            (-UNIT_X_VECTOR3, 2),  # -X face
            (UNIT_Y_VECTOR3, 2),  # +Y face
            (-UNIT_Y_VECTOR3, 2),  # -Y face
            (UNIT_Z_VECTOR3, 2),  # +Z face
            (-UNIT_Z_VECTOR3, 2),  # -Z face
        ]

        for expected_normal, expected_count in expected_normals:
            actual_count = sum(
                np.allclose(normal, expected_normal)
                for normal in surface.triangle_normals
            )
            assert actual_count == expected_count, (
                f"Expected {expected_count=} triangles with normal {expected_normal=}, "
                f"but found {actual_count}"
            )

        # test the uv sphere
        surface = TriSurface.from_stl_file(stl_path=stl_uv_sphere)

        assert isinstance(surface, Surface | TriSurface)

        bounds = surface.get_bounds()
        assert_vectors_close(bounds[0], (-1.0, -1.0, -1.0))
        assert_vectors_close(bounds[1], (1.0, 1.0, 1.0))

        # test the ico sphere
        surface = TriSurface.from_stl_file(stl_path=stl_ico_sphere)

        assert isinstance(surface, Surface | TriSurface)
        # only z dimension test here since ico spheres are compressed in the xy plane
        bounds = surface.get_bounds()
        assert isclose(bounds[0][2], -1)
        assert isclose(bounds[1][2], 1)

    def test_load_stl_invalid_path(self):
        with pytest.raises(FileNotFoundError, match="No STL file*"):
            _ = TriSurface.from_stl_file(stl_path="some/invalid/path.stl")


class TestPlaneSurface:
    def test_plane_yz(self):
        plane = Plane.create_yz()

        assert_vectors_close(plane.normal, UNIT_X_VECTOR3)

    def test_plane_ray_intersection(self):
        ray = OpticalRay.ray_x(origin=-1 * UNIT_X_VECTOR3)
        plane = Plane.create_yz()

        result = plane.intersect(ray)

        assert isinstance(result, IntersectionResult)
        assert result.hit is True
        assert isclose(result.distance, 1.0)
        assert_vectors_close(result.point, ORIGIN_POINT3)
        assert_vectors_close(result.normal, -UNIT_X_VECTOR3)
        assert result.surface_id == plane.surface_id

        ray = OpticalRay.ray_y(origin=-2 * UNIT_Y_VECTOR3)
        plane = Plane.create_xz()

        result = plane.intersect(ray)

        assert isinstance(result, IntersectionResult)
        assert result.hit is True
        assert isclose(result.distance, 2.0)
        assert_vectors_close(result.point, ORIGIN_POINT3)
        assert_vectors_close(result.normal, -UNIT_Y_VECTOR3)
        assert result.surface_id == plane.surface_id

        ray = OpticalRay.ray_z(origin=-3.5 * UNIT_Z_VECTOR3)
        plane = Plane.create_xy()

        result = plane.intersect(ray)

        assert isinstance(result, IntersectionResult)
        assert result.hit is True
        assert isclose(result.distance, 3.5)
        assert_vectors_close(result.point, ORIGIN_POINT3)
        assert_vectors_close(result.normal, -UNIT_Z_VECTOR3)
        assert result.surface_id == plane.surface_id
