import pytest

import numpy as np

from laser_cross_calibration.surfaces import TriSurface, Surface
from laser_cross_calibration.constants import VSMALL

@pytest.mark.unit
class TestTriangleSurface:
    def test_creation(self):
        tri_surface = TriSurface(
            vertices=np.array([[0, 0, 0], [1, 0, 0],[0, 1, 0]]), faces=np.array([[0, 1, 2]])
        )

        assert isinstance(tri_surface, (TriSurface, Surface))
        assert tri_surface.vertices.shape == (3, 3)
        assert tri_surface.faces.shape == (1, 3)
        assert len(tri_surface.triangle_normals) == 1

    def test_normal_calculation(self, create_horizontal_single_triangle_surface, create_vertical_single_triangle_surface):
        
        # check horizontal tri element with vertical normal
        normal = create_horizontal_single_triangle_surface.triangle_normals[0]
        assert np.allclose(normal, (0, 0, 1))
        assert np.isclose(np.linalg.norm(normal), 1.0)

        # check vertical tri element with horizontal normal
        normal = create_vertical_single_triangle_surface.triangle_normals[0]
        assert np.allclose(normal, (1, 0, 0))
        assert np.isclose(np.linalg.norm(normal), 1.0)


    def test_degenerate_triangle_handling(self):
        """Degenerate triangle (all same point) shouldn't crash, but normal will be zero."""
        tri_surface = TriSurface(
            vertices=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            faces=np.array([[0, 1, 2]])
        )
        normal = tri_surface.triangle_normals[0]
        # Should be zero or near-zero for degenerate triangle
        assert np.linalg.norm(normal) < VSMALL

    def test_normal_direction_right_hand_rule(self):
        """Normal should follow right-hand rule: (v1-v0) x (v2-v0)."""
        vertices = np.array([
            [0, 0, 0],  # v0
            [1, 0, 0],  # v1
            [0, 1, 0],  # v2
        ])
        faces = np.array([[0, 1, 2]])
        tri_surface = TriSurface(vertices=vertices, faces=faces)

        normal_right = tri_surface.triangle_normals[0]
        # (v1-v0) = (1,0,0), (v2-v0) = (0,1,0), cross = (0,0,1)
        assert np.allclose(normal_right, (0, 0, 1))

        faces = np.array([[2, 1, 0]])
        tri_surface = TriSurface(vertices=vertices, faces=faces)

        normal_left = tri_surface.triangle_normals[0]
        # (v1-v0) = (1,0,0), (v2-v0) = (0,1,0), cross = (0,0,1)
        assert np.allclose(normal_left, (0, 0, -1))

        # check if normals are opposed 
        assert np.allclose(normal_right, -normal_left)


    def test_mesh_validation_invalid_shape(self):
        """Invalid vertex/face shapes should raise ValueError."""
        with pytest.raises(ValueError, match="Vertices must have shape"):
            TriSurface(
                vertices=np.array([[0, 0]]),  # Only 2D
                faces=np.array([[0, 1, 2]])
            )

        with pytest.raises(ValueError, match="Faces must have shape"):
            TriSurface(
                vertices=np.array([[0, 0, 0], [1, 0, 0]]),
                faces=np.array([[0, 1]])  # Only 2 indices
            )

    def test_mesh_validation_invalid_indices(self):
        """Face indices out of bounds should raise ValueError."""
        with pytest.raises(ValueError, match="Face indices exceed"):
            TriSurface(
                vertices=np.array([[0, 0, 0], [1, 0, 0]]),
                faces=np.array([[0, 1, 5]])  # Index 5 doesn't exist
            )

    def test_scale_transformation(self, create_horizontal_single_triangle_surface):
        """Test scaling modifies vertices correctly."""
        surface = create_horizontal_single_triangle_surface
        original_vertices = surface.vertices.copy()

        surface.scale(scale_x=2.0, scale_z=0.5)

        assert np.allclose(surface.vertices[:, 0], original_vertices[:, 0] * 2.0)
        assert np.allclose(surface.vertices[:, 2], original_vertices[:, 2] * 0.5)

    def test_translate_transformation(self, create_horizontal_single_triangle_surface):
        """Test translation modifies vertices correctly."""
        surface = create_horizontal_single_triangle_surface
        original_vertices = surface.vertices.copy()

        surface.translate(translate_x=1.0, translate_y=2.0, translate_z=3.0)

        expected = original_vertices + np.array([1.0, 2.0, 3.0])
        assert np.allclose(surface.vertices, expected)

    def test_transformation_chaining(self):
        """Test that transformations can be chained."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        surface = TriSurface(vertices=vertices, faces=np.array([[0, 1, 2]]))

        surface.scale(x=2.0).translate(z=5.0)

        assert np.allclose(surface.vertices[0], [0, 0, 5])
        assert np.allclose(surface.vertices[1], [2, 0, 5])

    def test_bounding_box(self):
        """Test bounding box calculation."""
        vertices = np.array([
            [0, 0, 0],
            [2, 3, 1],
            [-1, 1, 5],
        ])
        surface = TriSurface(vertices=vertices, faces=np.array([[0, 1, 2]]))

        min_bounds, max_bounds = surface.get_bounds()

        assert np.allclose(min_bounds, [-1, 0, 0])
        assert np.allclose(max_bounds, [2, 3, 5])

    def test_smooth_vs_flat_shading(self):
          """Verify smooth=True creates vertex normals, smooth=False doesn't."""
          vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
          faces = np.array([[0, 1, 2]])

          smooth_surface = TriSurface(vertices=vertices, faces=faces, smooth=True)
          flat_surface = TriSurface(vertices=vertices, faces=faces, smooth=False)

          assert smooth_surface.vertex_normals is not None
          assert flat_surface.vertex_normals is None

    def test_vertex_normals_for_smooth_surface(self):
        """Verify vertex normals are computed for smooth shading."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])

        smooth_surface = TriSurface(vertices=vertices, faces=faces, smooth=True)

        assert smooth_surface.vertex_normals is not None
        assert smooth_surface.vertex_normals.shape == (3, 3)  # 3 vertices, 3D normals
        # For single triangle, all vertex normals should equal triangle normal
        for vertex_normal in smooth_surface.vertex_normals:
            assert np.allclose(vertex_normal, smooth_surface.triangle_normals[0])

    def test_vertex_normals_shared_vertices(self):
        """Vertex normals should average adjacent triangle normals."""
        # Two triangles forming a right-angled corner
        vertices = np.array([
            [0, 0, 0],  # Shared edge vertices
            [1, 0, 0],
            [0.5, 1, 0],  # Top of first triangle (horizontal)
            [0.5, 0, 1],  # Top of second triangle (vertical)
        ])
        faces = np.array([
            [0, 1, 2],  # Horizontal triangle (normal pointing up)
            [0, 1, 3],  # Vertical triangle (normal pointing forward)
        ])

        smooth_surface = TriSurface(vertices=vertices, faces=faces, smooth=True)

        assert smooth_surface.vertex_normals is not None
        # Vertices 0 and 1 are shared - their normals should be averaged
        assert smooth_surface.vertex_normals[0].shape == (3,)
        # Should not be zero (averaging two different triangle normals)
        assert np.linalg.norm(smooth_surface.vertex_normals[0]) > 0.5