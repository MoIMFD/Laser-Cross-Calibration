from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from stl import mesh as stl_mesh

from laser_cross_calibration.constants import VSMALL, VVSMALL
from laser_cross_calibration.surfaces.base import (
    IntersectionResult,
    Surface,
    get_surface_color,
)
from laser_cross_calibration.utils import normalize

if TYPE_CHECKING:
    import plotly.graph_objects as go
    from numpy.typing import NDArray

    from laser_cross_calibration.tracing.ray import OpticalRay
    from laser_cross_calibration.types import POINT3, VECTOR3


class TriSurface(Surface):
    """
    Base class for triangulated surfaces (STL, OBJ, PLY, etc.).

    Handles generic triangle mesh operations including ray-triangle intersection
    and visualization for any triangulated geometry.
    """

    def __init__(
        self,
        vertices: NDArray[np.floating],
        faces: NDArray[np.integer],
        is_smooth: bool = False,
        *,
        surface_id: int | None = None,
        **kwargs,
    ):
        """
        Initialize triangulated surface.

        Args:
            vertices: (N, 3) array of vertex coordinates
            faces: (M, 3) array of triangle vertex indices
            smooth: Whether to use smooth normal interpolation (True) or
                flat triangle normals (False)
        """
        super().__init__(surface_id=surface_id, **kwargs)

        # Store and validate mesh data
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=int)
        self.is_smooth = is_smooth
        self._validate_mesh()

        # Precompute triangle data for performance
        self.triangle_normals = self._compute_triangle_normals()

        # Compute vertex normals for smooth shading
        if self.is_smooth:
            self.vertex_normals = self._compute_vertex_normals()

        else:
            self.vertex_normals = None

    def scale(self, x: float = 1, y: float = 1, z: float = 1) -> TriSurface:
        self.vertices *= np.array((x, y, z))
        return self

    def translate(self, x: float = 0, y: float = 0, z: float = 0) -> TriSurface:
        self.vertices += np.array((x, y, z))
        return self

    def get_bounds(self) -> tuple[POINT3, POINT3]:
        """Get axis-aligned bounding box for this mesh."""
        min_bounds = np.min(self.vertices, axis=0)
        max_bounds = np.max(self.vertices, axis=0)
        return min_bounds, max_bounds

    def _validate_mesh(self):
        """Validate mesh geometry."""
        if len(self.vertices.shape) != 2:
            raise ValueError(
                f"Vertices must be two dimensional. Got {self.vertices.shape=}"
            )

        if self.vertices.shape[1] != 3:
            raise ValueError(
                f"Vertices must have shape (N, 3), got {self.vertices.shape}"
            )
        if self.faces.shape[1] != 3:
            raise ValueError(f"Faces must have shape (M, 3), got {self.faces.shape}")
        if self.faces.max() >= len(self.vertices):
            raise ValueError("Face indices exceed vertex array bounds")
        if self.faces.min() < 0:
            raise ValueError("Face indices must be non-negative")

    def _compute_triangle_normals(self) -> NDArray:
        """Compute normal vector for each triangle."""
        # Get triangle vertices
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        # Compute triangle edges
        edge1 = v1 - v0
        edge2 = v2 - v0

        # Cross product gives normal
        normals = np.cross(edge1, edge2)

        # Normalize
        norms = np.linalg.norm(normals, axis=1)
        # Avoid division by zero for degenerate triangles
        valid_mask = norms > VSMALL
        normals[valid_mask] = normals[valid_mask] / norms[valid_mask, np.newaxis]

        return normals

    def _compute_vertex_normals(self) -> np.ndarray:
        """
        Compute smooth vertex normals by averaging adjacent triangle normals.
        Uses area-weighted averaging for better quality (standard approach).
        """
        vertex_normals = np.zeros_like(self.vertices)

        # For each triangle, add its contribution to each vertex
        for i, face in enumerate(self.faces):
            triangle_normal = self.triangle_normals[i]

            # Area weighting: triangle area is 0.5 * |normal| before normalization
            v0, v1, v2 = self.vertices[face]
            triangle_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

            # Add area-weighted normal to each vertex
            for vertex_idx in face:
                vertex_normals[vertex_idx] += triangle_normal * triangle_area

        # Normalize vertex normals
        norms = np.linalg.norm(vertex_normals, axis=1)
        valid_mask = norms > VSMALL
        vertex_normals[valid_mask] = (
            vertex_normals[valid_mask] / norms[valid_mask, np.newaxis]
        )

        return vertex_normals

    def intersect(self, ray: OpticalRay) -> IntersectionResult:
        """
        Ray-triangle mesh intersection using Möller-Trumbore algorithm.
        Tests ray against all triangles and returns closest hit.
        """
        result = IntersectionResult()
        # closest_distance = np.inf

        # Ray origin and direction
        ray_origin = ray.position
        ray_dir = ray.current_direction

        # Get triangle vertices
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        # Möller-Trumbore algorithm (vectorized)
        edge1 = v1 - v0
        edge2 = v2 - v0

        # Begin calculating determinant - also used to calculate u parameter
        h = np.cross(ray_dir, edge2)
        det = np.sum(edge1 * h, axis=1)

        # If determinant is near zero, ray lies in plane of triangle
        valid_mask = np.abs(det) > VSMALL

        if not np.any(valid_mask):
            return result

        # Calculate distance from v0 to ray origin
        s = ray_origin - v0

        # Calculate u parameter and test bound
        u = np.sum(s * h, axis=1) / (det + VVSMALL)
        u_mask = (u >= 0.0) & (u <= 1.0)

        # Combine masks
        valid_mask = valid_mask & u_mask
        if not np.any(valid_mask):
            return result

        # Prepare to test v parameter
        q = np.cross(s, edge1)

        # Calculate v parameter and test bound
        v = np.sum(ray_dir * q, axis=1) / (det + VVSMALL)
        v_mask = (v >= 0.0) & (u + v <= 1.0)

        # Combine masks
        valid_mask = valid_mask & v_mask
        if not np.any(valid_mask):
            return result

        # Calculate t (distance along ray)
        t = np.sum(edge2 * q, axis=1) / (det + VVSMALL)
        t_mask = t > VSMALL  # Only forward intersections

        # Final valid hits
        valid_mask = valid_mask & t_mask
        if not np.any(valid_mask):
            return result

        # Find closest hit
        valid_distances = t[valid_mask]
        closest_idx = np.argmin(valid_distances)
        triangle_idx = np.where(valid_mask)[0][closest_idx]

        # Get barycentric coordinates from the intersection
        u_closest = u[triangle_idx]
        v_closest = v[triangle_idx]
        w_closest = 1.0 - u_closest - v_closest

        # Build intersection result
        result.hit = True
        result.distance = valid_distances[closest_idx]
        result.point = ray_origin + result.distance * ray_dir
        result.surface_id = self.surface_id
        result.triangle_id = triangle_idx
        result.barycentric_u = u_closest
        result.barycentric_v = v_closest
        result.barycentric_w = w_closest

        # Compute normal using smooth interpolation or flat triangle normal
        if self.is_smooth and self.vertex_normals is not None:
            face = self.faces[triangle_idx]
            result.normal = (
                w_closest * self.vertex_normals[face[0]]  # w corresponds to vertex 0
                + u_closest * self.vertex_normals[face[1]]  # u corresponds to vertex 1
                + v_closest * self.vertex_normals[face[2]]  # v corresponds to vertex 2
            )
            result.normal = normalize(result.normal)
        else:
            # Flat shading - use triangle normal
            result.normal = self.triangle_normals[triangle_idx]

        return result

    def to_plotly_surface(
        self, show_normals: bool = False
    ) -> list[go.Mesh3d] | list[go.Mesh3d | go.Scatter3d]:
        import plotly.graph_objects as go

        # Create triangle mesh
        mesh = go.Mesh3d(
            x=self.vertices[:, 0],
            y=self.vertices[:, 1],
            z=self.vertices[:, 2],
            i=self.faces[:, 0],
            j=self.faces[:, 1],
            k=self.faces[:, 2],
            opacity=0.2,
            color=get_surface_color(self.surface_id),
            name=f"Triangle Mesh {self.surface_id}",
            showscale=False,
        )

        if not show_normals:
            return [mesh]

        # Add triangle normal vectors (sample only some for clarity)
        triangle_centers = np.mean(self.vertices[self.faces], axis=1)
        n_samples = min(50, len(triangle_centers))  # Limit to avoid clutter
        sample_indices = np.linspace(0, len(triangle_centers) - 1, n_samples, dtype=int)

        centers = triangle_centers[sample_indices]
        normals = self.triangle_normals[sample_indices] * 0.1  # Scale for visibility

        normal_traces: list[go.Scatter3d] = []
        for _, (center, normal) in enumerate(zip(centers, normals, strict=False)):
            end_point = center + normal
            normal_traces.append(
                go.Scatter3d(
                    x=[center[0], end_point[0]],
                    y=[center[1], end_point[1]],
                    z=[center[2], end_point[2]],
                    mode="lines",
                    line=dict(width=2, color="red"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        return [mesh] + normal_traces

    @classmethod
    def from_arrays(
        cls,
        vertices: NDArray,
        faces: NDArray,
        surface_id: int | None = None,
        **kwargs,
    ):
        """Create triangle surface from raw vertex and face arrays."""

        return cls(vertices, faces, surface_id=surface_id, **kwargs)

    @classmethod
    def from_stl_file(
        cls, stl_path: str, surface_id: int | None = 0, is_smooth: bool = False
    ) -> TriSurface:
        """Create a TriSurface instance from an STL file."""
        try:
            stl_data = stl_mesh.Mesh.from_file(str(stl_path))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"No STL file found at {stl_path}") from e

        vertices, faces = cls._extract_vertices_faces(
            stl_data.vectors, is_smooth=is_smooth
        )
        return cls(vertices, faces, surface_id=surface_id, is_smooth=is_smooth)

    @staticmethod
    def _extract_vertices_faces(
        triangle_vectors: np.ndarray, is_smooth: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract vertices and face indices from STL triangle vectors.

        Args:
            triangle_vectors: (N, 3, 3) array where each triangle is 3 vertices
            smooth: If True, deduplicate vertices for smooth shading. If False,
                keep separate vertices.

        Returns:
            Tuple of (vertices, faces) arrays
        """
        if not is_smooth:
            # For flat shading, no need to deduplicate - keep all vertices separate
            vertices = triangle_vectors.reshape(-1, 3)
            faces = np.arange(len(vertices)).reshape(-1, 3)
            return vertices, faces

        # For smooth shading, deduplicate vertices
        all_vertices = triangle_vectors.reshape(-1, 3)

        # Fast deduplication using lexicographic sorting
        tolerance_decimals = 6
        rounded_vertices = np.round(all_vertices, tolerance_decimals)

        # Use numpy unique with return_inverse and return_index to get mapping
        unique_vertices, first_indices, inverse_indices = np.unique(
            rounded_vertices, axis=0, return_inverse=True, return_index=True
        )

        # Create faces array from inverse indices
        faces = inverse_indices.reshape(-1, 3)

        # Use original precision for actual vertices using first occurrence indices
        final_vertices = all_vertices[first_indices]

        return final_vertices, faces
