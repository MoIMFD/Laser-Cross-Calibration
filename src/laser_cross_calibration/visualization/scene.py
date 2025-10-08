from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from laser_cross_calibration.sources.base import LaserSource
from laser_cross_calibration.tracing import OpticalRay, OpticalSystem


class Scene:
    def __init__(self):
        self.sources: list[LaserSource] = []
        self.systems: list[OpticalSystem] = []
        self.rays: list[OpticalRay] = []

        self.bounds_min: tuple[float, float, float] | None = None
        self.bounds_max: tuple[float, float, float] | None = None

        self.title: str = "Ray Tracing Scene"

    def add_source(self, source: LaserSource) -> Scene:
        if not isinstance(source, LaserSource):
            raise ValueError(f"Can only add LaserSource s, got {type(source)=}")
        self.sources.append(source)
        return self

    def add_system(self, system: OpticalSystem) -> Scene:
        if not isinstance(system, OpticalSystem):
            raise ValueError(f"Can only add OpticalSystem s, got {type(system)=}")
        self.systems.append(system)
        return self

    def add_ray(self, ray: OpticalRay) -> Scene:
        if not isinstance(ray, OpticalRay):
            raise ValueError(f"Can only add OpticalRay s, got {type(ray)=}")
        self.rays.append(ray)
        return self

    def add_rays(self, rays: list[OpticalRay]) -> Scene:
        for ray in rays:
            if not isinstance(ray, OpticalRay):
                raise ValueError(f"Can only add OpticalRay s, got {type(ray)=}")
            self.rays.append(ray)
        return self

    def compute_scene_bounds(self, padding_factor: float = 0.1):
        """
        Compute bounding box from all scene elements.

        Args:
            padding_factor: Fraction of size to add as padding (default 10%)

        Returns:
            Tuple of (min_bounds, max_bounds) as 3D arrays
        """
        all_points = []

        for ray in self.rays:
            if ray.path_positions:
                if ray.is_alive and len(ray.path_positions) > 1:
                    all_points.extend(ray.path_positions[:-1])
                else:
                    all_points.extend(ray.path_positions)

        for source in self.sources:
            all_points.extend(source.get_origins())

        for system in self.systems:
            for interface in system.interfaces:
                surface = interface.geometry
                bounds = surface.get_bounds()
                if bounds is not None:
                    min_bound, max_bound = bounds
                    all_points.extend([min_bound, max_bound])

        if not all_points:
            return np.array([-500, -500, -500]), np.array([500, 500, 500])

        points_array = np.array(all_points)
        min_bounds = np.min(points_array, axis=0)
        max_bounds = np.max(points_array, axis=0)

        size = max_bounds - min_bounds
        padding = size * padding_factor
        # padding = np.maximum(padding, 5.0)

        min_bounds -= padding
        max_bounds += padding

        center = (min_bounds + max_bounds) / 2
        max_size = np.max(size + 2 * padding)
        half_size = max_size / 2

        min_bounds = center - half_size
        max_bounds = center + half_size

        return min_bounds, max_bounds

    def make_figure(self):
        fig = go.Figure()

        for system in self.systems:
            for interface in system.interfaces:
                surface = interface.geometry
                surface_traces = surface.to_plotly_surface()

                for trace in surface_traces:
                    fig.add_trace(trace)

        for source in self.sources:
            source_cones = source.to_plotly()
            for cone in source_cones:
                fig.add_trace(cone)

        for i, ray in enumerate(self.rays):
            trace = RayVisualizer(name=f"Ray {i}").ray_to_trace(ray)
            fig.add_trace(trace)

        if self.bounds_min is not None and self.bounds_max is not None:
            min_bounds = np.array(self.bounds_min)
            max_bounds = np.array(self.bounds_max)
        else:
            min_bounds, max_bounds = self.compute_scene_bounds()

        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[min_bounds[0], max_bounds[0]], title="X"),
                yaxis=dict(range=[min_bounds[1], max_bounds[1]], title="Y"),
                zaxis=dict(range=[min_bounds[2], max_bounds[2]], title="Z"),
                aspectmode="cube",
                aspectratio=dict(x=1, y=1, z=1),
            ),
            # width=800,
            # height=600,
        )

        return fig


class RayVisualizer:
    def __init__(self, color: str | None = None, name: str | None = None):
        self.color = color or (
            "rgba("
            f"{np.random.random()}, "
            f"{np.random.random()}, "
            f"{np.random.random()}, "
            "0.8)"
        )
        self.name = name or "Ray"

    def ray_to_trace(self, ray: OpticalRay) -> go.Scatter3d:
        positions = np.array(ray.path_positions)

        # Ray path
        trace = go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="lines+markers",
            line=dict(
                width=4,
                color=self.color,
            ),
            marker=dict(size=3),
            name=self.name,
            showlegend=False,
        )

        return trace
