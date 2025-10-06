"""3D visualization for ray tracing using Plotly."""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from laser_cross_calibration.types import POINT3
from laser_cross_calibration.ray_tracing.ray import OpticalRay
from laser_cross_calibration.ray_tracing.surfaces import Surface, TriSurface, STLSurface


def visualize_scene(
    rays: list[OpticalRay],
    surfaces: Optional[list[Surface]] = None,
    show_normals: bool = False,
    bounds: Optional[tuple] = None,
    title: str = "Ray Tracing Scene",
) -> go.Figure:
    """
    Create 3D visualization of rays and surfaces.

    Args:
        rays: List of OpticalRay objects to visualize
        surfaces: List of Surface objects to visualize
        show_normals: Whether to show surface normals
        bounds: Tuple of (min_coord, max_coord) for axis limits
        title: Plot title
    """
    fig = go.Figure()

    # Add rays
    for i, ray in enumerate(rays):
        if len(ray.path_positions) < 2:
            continue

        positions = np.array(ray.path_positions)

        # Ray path
        fig.add_trace(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="lines+markers",
                line=dict(
                    width=4,
                    color=f"rgba({50 + i * 50 % 200}, {100 + i * 30 % 200}, {200 - i * 20 % 200}, 0.8)",
                ),
                marker=dict(size=3),
                name=f"Ray {i + 1}",
                showlegend=True,
            )
        )

        # Ray direction arrows
        for j in range(len(ray.path_directions)):
            if j < len(positions):
                pos = positions[j]
                direction = ray.path_directions[j] * 0.2  # Scale for visibility

                fig.add_trace(
                    go.Scatter3d(
                        x=[pos[0], pos[0] + direction[0]],
                        y=[pos[1], pos[1] + direction[1]],
                        z=[pos[2], pos[2] + direction[2]],
                        mode="lines",
                        line=dict(width=2, color="red"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    # Add surfaces - now self-describing!
    if surfaces:
        for surface in surfaces:
            surface_traces = surface.to_plotly_surface(bounds, show_normals)
            if isinstance(surface_traces, list):
                # Multiple traces (e.g., surface + normals)
                for trace in surface_traces:
                    fig.add_trace(trace)
            else:
                # Single trace
                fig.add_trace(surface_traces)

    # Configure layout
    if bounds:
        if isinstance(bounds, tuple) and len(bounds) == 2:
            # Single range for all axes
            x_range = y_range = z_range = bounds
        else:
            # Separate ranges
            x_range, y_range, z_range = bounds
    else:
        # Auto-calculate bounds from rays
        all_positions = np.vstack(
            [np.array(ray.path_positions) for ray in rays if ray.path_positions]
        )
        margin = 0.5
        x_min, x_max = (
            all_positions[:, 0].min() - margin,
            all_positions[:, 0].max() + margin,
        )
        y_min, y_max = (
            all_positions[:, 1].min() - margin,
            all_positions[:, 1].max() + margin,
        )
        z_min, z_max = (
            all_positions[:, 2].min() - margin,
            all_positions[:, 2].max() + margin,
        )

        # Make ranges equal for true cube aspect
        range_size = max(x_max - x_min, y_max - y_min, z_max - z_min)
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        z_center = (z_max + z_min) / 2

        half_range = range_size / 2
        x_range = [x_center - half_range, x_center + half_range]
        y_range = [y_center - half_range, y_center + half_range]
        z_range = [z_center - half_range, z_center + half_range]

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=x_range, title="X"),
            yaxis=dict(range=y_range, title="Y"),
            zaxis=dict(range=z_range, title="Z"),
            aspectmode="cube",
            aspectratio=dict(x=1, y=1, z=1),  # Force equal ratios
        ),
        width=800,
        height=600,
    )

    return fig


# All surface visualization logic moved to Surface.to_plotly_surface() methods!
# This eliminates code duplication and makes surfaces self-describing.


def add_points_to_figure(
    fig: go.Figure,
    points: list[POINT3],
    name: str = "Points",
    color: str = "red",
    size: int = 8,
) -> None:
    """
    Add points to an existing plotly figure.

    Args:
        fig: Plotly figure to modify
        points: List of 3D points to add
        name: Name for the trace legend
        color: Color of the points
        size: Size of the markers
    """
    if not points:
        return

    points_array = np.array(points)

    fig.add_trace(
        go.Scatter3d(
            x=points_array[:, 0],
            y=points_array[:, 1],
            z=points_array[:, 2],
            mode="markers",
            marker=dict(size=size, color=color),
            name=name,
        )
    )


def add_intersection_points(fig: go.Figure, ray1: OpticalRay, ray2: OpticalRay) -> None:
    """
    Find and add intersection points between two rays to the figure.

    Args:
        fig: Plotly figure to modify
        ray1: First ray
        ray2: Second ray
    """
    from laser_cross_calibration.ray_tracing.ray import ray_intersection

    intersections = ray_intersection(ray1, ray2)

    if intersections:
        add_points_to_figure(
            fig,
            intersections,
            name=f"Ray Intersections ({len(intersections)})",
            color="orange",
            size=12,
        )
