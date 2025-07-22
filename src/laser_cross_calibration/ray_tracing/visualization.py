"""3D visualization for ray tracing using Plotly."""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from ..types import POINT3
from .ray import OpticalRay
from .surfaces import Surface, Plane, RectangularPlane, InfiniteCylinder, EllipticCylinder


def visualize_scene(
    rays: list[OpticalRay],
    surfaces: Optional[list[Surface]] = None,
    show_normals: bool = False,
    bounds: Optional[tuple] = None,
    title: str = "Ray Tracing Scene"
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
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1], 
            z=positions[:, 2],
            mode='lines+markers',
            line=dict(width=4, color=f'rgba({50+i*50%200}, {100+i*30%200}, {200-i*20%200}, 0.8)'),
            marker=dict(size=3),
            name=f'Ray {i+1}',
            showlegend=True
        ))
        
        # Ray direction arrows
        for j in range(len(ray.path_directions)):
            if j < len(positions):
                pos = positions[j]
                direction = ray.path_directions[j] * 0.2  # Scale for visibility
                
                fig.add_trace(go.Scatter3d(
                    x=[pos[0], pos[0] + direction[0]],
                    y=[pos[1], pos[1] + direction[1]],
                    z=[pos[2], pos[2] + direction[2]],
                    mode='lines',
                    line=dict(width=2, color='red'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Add surfaces
    if surfaces:
        for i, surface in enumerate(surfaces):
            _add_surface_to_plot(fig, surface, bounds, show_normals)
    
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
        all_positions = np.vstack([np.array(ray.path_positions) for ray in rays if ray.path_positions])
        margin = 0.5
        x_min, x_max = all_positions[:, 0].min() - margin, all_positions[:, 0].max() + margin
        y_min, y_max = all_positions[:, 1].min() - margin, all_positions[:, 1].max() + margin
        z_min, z_max = all_positions[:, 2].min() - margin, all_positions[:, 2].max() + margin
        
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
            xaxis=dict(range=x_range, title='X'),
            yaxis=dict(range=y_range, title='Y'),
            zaxis=dict(range=z_range, title='Z'),
            aspectmode='cube',
            aspectratio=dict(x=1, y=1, z=1)  # Force equal ratios
        ),
        width=800,
        height=600
    )
    
    return fig


def _add_surface_to_plot(
    fig: go.Figure, 
    surface: Surface, 
    bounds: Optional[tuple] = None,
    show_normals: bool = False
):
    """Add a surface to the plot."""
    if isinstance(surface, Plane):
        _add_plane_to_plot(fig, surface, bounds, show_normals)
    elif isinstance(surface, RectangularPlane):
        _add_rectangular_plane_to_plot(fig, surface, bounds, show_normals)
    elif isinstance(surface, InfiniteCylinder):
        _add_cylinder_to_plot(fig, surface, bounds, show_normals)
    elif isinstance(surface, EllipticCylinder):
        _add_elliptic_cylinder_to_plot(fig, surface, bounds, show_normals)


def _add_plane_to_plot(
    fig: go.Figure,
    plane: Plane, 
    bounds: Optional[tuple] = None,
    show_normals: bool = False
):
    """Add a plane surface to the plot."""
    if bounds is None:
        bounds = (-2, 2)
    
    # Create grid points on the plane
    u_range = v_range = np.linspace(bounds[0], bounds[1], 10)
    u_grid, v_grid = np.meshgrid(u_range, v_range)
    
    # Create two orthogonal vectors in the plane
    if abs(plane.normal[0]) < 0.9:
        u_vec = np.cross(plane.normal, [1, 0, 0])
    else:
        u_vec = np.cross(plane.normal, [0, 1, 0])
    u_vec = u_vec / np.linalg.norm(u_vec)
    v_vec = np.cross(plane.normal, u_vec)
    
    # Generate plane points
    points = plane.point[:, np.newaxis, np.newaxis] + \
             u_vec[:, np.newaxis, np.newaxis] * u_grid[np.newaxis, :, :] + \
             v_vec[:, np.newaxis, np.newaxis] * v_grid[np.newaxis, :, :]
    
    fig.add_trace(go.Surface(
        x=points[0],
        y=points[1],
        z=points[2],
        opacity=0.3,
        colorscale='Blues',
        showscale=False,
        name=f'Plane {plane.surface_id}'
    ))
    
    if show_normals:
        # Add normal vector
        normal_end = plane.point + plane.normal * 0.5
        fig.add_trace(go.Scatter3d(
            x=[plane.point[0], normal_end[0]],
            y=[plane.point[1], normal_end[1]],
            z=[plane.point[2], normal_end[2]],
            mode='lines',
            line=dict(width=4, color='green'),
            name=f'Normal {plane.surface_id}',
            showlegend=False
        ))


def _add_rectangular_plane_to_plot(
    fig: go.Figure,
    rect_plane: RectangularPlane,
    bounds: Optional[tuple] = None,
    show_normals: bool = False
):
    """Add a rectangular plane surface to the plot."""
    # Get the four corners
    corners = rect_plane.get_corners()
    
    # Add as a mesh (2 triangles forming rectangle)
    x_coords = [corner[0] for corner in corners] + [corners[0][0]]  # Close the loop
    y_coords = [corner[1] for corner in corners] + [corners[0][1]]
    z_coords = [corner[2] for corner in corners] + [corners[0][2]]
    
    # Create two triangles to form the rectangle
    fig.add_trace(go.Mesh3d(
        x=[corners[0][0], corners[1][0], corners[2][0], corners[3][0]],
        y=[corners[0][1], corners[1][1], corners[2][1], corners[3][1]],
        z=[corners[0][2], corners[1][2], corners[2][2], corners[3][2]],
        i=[0, 0],  # Triangle 1: 0-1-2, Triangle 2: 0-2-3
        j=[1, 2],
        k=[2, 3],
        opacity=0.3,
        color='lightblue',
        name=f'Rectangular Plane {rect_plane.surface_id}',
        showscale=False
    ))
    
    # Add rectangle outline
    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='lines',
        line=dict(width=3, color='blue'),
        name=f'Rectangle Edge {rect_plane.surface_id}',
        showlegend=False
    ))
    
    if show_normals:
        # Add normal vector at center
        normal_end = rect_plane.center + rect_plane.normal * 0.5
        fig.add_trace(go.Scatter3d(
            x=[rect_plane.center[0], normal_end[0]],
            y=[rect_plane.center[1], normal_end[1]],
            z=[rect_plane.center[2], normal_end[2]],
            mode='lines',
            line=dict(width=4, color='green'),
            name=f'Normal {rect_plane.surface_id}',
            showlegend=False
        ))


def _add_cylinder_to_plot(
    fig: go.Figure,
    cylinder: InfiniteCylinder,
    bounds: Optional[tuple] = None,
    show_normals: bool = False
):
    """Add a cylinder surface to the plot."""
    if bounds is None:
        bounds = (-2, 2)
    
    # Generate cylinder surface in local coordinates
    theta = np.linspace(0, 2*np.pi, 30)
    s_range = np.linspace(bounds[0], bounds[1], 20)  # Parameter along axis
    theta_grid, s_grid = np.meshgrid(theta, s_range)
    
    # Create orthonormal basis with cylinder.axis as one axis
    axis = cylinder.axis
    
    # Find two perpendicular vectors to the axis
    if abs(axis[0]) < 0.9:
        u = np.cross(axis, [1, 0, 0])
    else:
        u = np.cross(axis, [0, 1, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    v = v / np.linalg.norm(v)
    
    # Generate cylinder points
    # Radial component in u-v plane + axial component
    x_local = cylinder.radius * np.cos(theta_grid)
    y_local = cylinder.radius * np.sin(theta_grid)
    
    # Transform to world coordinates
    points = (cylinder.center[:, np.newaxis, np.newaxis] + 
              s_grid[np.newaxis, :, :] * axis[:, np.newaxis, np.newaxis] +
              x_local[np.newaxis, :, :] * u[:, np.newaxis, np.newaxis] + 
              y_local[np.newaxis, :, :] * v[:, np.newaxis, np.newaxis])
    
    fig.add_trace(go.Surface(
        x=points[0],
        y=points[1], 
        z=points[2],
        opacity=0.4,
        colorscale='Reds',
        showscale=False,
        name=f'Cylinder {cylinder.surface_id}'
    ))


def _add_elliptic_cylinder_to_plot(
    fig: go.Figure,
    elliptic_cylinder: EllipticCylinder,
    bounds: Optional[tuple] = None,
    show_normals: bool = False
):
    """Add an elliptic cylinder surface to the plot."""
    if bounds is None:
        bounds = (-2, 2)
    
    # Generate elliptic cylinder surface in local coordinates
    theta = np.linspace(0, 2*np.pi, 30)
    s_range = np.linspace(bounds[0], bounds[1], 20)  # Parameter along axis
    theta_grid, s_grid = np.meshgrid(theta, s_range)
    
    # Ellipse parameters
    a = elliptic_cylinder.major_radius
    b = elliptic_cylinder.minor_radius
    
    # Generate elliptic cylinder points
    # Ellipse in major-minor axis plane: x = a*cos(θ), y = b*sin(θ)
    x_local = a * np.cos(theta_grid)
    y_local = b * np.sin(theta_grid)
    
    # Transform to world coordinates using the orthonormal basis
    points = (elliptic_cylinder.center[:, np.newaxis, np.newaxis] + 
              s_grid[np.newaxis, :, :] * elliptic_cylinder.axis[:, np.newaxis, np.newaxis] +
              x_local[np.newaxis, :, :] * elliptic_cylinder.major_axis[:, np.newaxis, np.newaxis] + 
              y_local[np.newaxis, :, :] * elliptic_cylinder.minor_axis[:, np.newaxis, np.newaxis])
    
    fig.add_trace(go.Surface(
        x=points[0],
        y=points[1], 
        z=points[2],
        opacity=0.4,
        colorscale='Greens',
        showscale=False,
        name=f'Elliptic Cylinder {elliptic_cylinder.surface_id}'
    ))


def add_points_to_figure(
    fig: go.Figure, 
    points: list[POINT3], 
    name: str = "Points", 
    color: str = "red", 
    size: int = 8
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
    
    fig.add_trace(go.Scatter3d(
        x=points_array[:, 0],
        y=points_array[:, 1],
        z=points_array[:, 2],
        mode='markers',
        marker=dict(size=size, color=color),
        name=name
    ))


def add_intersection_points(fig: go.Figure, ray1: OpticalRay, ray2: OpticalRay) -> None:
    """
    Find and add intersection points between two rays to the figure.
    
    Args:
        fig: Plotly figure to modify
        ray1: First ray
        ray2: Second ray
    """
    from .ray import ray_intersection
    
    intersections = ray_intersection(ray1, ray2)
    
    if intersections:
        add_points_to_figure(
            fig, intersections, 
            name=f"Ray Intersections ({len(intersections)})", 
            color="orange", 
            size=12
        )