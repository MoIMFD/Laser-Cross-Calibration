"""Debug utilities for ray intersection analysis."""

import numpy as np
import plotly.graph_objects as go
from .ray import OpticalRay, intersection
from ..types import POINT3


def debug_ray_paths(ray1: OpticalRay, ray2: OpticalRay) -> None:
    """Print detailed ray path information."""
    print("=== RAY 1 ===")
    print(f"Positions: {len(ray1.path_positions)}")
    for i, pos in enumerate(ray1.path_positions):
        print(f"  {i}: {pos}")
    
    print(f"Directions: {len(ray1.path_directions)}")
    for i, dir in enumerate(ray1.path_directions):
        print(f"  {i}: {dir}")
    
    print(f"Distances: {len(ray1.segment_distances)}")
    for i, dist in enumerate(ray1.segment_distances):
        print(f"  {i}: {dist}")
    
    print("\n=== RAY 2 ===")
    print(f"Positions: {len(ray2.path_positions)}")
    for i, pos in enumerate(ray2.path_positions):
        print(f"  {i}: {pos}")
    
    print(f"Directions: {len(ray2.path_directions)}")
    for i, dir in enumerate(ray2.path_directions):
        print(f"  {i}: {dir}")
    
    print(f"Distances: {len(ray2.segment_distances)}")
    for i, dist in enumerate(ray2.segment_distances):
        print(f"  {i}: {dist}")


def visualize_segments_with_labels(ray1: OpticalRay, ray2: OpticalRay) -> go.Figure:
    """Create detailed visualization showing segment indices and endpoints."""
    fig = go.Figure()
    
    # Ray 1 segments
    for i in range(len(ray1.segment_distances)):
        start = ray1.path_positions[i]
        end = ray1.path_positions[i + 1]
        
        # Segment line
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines+markers',
            line=dict(width=6, color='blue'),
            marker=dict(size=4),
            name=f'Ray1 Seg{i}',
            showlegend=True
        ))
        
        # Segment midpoint label
        mid = (start + end) / 2
        fig.add_trace(go.Scatter3d(
            x=[mid[0]],
            y=[mid[1]],
            z=[mid[2]],
            mode='text',
            text=[f'R1-{i}'],
            textposition='middle center',
            showlegend=False
        ))
    
    # Ray 2 segments  
    for i in range(len(ray2.segment_distances)):
        start = ray2.path_positions[i]
        end = ray2.path_positions[i + 1]
        
        # Segment line
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines+markers',
            line=dict(width=6, color='red'),
            marker=dict(size=4),
            name=f'Ray2 Seg{i}',
            showlegend=True
        ))
        
        # Segment midpoint label
        mid = (start + end) / 2
        fig.add_trace(go.Scatter3d(
            x=[mid[0]],
            y=[mid[1]],
            z=[mid[2]],
            mode='text',
            text=[f'R2-{i}'],
            textposition='middle center',
            showlegend=False
        ))
    
    # Add intersection points
    intersections = intersection(ray1, ray2, threshold=1e-3, debug=True)
    if intersections:
        points = np.array(intersections)
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(size=15, color='orange', symbol='diamond'),
            name='Intersections'
        ))
    
    fig.update_layout(
        title="Ray Segments with Labels",
        scene=dict(aspectmode='cube'),
        width=800,
        height=600
    )
    
    return fig


def check_visual_intersection(ray1: OpticalRay, ray2: OpticalRay, 
                            expected_point: POINT3, tolerance: float = 0.1) -> None:
    """Check if calculated intersections match visual expectation."""
    intersections = intersection(ray1, ray2, threshold=1e-3)
    
    print(f"\nExpected intersection near: {expected_point}")
    print(f"Calculated intersections: {len(intersections)}")
    
    for i, point in enumerate(intersections):
        distance = np.linalg.norm(point - expected_point)
        print(f"  {i}: {point} (distance from expected: {distance:.6f})")
        
        if distance < tolerance:
            print(f"    ✓ Match within tolerance!")
        else:
            print(f"    ✗ Outside tolerance ({tolerance})")