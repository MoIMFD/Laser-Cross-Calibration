"""Ray tracing submodule for optical simulations."""

# Materials
from .materials import Medium, AIR, WATER, GLASS_BK7, GLASS_FUSED_SILICA, PMMA, POLYCARBONATE

# Core ray classes
from .ray import OpticalRay, ray_intersection, line_segment_intersection

# Surface classes
from .surfaces import (
    Surface, IntersectionResult, 
    Plane, RectangularPlane, 
    InfiniteCylinder, EllipticCylinder
)

# Complete optical system
from .system import OpticalSystem

# Enhanced optical components
from .optics import OpticalInterface, OpticalElement, MaterialLibrary, WaterTank

# Visualization
from .visualization import visualize_scene, add_points_to_figure, add_intersection_points

__all__ = [
    # Materials
    "Medium", 
    "AIR", 
    "WATER", 
    "GLASS_BK7", 
    "GLASS_FUSED_SILICA", 
    "PMMA", 
    "POLYCARBONATE",
    
    # Ray classes and functions
    "OpticalRay", 
    "ray_intersection", 
    "line_segment_intersection",
    
    # Surface classes
    "Surface", 
    "IntersectionResult",
    "Plane", 
    "RectangularPlane", 
    "InfiniteCylinder", 
    "EllipticCylinder",
    
    # Optical system
    "OpticalSystem",
    
    # Enhanced optical components  
    "OpticalInterface",
    "OpticalElement", 
    "MaterialLibrary",
    "WaterTank",
    
    # Visualization functions
    "visualize_scene", 
    "add_points_to_figure", 
    "add_intersection_points",
]
