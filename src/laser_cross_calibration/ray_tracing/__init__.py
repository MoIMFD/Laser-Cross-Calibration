"""Ray tracing submodule for optical simulations."""

# Materials
from .materials import (
    AIR,
    GLASS_BK7,
    GLASS_FUSED_SILICA,
    GLYCEROL,
    PMMA,
    POLYCARBONATE,
    WATER,
    WATER_GLYCEROL_MIXTURE_90,
    BaseMaterial,
    ConstantMaterial,
    SellmeierGlass,
    WaterGlycerolMixture,
    get_material,
    get_material_or_raise,
)

# Enhanced optical components
from .optics import MaterialLibrary, OpticalElement, OpticalInterface, WaterTank

# Core ray classes
from .ray import OpticalRay, line_segment_intersection, ray_intersection

# Surface classes
from .surfaces import (
    EllipticCylinder,
    FiniteCylinder,
    InfiniteCylinder,
    IntersectionResult,
    Plane,
    RectangularPlane,
    STLSurface,
    Surface,
    TriSurface,
)

# Complete optical system
from .system import OpticalSystem

# Visualization
from .visualization import (
    add_intersection_points,
    add_points_to_figure,
    visualize_scene,
)

__all__ = [
    # Materials
    "BaseMaterial",
    "ConstantMaterial",
    "WaterGlycerolMixture",
    "SellmeierGlass",
    "AIR",
    "WATER",
    "GLYCEROL",
    "GLASS_BK7",
    "GLASS_FUSED_SILICA",
    "PMMA",
    "POLYCARBONATE",
    "WATER_GLYCEROL_MIXTURE_90",
    "get_material",
    "get_material_or_raise",
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
    "FiniteCylinder",
    "EllipticCylinder",
    "TriSurface",
    "STLSurface",
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
