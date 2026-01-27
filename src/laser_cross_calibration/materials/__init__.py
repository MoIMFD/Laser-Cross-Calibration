"""Material definitions for optical ray tracing."""

from __future__ import annotations

from laser_cross_calibration.materials.base import BaseMaterial
from laser_cross_calibration.materials.constant import ConstantMaterial
from laser_cross_calibration.materials.glass import SellmeierGlass
from laser_cross_calibration.materials.mixtures import WaterGlycerolMixture
from laser_cross_calibration.materials.registry import MaterialRegistry

MATERIAL_REGISTRY = MaterialRegistry()

AIR = ConstantMaterial("air", 1.0)
WATER = ConstantMaterial("water", 1.33)
GLYCEROL = ConstantMaterial("glycerol", 1.47)
PMMA = ConstantMaterial("PMMA", 1.49)
ELASTOSIL_RT601 = ConstantMaterial("Elastosil RT601", 1.408)
POLYCARBONATE = ConstantMaterial("polycarbonate", 1.59)
WATER_GLYCEROL_MIXTURE_90 = WaterGlycerolMixture(
    "water-glycerol", glycerol_fraction=0.9
)

GLASS_BK7 = SellmeierGlass(
    "BK7 glass",
    [1.03961212, 0.231792344, 1.01046945],
    [6.00069867e-3, 2.00179144e-2, 103.560653],
)

GLASS_FUSED_SILICA = SellmeierGlass(
    "fused silica",
    [0.696166300, 0.407942600, 0.897479400],
    [4.67914826e-3, 1.35120631e-2, 97.9340025],
)

MATERIAL_REGISTRY.register(AIR, ["Air", "vacuum"])
MATERIAL_REGISTRY.register(WATER, ["Water", "H2O"])
MATERIAL_REGISTRY.register(GLYCEROL, ["glycerin"])
MATERIAL_REGISTRY.register(PMMA, ["pmma", "acrylic", "plexiglass"])
MATERIAL_REGISTRY.register(POLYCARBONATE, ["PC", "pc"])
MATERIAL_REGISTRY.register(WATER_GLYCEROL_MIXTURE_90, ["water-glycerin-90"])
MATERIAL_REGISTRY.register(GLASS_BK7, ["BK7", "bk7", "glass", "Glass"])
MATERIAL_REGISTRY.register(GLASS_FUSED_SILICA, ["fused-silica", "quartz"])


def get_material(name: str) -> BaseMaterial | None:
    """
    Convenience function to get material from global registry.

    Args:
        name: Material name or alias

    Returns:
        BaseMaterial object if found, None otherwise
    """
    return MATERIAL_REGISTRY.get(name)


def get_material_or_raise(name: str) -> BaseMaterial:
    """
    Convenience function to get material from global registry or raise error.

    Args:
        name: Material name or alias

    Returns:
        BaseMaterial object

    Raises:
        ValueError: If material not found
    """
    return MATERIAL_REGISTRY.get_or_raise(name)


__all__ = [
    "BaseMaterial",
    "ConstantMaterial",
    "SellmeierGlass",
    "WaterGlycerolMixture",
    "MaterialRegistry",
    "MATERIAL_REGISTRY",
    "AIR",
    "WATER",
    "GLYCEROL",
    "PMMA",
    "POLYCARBONATE",
    "WATER_GLYCEROL_MIXTURE_90",
    "GLASS_BK7",
    "GLASS_FUSED_SILICA",
    "get_material",
    "get_material_or_raise",
]
