"""Material definitions for optical ray tracing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Medium:
    """
    Optical medium with refractive properties.
    
    Attributes:
        name: Human-readable name of the medium
        ior: Index of refraction (dimensionless)
    """
    name: str
    ior: float
    
    def __post_init__(self) -> None:
        """Validate medium properties."""
        if self.ior <= 0:
            raise ValueError(f"Index of refraction must be positive, got {self.ior}")


# Common materials for convenience
AIR = Medium("air", 1.0)
WATER = Medium("water", 1.33)
GLASS_BK7 = Medium("BK7 glass", 1.517)
GLASS_FUSED_SILICA = Medium("fused silica", 1.46)
PMMA = Medium("PMMA", 1.49)
POLYCARBONATE = Medium("polycarbonate", 1.59)