from __future__ import annotations

import pytest

from laser_cross_calibration.materials import ConstantMaterial


@pytest.mark.unit
class TestConstantMaterial:
    """Tests for ConstantMaterial class."""

    def test_creation(self):
        mat = ConstantMaterial(name="Water", ior=1.333)
        assert mat.name == "Water"
        assert mat.ior == 1.333

    def test_refractive_index_constant(self):
        mat = ConstantMaterial(name="Glass", ior=1.5)
        assert mat.n(wavelength=400e-9) == 1.5
        assert mat.n(wavelength=700e-9) == 1.5
        assert mat.n() == 1.5

    def test_invalid_ior_raises_error(self):
        with pytest.raises(ValueError, match="Index of refraction must be positive"):
            ConstantMaterial(name="Invalid", ior=0.0)

        with pytest.raises(ValueError, match="Index of refraction must be positive"):
            ConstantMaterial(name="Invalid", ior=-1.0)

    def test_string_representation(self):
        mat = ConstantMaterial(name="Air", ior=1.0)
        assert "ConstantMaterial" in str(mat)
        assert "Air" in str(mat)
        assert "1.0" in str(mat)
