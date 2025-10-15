from __future__ import annotations

from numpy.testing import assert_allclose


def assert_vectors_close(actual, expected, atol=1e-6, rtol=1e-6):
    """Assert vectors are close with reasonable tolerances for geometry."""
    assert_allclose(actual, expected, atol=atol, rtol=rtol)


def assert_allclose_list_of_vectors(obj1, obj2):
    for elem1, elem2 in zip(obj1, obj2, strict=False):
        assert_vectors_close(elem1, elem2)
