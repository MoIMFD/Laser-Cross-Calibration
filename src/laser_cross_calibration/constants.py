from __future__ import annotations

VSMALL: float = 1e-8
VVSMALL: float = 1e-12  # For preventing division by zero
INTERSECTION_THRESHOLD: float = 1e-4

# Minimum distance for a ray to re-hit the interface it just refracted/reflected
# off of. Guards against spurious self-intersections on triangulated (TriSurface)
# meshes, where the localize/globalize round-trip after a hit can leave the ray
# just off-surface, causing the next intersection test to immediately re-hit the
# same mesh at a distance on the order of VSMALL. See
# tests/test_trisurface_self_intersection.py.
SELF_INTERSECTION_EPSILON: float = 1e-6
