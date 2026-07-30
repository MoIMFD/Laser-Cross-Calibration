"""Regression test for spurious self-intersection on TriSurface entry.

When a ray refracts through a `TriSurface` interface, the next call to
`OpticalSystem.find_next_intersection` can immediately re-hit the *same* mesh
at a near-zero distance (~1e-8 m), caused by floating-point error in the
localize/globalize round-trip after `ray.propagate`. `VSMALL` (1e-8) is not
reliably larger than this error, so the spurious hit passes the `t > VSMALL`
filter in `TriSurface.intersect`.

That phantom hit flips the ray's tracked medium back to `material_pre`
(`OpticalInterface.get_next_medium` sees `current_medium == material_post`
and returns `material_pre`), even though the ray never physically left the
transmitted medium. With only one interface defined, the ray incorrectly
ends up back in the incident medium for the rest of its path instead of
staying in the transmitted one.

The reproducing mesh below is a 13-triangle patch extracted verbatim (vertex
positions and winding) from `examples/stl-files/simple-bend-outer.stl`
around the exact triangle a straight probe ray hits, so this is real
mesh data, not a synthetic edge case. A single flat 2-triangle plane in the
same orientation does NOT reproduce it, so the fault seems specific to a
ray landing near a shared edge/vertex of adjacent triangles in a denser mesh.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import laser_cross_calibration as lcc
from laser_cross_calibration.surfaces import TriSurface

if TYPE_CHECKING:
    from hazy import Frame

# Vertex positions and winding taken as-is from the outer wall mesh in
# examples/stl-files/simple-bend-outer.stl (triangle containing the point
# hit by a straight ray at x=0, y=-0.07, travelling in +z), plus its
# geometric neighbors sharing a vertex.
_PATCH_VERTICES = np.array(
    [
        [-0.001121273, -0.065985062, -0.014142126],
        [-0.000928864, -0.068201462, -0.013431170],
        [-0.000673534, -0.070408554, -0.012687857],
        [-0.000463609, -0.065714996, -0.014819013],
        [-0.000305894, -0.067936722, -0.014142126],
        [-0.000086517, -0.070150620, -0.013431170],
        [0.000195674, -0.072354446, -0.012687857],
        [0.000348432, -0.067658669, -0.014819013],
        [0.000533184, -0.069878319, -0.014142126],
        [0.000779507, -0.072089378, -0.013431170],
        [0.001184075, -0.069592308, -0.014819013],
        [0.001395846, -0.071809547, -0.014142126],
    ]
)
_PATCH_FACES = np.array(
    [
        [1, 5, 2],
        [5, 9, 6],
        [5, 6, 2],
        [0, 4, 1],
        [4, 8, 5],
        [4, 5, 1],
        [8, 11, 9],
        [8, 9, 5],
        [3, 7, 4],
        [3, 4, 0],
        [7, 10, 8],
        [7, 8, 4],
        [10, 11, 8],
    ]
)


@pytest.fixture
def bend_wall_patch(frame: Frame) -> TriSurface:
    """Small real-mesh patch known to trigger the self-intersection bug."""
    return TriSurface(
        frame=frame,
        vertices=_PATCH_VERTICES.copy(),
        faces=_PATCH_FACES.copy(),
        is_smooth=False,
    )


@pytest.mark.unit
def test_ray_stays_in_transmitted_medium_after_trisurface_entry(
    bend_wall_patch: TriSurface, frame: Frame
):
    """A ray crossing a single TriSurface interface must stay in material_post.

    With exactly one interface defined (air -> PMMA) and nothing beyond it,
    a ray that enters the mesh has nothing left to refract off of and must
    remain in PMMA for the rest of its path.
    """
    system = lcc.tracing.OpticalSystem(final_propagation_distance=0.05)
    system.add_interface(
        lcc.tracing.OpticalInterface(
            geometry=bend_wall_patch,
            material_pre=lcc.materials.AIR,
            material_post=lcc.materials.PMMA,
        )
    )
    tracer = lcc.tracing.RayTracer(optical_system=system)

    source = lcc.sources.SingleLaserSource(
        origin=frame.point(0.0, -0.07, -0.05), direction=frame.vector(0.0, 0.0, 1.0)
    )
    rays, _ = tracer.trace_and_find_crossings(sources=[source])
    ray = rays[0]

    media_names = [medium.name for medium in ray.media_history]

    assert media_names[-1] == "PMMA", (
        "Ray should remain in PMMA after crossing the only interface in the "
        f"system, but ended up in {media_names[-1]!r} "
        f"(full sequence: {media_names})"
    )
    assert "air" not in media_names[1:], (
        "Ray should not spuriously return to air after entering PMMA "
        f"(full sequence: {media_names})"
    )
