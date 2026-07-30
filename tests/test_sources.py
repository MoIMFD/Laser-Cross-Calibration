from __future__ import annotations

import pytest
from hazy import Frame, Point, Vector
from plotly.graph_objects import Cone

from laser_cross_calibration.sources import (
    DualLaserStageSource,
    LaserSource,
    SingleLaserSource,
)
from tests.utils import assert_vectors_close


@pytest.mark.unit
class TestSingleLaserSource:
    def test_creation(self, frame: Frame):
        source = SingleLaserSource(frame.origin, frame.x_axis)

        assert isinstance(source, LaserSource)

    def test_common_methods(self, frame: Frame):
        source = SingleLaserSource(frame.origin, frame.x_axis)

        origins = source.get_origins()
        assert isinstance(origins, list)
        assert len(origins) == 1
        assert all(isinstance(origin, Point) for origin in origins)
        assert_vectors_close(origins[0], frame.origin)

        directions = source.get_directions()
        assert isinstance(directions, list)
        assert len(directions) == 1
        assert all(isinstance(direction, Vector) for direction in directions)
        assert_vectors_close(directions[0], frame.x_axis)

        traces = source.to_plotly()
        assert isinstance(traces, list)
        assert len(traces) == 1
        assert all(isinstance(trace, Cone) for trace in traces)


@pytest.mark.unit
class TestDualLaserStageSource:
    def test_creation(self, frame: Frame):
        source = DualLaserStageSource(
            frame.origin, frame.x_axis, frame.y_axis, frame.y_axis, frame.x_axis
        )

        assert isinstance(source, LaserSource | DualLaserStageSource)

    def test_common_methods(self, frame: Frame):
        source = DualLaserStageSource(
            frame.origin, frame.x_axis, frame.y_axis, frame.y_axis, frame.x_axis
        )

        origins = source.get_origins()
        assert isinstance(origins, list)
        assert len(origins) == 2
        assert all(isinstance(origin, Point) for origin in origins)
        assert_vectors_close(origins[0], frame.origin + frame.x_axis)
        assert_vectors_close(origins[1], frame.origin + frame.y_axis)

        directions = source.get_directions()
        assert isinstance(directions, list)
        assert len(directions) == 2
        assert all(isinstance(direction, Vector) for direction in directions)
        assert_vectors_close(directions[0], frame.y_axis)
        assert_vectors_close(directions[1], frame.x_axis)
