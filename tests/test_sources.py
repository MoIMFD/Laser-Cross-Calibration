from __future__ import annotations

import numpy as np
import pytest
from plotly.graph_objects import Cone

from laser_cross_calibration.constants import (
    ORIGIN_POINT3,
    UNIT_X_VECTOR3,
    UNIT_Y_VECTOR3,
)
from laser_cross_calibration.sources import (
    DualLaserStageSource,
    LaserSource,
    SingleLaserSource,
)
from tests.utils import assert_vectors_close


@pytest.mark.unit
class TestSingleLaserSource:
    def test_creation(self, single_laser_source_x: SingleLaserSource):
        source = single_laser_source_x

        assert isinstance(source, LaserSource)

    def test_common_methods(self, single_laser_source_x: SingleLaserSource):
        source = single_laser_source_x

        origins = source.get_origins()
        assert isinstance(origins, list)
        assert len(origins) == 1
        assert all(isinstance(origin, np.ndarray) for origin in origins)
        assert_vectors_close(origins[0], ORIGIN_POINT3)

        directions = source.get_directions()
        assert isinstance(directions, list)
        assert len(directions) == 1
        assert all(isinstance(direction, np.ndarray) for direction in directions)
        assert_vectors_close(directions[0], UNIT_X_VECTOR3)

        traces = source.to_plotly()
        assert isinstance(traces, list)
        assert len(traces) == 1
        assert all(isinstance(trace, Cone) for trace in traces)


@pytest.mark.unit
class TestDualLaserStageSource:
    def test_creation(self, dual_laser_stage_source_xy: DualLaserStageSource):
        source = dual_laser_stage_source_xy

        assert isinstance(source, LaserSource | DualLaserStageSource)

    def test_common_methods(self, dual_laser_stage_source_xy: DualLaserStageSource):
        source = dual_laser_stage_source_xy

        origins = source.get_origins()
        assert isinstance(origins, list)
        assert len(origins) == 2
        assert all(isinstance(origin, np.ndarray) for origin in origins)
        assert_vectors_close(origins[0], ORIGIN_POINT3 + UNIT_X_VECTOR3)
        assert_vectors_close(origins[1], ORIGIN_POINT3 + UNIT_Y_VECTOR3)

        directions = source.get_directions()
        assert isinstance(directions, list)
        assert len(directions) == 2
        assert all(isinstance(direction, np.ndarray) for direction in directions)
        assert_vectors_close(directions[0], UNIT_Y_VECTOR3)
        assert_vectors_close(directions[1], UNIT_X_VECTOR3)

        traces = source.to_plotly()
        assert isinstance(traces, list)
        assert len(traces) == 2
        assert all(isinstance(trace, Cone) for trace in traces)
