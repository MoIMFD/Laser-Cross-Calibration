from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from laser_cross_calibration.coordinate_system import Frame, Point, Vector


@pytest.mark.unit
class TestFrame:
    def test_initialization(self):
        frame = Frame(name="test_frame")

        assert frame.name == "test_frame"
        assert frame.parent is None
        assert not frame._is_frozen

    def test_initialization_with_parent(self):
        parent = Frame(name="parent")
        child = Frame(parent=parent, name="child")

        assert child.parent == parent
        assert child.name == "child"

    def test_auto_generated_name(self):
        frame = Frame()
        assert frame.name.startswith("Frame-")

    def test_identity_transformation(self):
        frame = Frame(name="test")
        transform = frame.transform_to_parent

        np.testing.assert_array_almost_equal(transform, np.eye(4))

    def test_global_frame_singleton(self):
        global1 = Frame.global_frame()
        global2 = Frame.global_frame()

        assert global1 is global2
        assert global1.name == "global"
        assert global1.parent is None


@pytest.mark.unit
class TestFrameTranslation:
    def test_translate_x(self):
        frame = Frame()
        frame.translate(x=5.0)

        expected_translation = np.array([5.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(
            frame.combined_translation, expected_translation
        )

    def test_translate_xyz(self):
        frame = Frame()
        frame.translate(x=1.0, y=2.0, z=3.0)

        expected_translation = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(
            frame.combined_translation, expected_translation
        )

    def test_multiple_translations_accumulate(self):
        frame = Frame()
        frame.translate(x=1.0, y=2.0).translate(x=3.0, z=4.0)

        expected_translation = np.array([4.0, 2.0, 4.0])
        np.testing.assert_array_almost_equal(
            frame.combined_translation, expected_translation
        )

    def test_translation_in_transform_matrix(self):
        frame = Frame()
        frame.translate(x=5.0, y=10.0, z=15.0)

        transform = frame.transform_to_parent
        np.testing.assert_array_almost_equal(transform[:3, 3], [5.0, 10.0, 15.0])


@pytest.mark.unit
class TestFrameRotation:
    def test_rotate_euler_z_90_degrees(self):
        frame = Frame()
        frame.rotate_euler(z=90, degrees=True)

        rotation_matrix = frame.combined_rotation.as_matrix()
        expected = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        np.testing.assert_array_almost_equal(rotation_matrix, expected)

    def test_rotate_euler_xyz(self):
        frame = Frame()
        frame.rotate_euler(x=30, y=45, z=60, degrees=True)

        rotation_matrix = frame.combined_rotation.as_matrix()
        expected = Rotation.from_euler("xyz", [30, 45, 60], degrees=True).as_matrix()
        np.testing.assert_array_almost_equal(rotation_matrix, expected)

    def test_multiple_rotations_accumulate(self):
        frame = Frame()
        frame.rotate_euler(z=45, degrees=True).rotate_euler(x=30, degrees=True)

        r1 = Rotation.from_euler("z", 45, degrees=True)
        r2 = Rotation.from_euler("x", 30, degrees=True)
        expected = (r1 * r2).as_matrix()

        np.testing.assert_array_almost_equal(
            frame.combined_rotation.as_matrix(), expected
        )

    def test_rotate_with_matrix(self):
        frame = Frame()
        rotation_matrix = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        frame.rotate(rotation_matrix)

        expected = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        np.testing.assert_array_almost_equal(
            frame.combined_rotation.as_matrix(), expected
        )


@pytest.mark.unit
class TestFrameScaling:
    def test_uniform_scale(self):
        frame = Frame()
        frame.scale(2.0)

        expected_scale = np.diag([2.0, 2.0, 2.0, 1.0])
        np.testing.assert_array_almost_equal(frame.combined_scale, expected_scale)

    def test_non_uniform_scale(self):
        frame = Frame()
        frame.scale((2.0, 3.0, 4.0))

        expected_scale = np.diag([2.0, 3.0, 4.0, 1.0])
        np.testing.assert_array_almost_equal(frame.combined_scale, expected_scale)

    def test_multiple_scales_accumulate(self):
        frame = Frame()
        frame.scale(2.0).scale((1.0, 2.0, 1.0))

        expected_scale = np.diag([2.0, 4.0, 2.0, 1.0])
        np.testing.assert_array_almost_equal(frame.combined_scale, expected_scale)

    def test_scale_invalid_tuple_raises_error(self):
        frame = Frame()
        with pytest.raises(ValueError):
            frame.scale((1.0, 2.0))


@pytest.mark.unit
class TestFrameTransformOrder:
    def test_transform_order_is_scale_rotate_translate(self):
        frame = Frame()
        frame.scale(2.0).rotate_euler(z=90, degrees=True).translate(x=10.0)

        point_local = np.array([1.0, 0.0, 0.0, 1.0])
        transform = frame.transform_to_parent
        point_parent = transform @ point_local

        expected = np.array([10.0, 2.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(point_parent, expected)


@pytest.mark.unit
class TestFrameHierarchy:
    def test_child_to_parent_transformation(self):
        parent = Frame(name="parent")
        child = Frame(parent=parent, name="child")
        child.translate(x=5.0)

        point_in_child = np.array([1.0, 0.0, 0.0, 1.0])
        transform = child.transform_to_parent
        point_in_parent = transform @ point_in_child

        expected = np.array([6.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(point_in_parent, expected)

    def test_nested_hierarchy_to_global(self):
        global_frame = Frame.global_frame()

        parent = Frame(parent=global_frame, name="parent")
        parent.translate(x=10.0)

        child = Frame(parent=parent, name="child")
        child.translate(y=5.0)

        grandchild = Frame(parent=child, name="grandchild")
        grandchild.translate(z=2.0)

        transform = grandchild.transform_to_global
        point_local = np.array([0.0, 0.0, 0.0, 1.0])
        point_global = transform @ point_local

        expected = np.array([10.0, 5.0, 2.0, 1.0])
        np.testing.assert_array_almost_equal(point_global, expected)

    def test_transform_between_sibling_frames(self):
        parent = Frame(name="parent")
        parent.translate(x=100.0)

        child1 = Frame(parent=parent, name="child1")
        child1.translate(y=10.0)

        child2 = Frame(parent=parent, name="child2")
        child2.translate(y=20.0)

        transform = child1.transform_to(child2)
        point_in_child1 = np.array([0.0, 0.0, 0.0, 1.0])
        point_in_child2 = transform @ point_in_child1

        expected = np.array([0.0, -10.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(point_in_child2, expected)

    def test_transform_to_self_is_identity(self):
        frame = Frame()
        transform = frame.transform_to(frame)
        np.testing.assert_array_almost_equal(transform, np.eye(4))


@pytest.mark.unit
class TestFrameCaching:
    def test_transform_is_cached(self):
        frame = Frame()
        frame.translate(x=5.0)

        transform1 = frame.transform_to_parent
        transform2 = frame.transform_to_parent

        np.testing.assert_array_equal(transform1, transform2)
        assert transform1 is not transform2

    def test_cache_invalidated_on_translate(self):
        frame = Frame()
        frame.translate(x=5.0)
        transform1 = frame.transform_to_parent

        frame.translate(y=10.0)
        transform2 = frame.transform_to_parent

        assert not np.array_equal(transform1, transform2)

    def test_cache_invalidated_on_rotate(self):
        frame = Frame()
        transform1 = frame.transform_to_parent

        frame.rotate_euler(z=90, degrees=True)
        transform2 = frame.transform_to_parent

        assert not np.array_equal(transform1, transform2)

    def test_cache_invalidated_on_scale(self):
        frame = Frame()
        transform1 = frame.transform_to_parent

        frame.scale(2.0)
        transform2 = frame.transform_to_parent

        assert not np.array_equal(transform1, transform2)


@pytest.mark.unit
class TestFrameFreeze:
    def test_freeze_prevents_translation(self):
        frame = Frame()
        frame.freeze()

        with pytest.raises(RuntimeError, match="Can not modify frozen frame"):
            frame.translate(x=5.0)

    def test_freeze_prevents_rotation(self):
        frame = Frame()
        frame.freeze()

        with pytest.raises(RuntimeError, match="Can not modify frozen frame"):
            frame.rotate_euler(z=90, degrees=True)

    def test_freeze_prevents_scaling(self):
        frame = Frame()
        frame.freeze()

        with pytest.raises(RuntimeError, match="Can not modify frozen frame"):
            frame.scale(2.0)

    def test_unfreeze_allows_modification(self):
        frame = Frame()
        frame.freeze()
        frame.unfreeze()

        frame.translate(x=5.0)

        expected = np.array([5.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(frame.combined_translation, expected)

    def test_freeze_returns_self(self):
        frame = Frame()
        result = frame.freeze()
        assert result is frame

    def test_unfreeze_returns_self(self):
        frame = Frame()
        result = frame.unfreeze()
        assert result is frame


@pytest.mark.unit
class TestFrameUnitVectors:
    def test_unit_x_in_local_frame(self):
        frame = Frame()
        unit_x = frame.unit_x

        assert isinstance(unit_x, Vector)
        assert unit_x.frame == frame
        np.testing.assert_array_almost_equal(unit_x.coords, [1.0, 0.0, 0.0])

    def test_unit_y_in_local_frame(self):
        frame = Frame()
        unit_y = frame.unit_y

        assert isinstance(unit_y, Vector)
        np.testing.assert_array_almost_equal(unit_y.coords, [0.0, 1.0, 0.0])

    def test_unit_z_in_local_frame(self):
        frame = Frame()
        unit_z = frame.unit_z

        assert isinstance(unit_z, Vector)
        np.testing.assert_array_almost_equal(unit_z.coords, [0.0, 0.0, 1.0])

    def test_unit_vectors_in_global_frame(self):
        global_frame = Frame.global_frame()
        frame = Frame(parent=global_frame, name="rotated")
        frame.rotate_euler(z=90, degrees=True).translate(x=5.0)

        unit_x_global = frame.unit_x_global
        unit_y_global = frame.unit_y_global

        np.testing.assert_array_almost_equal(
            unit_x_global.coords, [0.0, 1.0, 0.0], decimal=5
        )
        np.testing.assert_array_almost_equal(
            unit_y_global.coords, [-1.0, 0.0, 0.0], decimal=5
        )

    def test_unit_vectors_with_scaling(self):
        frame = Frame()
        frame.scale((2.0, 3.0, 4.0))

        unit_x = frame.unit_x
        unit_y = frame.unit_y
        unit_z = frame.unit_z

        np.testing.assert_array_almost_equal(unit_x.coords, [2.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(unit_y.coords, [0.0, 3.0, 0.0])
        np.testing.assert_array_almost_equal(unit_z.coords, [0.0, 0.0, 4.0])


@pytest.mark.unit
class TestFrameOrigin:
    def test_origin_in_local_frame(self):
        frame = Frame()
        origin = frame.origin

        assert isinstance(origin, Point)
        assert origin.frame == frame
        np.testing.assert_array_almost_equal(origin.coords, [0.0, 0.0, 0.0])

    def test_origin_in_global_frame(self):
        global_frame = Frame.global_frame()
        frame = Frame(parent=global_frame, name="translated")
        frame.translate(x=5.0, y=10.0, z=15.0)

        origin_global = frame.origin_global

        assert isinstance(origin_global, Point)
        np.testing.assert_array_almost_equal(origin_global.coords, [5.0, 10.0, 15.0])


@pytest.mark.unit
class TestPoint:
    def test_initialization(self):
        frame = Frame(name="test")
        point = Point(x=1.0, y=2.0, z=3.0, frame=frame)

        assert point.x == 1.0
        assert point.y == 2.0
        assert point.z == 3.0
        assert point.frame == frame
        np.testing.assert_array_almost_equal(point._homogeneous, [1.0, 2.0, 3.0, 1.0])

    def test_coords_property(self):
        frame = Frame()
        point = Point(x=1.0, y=2.0, z=3.0, frame=frame)

        coords = point.coords
        np.testing.assert_array_almost_equal(coords, [1.0, 2.0, 3.0])

    def test_getitem(self):
        frame = Frame()
        point = Point(x=1.0, y=2.0, z=3.0, frame=frame)

        assert point[0] == 1.0
        assert point[1] == 2.0
        assert point[2] == 3.0

    def test_iter(self):
        frame = Frame()
        point = Point(x=1.0, y=2.0, z=3.0, frame=frame)

        coords = list(point)
        assert coords == [1.0, 2.0, 3.0]

    def test_array_conversion(self):
        frame = Frame()
        point = Point(x=1.0, y=2.0, z=3.0, frame=frame)

        arr = np.array(point)
        np.testing.assert_array_almost_equal(arr, [1.0, 2.0, 3.0])

    def test_repr(self):
        frame = Frame(name="test_frame")
        point = Point(x=1.0, y=2.0, z=3.0, frame=frame)

        repr_str = repr(point)
        assert "Point" in repr_str
        assert "x=1.0" in repr_str
        assert "y=2.0" in repr_str
        assert "z=3.0" in repr_str
        assert "frame=test_frame" in repr_str


@pytest.mark.unit
class TestVector:
    def test_initialization(self):
        frame = Frame(name="test")
        vector = Vector(x=1.0, y=2.0, z=3.0, frame=frame)

        assert vector.x == 1.0
        assert vector.y == 2.0
        assert vector.z == 3.0
        assert vector.frame == frame
        np.testing.assert_array_almost_equal(vector._homogeneous, [1.0, 2.0, 3.0, 0.0])

    def test_coords_property(self):
        frame = Frame()
        vector = Vector(x=1.0, y=2.0, z=3.0, frame=frame)

        coords = vector.coords
        np.testing.assert_array_almost_equal(coords, [1.0, 2.0, 3.0])

    def test_repr(self):
        frame = Frame(name="test_frame")
        vector = Vector(x=1.0, y=2.0, z=3.0, frame=frame)

        repr_str = repr(vector)
        assert "Vector" in repr_str
        assert "x=1.0" in repr_str
        assert "frame=test_frame" in repr_str


@pytest.mark.unit
class TestPointArithmetic:
    def test_point_minus_point_gives_vector(self):
        frame = Frame()
        p1 = Point(x=5.0, y=3.0, z=1.0, frame=frame)
        p2 = Point(x=2.0, y=1.0, z=0.0, frame=frame)

        result = p1 - p2

        assert isinstance(result, Vector)
        assert result.frame == frame
        np.testing.assert_array_almost_equal(result.coords, [3.0, 2.0, 1.0])

    def test_point_minus_vector_gives_point(self):
        frame = Frame()
        point = Point(x=5.0, y=3.0, z=1.0, frame=frame)
        vector = Vector(x=2.0, y=1.0, z=0.5, frame=frame)

        result = point - vector

        assert isinstance(result, Point)
        assert result.frame == frame
        np.testing.assert_array_almost_equal(result.coords, [3.0, 2.0, 0.5])

    def test_point_plus_vector_gives_point(self):
        frame = Frame()
        point = Point(x=1.0, y=2.0, z=3.0, frame=frame)
        vector = Vector(x=4.0, y=5.0, z=6.0, frame=frame)

        result = point + vector

        assert isinstance(result, Point)
        assert result.frame == frame
        np.testing.assert_array_almost_equal(result.coords, [5.0, 7.0, 9.0])

    def test_point_plus_point_raises_error(self):
        frame = Frame()
        p1 = Point(x=1.0, y=2.0, z=3.0, frame=frame)
        p2 = Point(x=4.0, y=5.0, z=6.0, frame=frame)

        with pytest.raises(TypeError, match="Can not add 2 Points"):
            p1 + p2


@pytest.mark.unit
class TestVectorArithmetic:
    def test_vector_plus_vector_gives_vector(self):
        frame = Frame()
        v1 = Vector(x=1.0, y=2.0, z=3.0, frame=frame)
        v2 = Vector(x=4.0, y=5.0, z=6.0, frame=frame)

        result = v1 + v2

        assert isinstance(result, Vector)
        assert result.frame == frame
        np.testing.assert_array_almost_equal(result.coords, [5.0, 7.0, 9.0])

    def test_vector_minus_vector_gives_vector(self):
        frame = Frame()
        v1 = Vector(x=5.0, y=3.0, z=1.0, frame=frame)
        v2 = Vector(x=2.0, y=1.0, z=0.5, frame=frame)

        result = v1 - v2

        assert isinstance(result, Vector)
        assert result.frame == frame
        np.testing.assert_array_almost_equal(result.coords, [3.0, 2.0, 0.5])

    def test_vector_plus_point_gives_point(self):
        frame = Frame()
        vector = Vector(x=1.0, y=2.0, z=3.0, frame=frame)
        point = Point(x=4.0, y=5.0, z=6.0, frame=frame)

        result = vector + point

        assert isinstance(result, Point)
        assert result.frame == frame
        np.testing.assert_array_almost_equal(result.coords, [5.0, 7.0, 9.0])

    def test_vector_minus_point_raises_error(self):
        frame = Frame()
        vector = Vector(x=1.0, y=2.0, z=3.0, frame=frame)
        point = Point(x=4.0, y=5.0, z=6.0, frame=frame)

        with pytest.raises(TypeError, match="Cannot subtract Point from Vector"):
            vector - point


@pytest.mark.unit
class TestFrameCompatibility:
    def test_point_point_different_frames_raises_error(self):
        frame1 = Frame(name="frame1")
        frame2 = Frame(name="frame2")

        p1 = Point(x=1.0, y=2.0, z=3.0, frame=frame1)
        p2 = Point(x=4.0, y=5.0, z=6.0, frame=frame2)

        with pytest.raises(
            RuntimeError, match="Can only process objects in same coordinate system"
        ):
            p1 - p2

    def test_point_vector_different_frames_raises_error(self):
        frame1 = Frame(name="frame1")
        frame2 = Frame(name="frame2")

        point = Point(x=1.0, y=2.0, z=3.0, frame=frame1)
        vector = Vector(x=4.0, y=5.0, z=6.0, frame=frame2)

        with pytest.raises(
            RuntimeError, match="Can only process objects in same coordinate system"
        ):
            point + vector

    def test_vector_vector_different_frames_raises_error(self):
        frame1 = Frame(name="frame1")
        frame2 = Frame(name="frame2")

        v1 = Vector(x=1.0, y=2.0, z=3.0, frame=frame1)
        v2 = Vector(x=4.0, y=5.0, z=6.0, frame=frame2)

        with pytest.raises(
            RuntimeError, match="Can only process objects in same coordinate system"
        ):
            v1 + v2


@pytest.mark.unit
class TestPrimitiveTransformation:
    def test_point_to_frame_translation_only(self):
        global_frame = Frame.global_frame()

        frame1 = Frame(parent=global_frame, name="frame1")

        frame2 = Frame(parent=global_frame, name="frame2")
        frame2.translate(x=10.0, y=5.0, z=2.0)

        point_in_frame1 = Point(x=1.0, y=2.0, z=3.0, frame=frame1)
        point_in_frame2 = point_in_frame1.to_frame(frame2)

        assert isinstance(point_in_frame2, Point)
        assert point_in_frame2.frame == frame2
        np.testing.assert_array_almost_equal(point_in_frame2.coords, [-9.0, -3.0, 1.0])

    def test_point_to_frame_rotation_and_translation(self):
        global_frame = Frame.global_frame()

        frame1 = Frame(parent=global_frame, name="frame1")

        frame2 = Frame(parent=global_frame, name="frame2")
        frame2.rotate_euler(z=90, degrees=True).translate(x=5.0)

        point_in_frame1 = Point(x=1.0, y=0.0, z=0.0, frame=frame1)
        point_in_frame2 = point_in_frame1.to_frame(frame2)

        np.testing.assert_array_almost_equal(
            point_in_frame2.coords, [0.0, 4.0, 0.0], decimal=5
        )

    def test_vector_to_frame_translation_invariant(self):
        frame1 = Frame(name="frame1")
        frame2 = Frame(name="frame2")
        frame2.translate(x=100.0, y=200.0, z=300.0)

        vector_in_frame1 = Vector(x=1.0, y=2.0, z=3.0, frame=frame1)
        vector_in_frame2 = vector_in_frame1.to_frame(frame2)

        assert isinstance(vector_in_frame2, Vector)
        assert vector_in_frame2.frame == frame2
        np.testing.assert_array_almost_equal(vector_in_frame2.coords, [1.0, 2.0, 3.0])

    def test_vector_to_frame_rotation(self):
        global_frame = Frame.global_frame()

        frame1 = Frame(parent=global_frame, name="frame1")

        frame2 = Frame(parent=global_frame, name="frame2")
        frame2.rotate_euler(z=90, degrees=True)

        vector_in_frame1 = Vector(x=1.0, y=0.0, z=0.0, frame=frame1)
        vector_in_frame2 = vector_in_frame1.to_frame(frame2)

        np.testing.assert_array_almost_equal(
            vector_in_frame2.coords, [0.0, -1.0, 0.0], decimal=5
        )

    def test_vector_magnitude_preserved_under_rotation(self):
        frame1 = Frame(name="frame1")

        frame2 = Frame(name="frame2")
        frame2.rotate_euler(x=30, y=45, z=60, degrees=True)

        vector_in_frame1 = Vector(x=3.0, y=4.0, z=0.0, frame=frame1)
        vector_in_frame2 = vector_in_frame1.to_frame(frame2)

        magnitude1 = np.linalg.norm(vector_in_frame1.coords)
        magnitude2 = np.linalg.norm(vector_in_frame2.coords)

        np.testing.assert_almost_equal(magnitude1, magnitude2)

    def test_point_to_same_frame_unchanged(self):
        frame = Frame()
        point = Point(x=1.0, y=2.0, z=3.0, frame=frame)

        transformed = point.to_frame(frame)

        np.testing.assert_array_almost_equal(transformed.coords, point.coords)

    def test_vector_to_same_frame_unchanged(self):
        frame = Frame()
        vector = Vector(x=1.0, y=2.0, z=3.0, frame=frame)

        transformed = vector.to_frame(frame)

        np.testing.assert_array_almost_equal(transformed.coords, vector.coords)


@pytest.mark.unit
class TestNumpyIntegration:
    def test_point_with_numpy_add(self):
        frame = Frame()
        point = Point(x=1.0, y=2.0, z=3.0, frame=frame)

        result = np.add(point, np.array([4.0, 5.0, 6.0]))

        np.testing.assert_array_almost_equal(result, [5.0, 7.0, 9.0])

    def test_vector_with_numpy_add(self):
        frame = Frame()
        vector = Vector(x=1.0, y=2.0, z=3.0, frame=frame)

        result = np.add(vector, np.array([4.0, 5.0, 6.0]))

        np.testing.assert_array_almost_equal(result, [5.0, 7.0, 9.0])

    def test_point_array_conversion_with_dtype(self):
        frame = Frame()
        point = Point(x=1.5, y=2.5, z=3.5, frame=frame)

        arr = np.array(point, dtype=np.int32)

        assert arr.dtype == np.int32
        np.testing.assert_array_equal(arr, [1, 2, 3])


@pytest.mark.unit
class TestEdgeCases:
    def test_zero_vector(self):
        frame = Frame()
        vector = Vector(x=0.0, y=0.0, z=0.0, frame=frame)

        np.testing.assert_array_almost_equal(vector.coords, [0.0, 0.0, 0.0])

    def test_negative_coordinates(self):
        frame = Frame()
        point = Point(x=-1.0, y=-2.0, z=-3.0, frame=frame)

        assert point.x == -1.0
        assert point.y == -2.0
        assert point.z == -3.0

    def test_large_coordinate_values(self):
        frame = Frame()
        point = Point(x=1e10, y=1e10, z=1e10, frame=frame)

        assert point.x == 1e10

    def test_very_small_coordinate_values(self):
        frame = Frame()
        point = Point(x=1e-10, y=1e-10, z=1e-10, frame=frame)

        np.testing.assert_almost_equal(point.x, 1e-10)

    def test_transform_chain_consistency(self):
        frame1 = Frame(name="frame1")
        frame2 = Frame(name="frame2")
        frame2.translate(x=5.0)

        frame3 = Frame(name="frame3")
        frame3.translate(y=10.0)

        point = Point(x=1.0, y=2.0, z=3.0, frame=frame1)

        point_2_direct = point.to_frame(frame2)
        point_3_via_2 = point_2_direct.to_frame(frame3)
        point_3_direct = point.to_frame(frame3)

        np.testing.assert_array_almost_equal(
            point_3_via_2.coords, point_3_direct.coords
        )


@pytest.mark.integration
class TestComplexScenarios:
    def test_robotic_arm_kinematics(self):
        base = Frame(name="base")

        shoulder = Frame(parent=base, name="shoulder")
        shoulder.translate(z=0.5)

        elbow = Frame(parent=shoulder, name="elbow")
        elbow.rotate_euler(y=45, degrees=True).translate(x=1.0)

        wrist = Frame(parent=elbow, name="wrist")
        wrist.rotate_euler(y=-30, degrees=True).translate(x=0.8)

        tool_point = Point(x=0.1, y=0.0, z=0.0, frame=wrist)
        tool_point_in_base = tool_point.to_frame(base)

        assert isinstance(tool_point_in_base, Point)
        assert tool_point_in_base.frame == base

    def test_coordinate_system_with_scaling_and_rotation(self):
        parent = Frame(name="parent")
        child = Frame(parent=parent, name="child")
        child.scale(2.0).rotate_euler(z=45, degrees=True).translate(x=10.0, y=5.0)

        point_in_child = Point(x=1.0, y=0.0, z=0.0, frame=child)
        point_in_parent = point_in_child.to_frame(parent)

        expected_x = 10.0 + 2.0 * np.cos(np.radians(45))
        expected_y = 5.0 + 2.0 * np.sin(np.radians(45))

        np.testing.assert_almost_equal(point_in_parent.x, expected_x, decimal=5)
        np.testing.assert_almost_equal(point_in_parent.y, expected_y, decimal=5)

    def test_vector_field_transformation(self):
        source_frame = Frame(name="source")
        target_frame = Frame(name="target")
        target_frame.rotate_euler(z=90, degrees=True).translate(x=5.0, y=5.0)

        vectors = [
            Vector(x=1.0, y=0.0, z=0.0, frame=source_frame),
            Vector(x=0.0, y=1.0, z=0.0, frame=source_frame),
            Vector(x=0.0, y=0.0, z=1.0, frame=source_frame),
        ]

        transformed_vectors = [v.to_frame(target_frame) for v in vectors]

        for original, transformed in zip(vectors, transformed_vectors, strict=False):
            original_magnitude = np.linalg.norm(original.coords)
            transformed_magnitude = np.linalg.norm(transformed.coords)
            np.testing.assert_almost_equal(original_magnitude, transformed_magnitude)
