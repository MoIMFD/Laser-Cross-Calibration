from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest

from lcc_control_gui.serial_interface import SerialInterface
from lcc_control_gui.stage_controller import StageController


@pytest.fixture
def mock_serial() -> MagicMock:
    return MagicMock(spec=SerialInterface)


@pytest.fixture
def controller(mock_serial) -> StageController:
    return StageController(mock_serial)


class TestStageController:
    def test_home_all_axes(self, controller: StageController, mock_serial):
        mock_serial.send_command.return_value = (SerialInterface.ReplyStatus.OK, "")

        status = controller.home_axes()

        mock_serial.send_command.assert_called_once_with("G28 X Y Z", timeout=30)
        assert status == SerialInterface.ReplyStatus.OK

    def test_home_single_axis(self, controller, mock_serial):
        mock_serial.send_command.return_value = (SerialInterface.ReplyStatus.OK, "")

        status = controller.home_axes(axes=["X"])

        mock_serial.send_command.assert_called_once_with("G28 X", timeout=30)
        assert status == SerialInterface.ReplyStatus.OK

    def test_home_multiple_axes(self, controller, mock_serial):
        mock_serial.send_command.return_value = (SerialInterface.ReplyStatus.OK, "")

        status = controller.home_axes(axes=["X", "Z"])

        mock_serial.send_command.assert_called_once_with("G28 X Z", timeout=30)
        assert status == SerialInterface.ReplyStatus.OK

    def test_move_relative_valid_axis(self, controller, mock_serial):
        mock_serial.send_command.return_value = (SerialInterface.ReplyStatus.OK, "")

        status = controller.move_relative("X", 10.5, 1000.0)

        assert mock_serial.send_command.call_count == 3
        calls = mock_serial.send_command.call_args_list
        assert calls[0][0][0] == "G91"
        assert calls[1][0][0] == "G0 X10.500 F1000.000"
        assert calls[2][0][0] == "G90"
        assert status == SerialInterface.ReplyStatus.OK

    def test_move_relative_lowercase_axis(self, controller, mock_serial):
        mock_serial.send_command.return_value = (SerialInterface.ReplyStatus.OK, "")

        status = controller.move_relative("y", 5.0, 500.0)

        calls = mock_serial.send_command.call_args_list
        assert calls[1][0][0] == "G0 Y5.000 F500.000"
        assert status == SerialInterface.ReplyStatus.OK

    def test_move_relative_invalid_axis(self, controller, mock_serial):
        status = controller.move_relative("W", 10.0, 1000.0)

        assert status == SerialInterface.ReplyStatus.ERROR
        mock_serial.send_command.assert_not_called()

    def test_move_absolute(self, controller, mock_serial):
        mock_serial.send_command.return_value = (SerialInterface.ReplyStatus.OK, "")

        status = controller.move_absolute(10.5, 20.3, 5.1, 1500.0)

        expected_cmd = "G90\nG0 X10.500 Y20.300 Z5.100 F1500.000"
        mock_serial.send_command.assert_called_once_with(expected_cmd)
        assert status == SerialInterface.ReplyStatus.OK

    def test_set_acceleration(self, controller, mock_serial):
        mock_serial.send_command.return_value = (SerialInterface.ReplyStatus.OK, "")

        status = controller.set_acceleration(500.0)

        mock_serial.send_command.assert_called_once_with("M204 P500.000")
        assert status == SerialInterface.ReplyStatus.OK

    def test_set_max_feedrate(self, controller, mock_serial):
        mock_serial.send_command.return_value = (SerialInterface.ReplyStatus.OK, "")

        status = controller.set_max_feedrate(100.0, 200.0, 50.0)

        mock_serial.send_command.assert_called_once_with(
            "M203 X100.000 Y200.000 Z50.000"
        )
        assert status == SerialInterface.ReplyStatus.OK

    def test_read_position_success(self, controller, mock_serial):
        mock_serial.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "X:10.500 Y:20.300 Z:5.100",
        )

        position = controller.read_position()

        mock_serial.send_command.assert_called_once_with("M114")
        assert position == (10.5, 20.3, 5.1)

    def test_read_position_with_extra_text(
        self, controller: StageController, mock_serial
    ):
        mock_serial.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "Count X:100 Y:200 Z:50\nX:10.500 Y:20.300 Z:5.100",
        )

        position = controller.read_position()

        assert position == (10.5, 20.3, 5.1)

    def test_read_position_error_status(self, controller, mock_serial):
        mock_serial.send_command.return_value = (
            SerialInterface.ReplyStatus.ERROR,
            "",
        )

        position = controller.read_position()

        assert position is None

    def test_read_position_empty_response(self, controller, mock_serial):
        mock_serial.send_command.return_value = (SerialInterface.ReplyStatus.OK, "")

        position = controller.read_position()

        assert position is None

    def test_read_position_malformed_response(self, controller, mock_serial):
        mock_serial.send_command.return_value = (
            SerialInterface.ReplyStatus.OK,
            "Invalid response",
        )

        position = controller.read_position()

        assert position is None

    def test_send_gcode_line(self, controller, mock_serial):
        mock_serial.send_command.return_value = (SerialInterface.ReplyStatus.OK, "")

        status = controller.send_gcode_line("G0 X10")

        mock_serial.send_command.assert_called_once_with("G0 X10")
        assert status == SerialInterface.ReplyStatus.OK

    def test_emergency_stop(self, controller, mock_serial):
        mock_serial.send_command.return_value = (SerialInterface.ReplyStatus.OK, "")

        status = controller.emergency_stop()

        mock_serial.send_command.assert_called_once_with("M410")
        assert status == SerialInterface.ReplyStatus.OK

    def test_trigger_recording(self, controller, mock_serial):
        mock_serial.send_command.return_value = (SerialInterface.ReplyStatus.OK, "")

        status = controller.trigger_recording()

        assert mock_serial.send_command.call_count == 5
        calls = [call[0][0] for call in mock_serial.send_command.call_args_list]
        assert "G4 P250" in calls
        assert "M42 P67 S255" in calls
        assert "M42 P67 S0" in calls
        assert status == SerialInterface.ReplyStatus.OK

    def test_trigger_recording_error(self, controller, mock_serial):
        mock_serial.send_command.side_effect = [
            (SerialInterface.ReplyStatus.OK, ""),
            (SerialInterface.ReplyStatus.ERROR, ""),
        ]

        status = controller.trigger_recording()

        assert status == SerialInterface.ReplyStatus.ERROR

    def test_reset_to_defaults(self, controller, mock_serial):
        mock_serial.send_command.return_value = (SerialInterface.ReplyStatus.OK, "")

        status = controller.reset_to_defaults()

        mock_serial.send_command.assert_called_once_with("M502")
        assert status == SerialInterface.ReplyStatus.OK

    def test_run_gcode_file_success(
        self, controller: StageController, mock_serial, tmp_path
    ):
        gcode_file = tmp_path / "test.gcode"
        gcode_file.write_text("G28\nG0 X10\n; Comment\nG0 Y20\n")

        mock_serial.send_command.return_value = (SerialInterface.ReplyStatus.OK, "")
        progress_callback = Mock()

        result = None

        def callback(x):
            nonlocal result
            result = x

        controller._run_gcode_thread(
            gcode_file, progress_callback, completion_callback=callback
        )

        assert result == SerialInterface.ReplyStatus.OK
        assert mock_serial.send_command.call_count == 3
        assert progress_callback.call_count == 3

    def test_run_gcode_file_with_error(
        self, controller: StageController, mock_serial, tmp_path
    ):
        gcode_file = tmp_path / "test.gcode"
        gcode_file.write_text("G28\nG0 X10\n")

        mock_serial.send_command.side_effect = [
            (SerialInterface.ReplyStatus.OK, ""),
            (SerialInterface.ReplyStatus.ERROR, ""),
        ]

        result = None

        def callback(x):
            nonlocal result
            result = x

        controller._run_gcode_thread(gcode_file, completion_callback=callback)

        assert result == SerialInterface.ReplyStatus.ERROR

    def test_run_gcode_file_not_found(self, controller: StageController, mock_serial):
        result = None

        def callback(x):
            nonlocal result
            result = x

        controller._run_gcode_thread(
            "/nonexistent/file.gcode", completion_callback=callback
        )

        assert result == SerialInterface.ReplyStatus.ERROR
        mock_serial.send_command.assert_not_called()

    def test_run_gcode_file_skip_comments(
        self, controller: StageController, mock_serial, tmp_path
    ):
        gcode_file = tmp_path / "test.gcode"
        gcode_file.write_text("; Full line comment\nG28\n; Another comment\n")

        mock_serial.send_command.return_value = (SerialInterface.ReplyStatus.OK, "")

        result = None

        def callback(x):
            nonlocal result
            result = x

        controller._run_gcode_thread(gcode_file, completion_callback=callback)

        assert result == SerialInterface.ReplyStatus.OK
        assert mock_serial.send_command.call_count == 1
        mock_serial.send_command.assert_called_with("G28")

    def test_run_gcode_file_skip_empty_lines(
        self, controller: StageController, mock_serial, tmp_path
    ):
        gcode_file = tmp_path / "test.gcode"
        gcode_file.write_text("G28\n\n\nG0 X10\n")

        mock_serial.send_command.return_value = (SerialInterface.ReplyStatus.OK, "")

        result = None

        def callback(x):
            nonlocal result
            result = x

        controller._run_gcode_thread(gcode_file, completion_callback=callback)

        assert result == SerialInterface.ReplyStatus.OK
        assert mock_serial.send_command.call_count == 2
