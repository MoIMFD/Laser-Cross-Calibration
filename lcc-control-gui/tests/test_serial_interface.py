from __future__ import annotations

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from lcc_control_gui.serial_interface import SerialInterface


@pytest.fixture
def mock_serial():
    with patch("lcc_control_gui.serial_interface.serial.Serial") as mock:
        yield mock


class TestSerialInterface:
    def test_check_log_msg_debug(self):
        serial_interface = SerialInterface.__new__(SerialInterface)
        level, msg = serial_interface._check_log_msg("D)Debug message")
        assert level == SerialInterface.LogLevel.DEBUG
        assert msg == "Debug message"

    def test_check_log_msg_info(self):
        serial_interface = SerialInterface.__new__(SerialInterface)
        level, msg = serial_interface._check_log_msg("I)Info message")
        assert level == SerialInterface.LogLevel.INFO
        assert msg == "Info message"

    def test_check_log_msg_warning(self):
        serial_interface = SerialInterface.__new__(SerialInterface)
        level, msg = serial_interface._check_log_msg("W)Warning message")
        assert level == SerialInterface.LogLevel.WARNING
        assert msg == "Warning message"

    def test_check_log_msg_error(self):
        serial_interface = SerialInterface.__new__(SerialInterface)
        level, msg = serial_interface._check_log_msg("E)Error message")
        assert level == SerialInterface.LogLevel.ERROR
        assert msg == "Error message"

    def test_check_log_msg_no_prefix(self):
        serial_interface = SerialInterface.__new__(SerialInterface)
        level, msg = serial_interface._check_log_msg("Regular message")
        assert level is None
        assert msg == "Regular message"

    def test_check_log_msg_too_short(self):
        serial_interface = SerialInterface.__new__(SerialInterface)
        level, msg = serial_interface._check_log_msg("X")
        assert level is None
        assert msg == ""

    def test_handle_line_log_message(self, mocker):
        serial_interface = SerialInterface.__new__(SerialInterface)
        serial_interface._lock = mocker.MagicMock()
        serial_interface._waiting_for_response = False
        callback_mock = Mock()
        serial_interface.log_message_callback = callback_mock

        serial_interface._handle_line("I)Test log message")

        callback_mock.assert_called_once_with(
            SerialInterface.LogLevel.INFO, "Test log message"
        )

    def test_handle_line_ok_response(self, mocker):
        serial_interface = SerialInterface.__new__(SerialInterface)
        serial_interface._lock = mocker.MagicMock()
        serial_interface._condition = mocker.MagicMock()
        serial_interface._waiting_for_response = True
        serial_interface._response_status = None
        serial_interface._response_string = ""
        serial_interface.log_message_callback = None

        serial_interface._handle_line("ok")

        assert serial_interface._response_status == SerialInterface.ReplyStatus.OK
        serial_interface._condition.notify.assert_called_once()

    def test_handle_line_error_response(self, mocker):
        serial_interface = SerialInterface.__new__(SerialInterface)
        serial_interface._lock = mocker.MagicMock()
        serial_interface._condition = mocker.MagicMock()
        serial_interface._waiting_for_response = True
        serial_interface._response_status = None
        serial_interface._response_string = ""
        serial_interface._response_error_msg = None
        serial_interface.log_message_callback = None

        serial_interface._handle_line("error: Invalid command")

        assert serial_interface._response_status == SerialInterface.ReplyStatus.ERROR
        assert serial_interface._response_error_msg == "Invalid command"
        serial_interface._condition.notify.assert_called_once()

    def test_handle_line_busy_response(self, mocker):
        serial_interface = SerialInterface.__new__(SerialInterface)
        serial_interface._lock = mocker.MagicMock()
        serial_interface._condition = mocker.MagicMock()
        serial_interface._waiting_for_response = True
        serial_interface._response_status = None
        serial_interface._response_string = ""
        serial_interface.log_message_callback = None

        serial_interface._handle_line("busy")

        assert serial_interface._response_status == SerialInterface.ReplyStatus.BUSY
        serial_interface._condition.notify.assert_called_once()

    def test_handle_line_data_before_status(self, mocker):
        serial_interface = SerialInterface.__new__(SerialInterface)
        serial_interface._lock = mocker.MagicMock()
        serial_interface._condition = mocker.MagicMock()
        serial_interface._waiting_for_response = True
        serial_interface._response_status = None
        serial_interface._response_string = ""
        serial_interface.log_message_callback = None

        serial_interface._handle_line("X:10.5 Y:20.3 Z:5.1")
        serial_interface._handle_line("ok")

        assert serial_interface._response_string == "X:10.5 Y:20.3 Z:5.1\n"
        assert serial_interface._response_status == SerialInterface.ReplyStatus.OK

    def test_handle_line_unsolicited_message(self, mocker):
        serial_interface = SerialInterface.__new__(SerialInterface)
        serial_interface._lock = mocker.MagicMock()
        serial_interface._waiting_for_response = False
        callback_mock = Mock()
        serial_interface.log_message_callback = None
        serial_interface.unsolicited_msg_callback = callback_mock

        serial_interface._handle_line("Unsolicited message")

        callback_mock.assert_called_once_with("Unsolicited message")

    def test_send_command_serial_not_open(self, mocker):
        serial_interface = SerialInterface.__new__(SerialInterface)
        serial_interface._lock = mocker.MagicMock()
        serial_interface.serial = None

        status, response = serial_interface.send_command("G28")

        assert status == SerialInterface.ReplyStatus.ERROR
        assert response == "Serial not open"

    def test_send_command_timeout(self, mocker):
        serial_interface = SerialInterface.__new__(SerialInterface)
        serial_interface._lock = mocker.MagicMock()
        serial_interface._condition = mocker.MagicMock()
        mock_serial_obj = MagicMock()
        mock_serial_obj.is_open = True
        serial_interface.serial = mock_serial_obj
        serial_interface.command_msg_callback = None
        serial_interface._response_status = None
        serial_interface._waiting_for_response = False

        serial_interface._condition.wait.side_effect = lambda timeout: time.sleep(
            timeout
        )

        status, response = serial_interface.send_command("G28", timeout=0.1)

        assert status == SerialInterface.ReplyStatus.TIMEOUT
        assert serial_interface._waiting_for_response is False
