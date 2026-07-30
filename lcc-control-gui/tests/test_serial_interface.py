from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, Mock, patch

import pytest
import serial

from lcc_control_gui.serial_interface import SerialInterface

ConnectionStatus = SerialInterface.ConnectionStatus


@pytest.fixture
def mock_serial():
    with patch("lcc_control_gui.serial_interface.serial.Serial") as mock:
        yield mock


def _new_interface() -> SerialInterface:
    """Bare instance for poking internals directly, bypassing __init__/hardware."""
    return SerialInterface.__new__(SerialInterface)


class TestSerialInterface:
    def test_check_log_msg_debug(self):
        serial_interface = _new_interface()
        level, msg = serial_interface._check_log_msg("D)Debug message")
        assert level == SerialInterface.LogLevel.DEBUG
        assert msg == "Debug message"

    def test_check_log_msg_info(self):
        serial_interface = _new_interface()
        level, msg = serial_interface._check_log_msg("I)Info message")
        assert level == SerialInterface.LogLevel.INFO
        assert msg == "Info message"

    def test_check_log_msg_warning(self):
        serial_interface = _new_interface()
        level, msg = serial_interface._check_log_msg("W)Warning message")
        assert level == SerialInterface.LogLevel.WARNING
        assert msg == "Warning message"

    def test_check_log_msg_error(self):
        serial_interface = _new_interface()
        level, msg = serial_interface._check_log_msg("E)Error message")
        assert level == SerialInterface.LogLevel.ERROR
        assert msg == "Error message"

    def test_check_log_msg_no_prefix(self):
        serial_interface = _new_interface()
        level, msg = serial_interface._check_log_msg("Regular message")
        assert level is None
        assert msg == "Regular message"

    def test_check_log_msg_too_short(self):
        serial_interface = _new_interface()
        level, msg = serial_interface._check_log_msg("X")
        assert level is None
        assert msg == ""


class TestClassifyLine:
    """Pure classification logic - no lock/condition mocking needed."""

    @pytest.fixture
    def serial_interface(self):
        si = _new_interface()
        si._greetings = ("start", "marlin", "grbl ")
        return si

    def test_log_prefix_debug(self, serial_interface):
        c = serial_interface._classify_line("D)Debug message")
        assert c.kind == SerialInterface.LineKind.LOG
        assert c.log_level == SerialInterface.LogLevel.DEBUG
        assert c.log_text == "Debug message"
        assert c.is_online_signal is False

    def test_ok_response(self, serial_interface):
        c = serial_interface._classify_line("ok")
        assert c.kind == SerialInterface.LineKind.STATUS_OK
        assert c.is_online_signal is True

    def test_ok_response_uppercase(self, serial_interface):
        c = serial_interface._classify_line("OK")
        assert c.kind == SerialInterface.LineKind.STATUS_OK

    def test_bare_busy_is_not_a_real_marlin_line(self, serial_interface):
        # Real Marlin always sends "echo:busy: ..." - a bare "busy" prefix
        # with no colon is not part of the actual protocol, so it must not
        # be classified as a busy status (it falls through to PLAIN).
        c = serial_interface._classify_line("busy")
        assert c.kind != SerialInterface.LineKind.STATUS_BUSY

    def test_marlin_echo_busy_processing(self, serial_interface):
        c = serial_interface._classify_line("echo:busy: processing")
        assert c.kind == SerialInterface.LineKind.STATUS_BUSY
        assert c.is_online_signal is True

    def test_marlin_echo_busy_paused_for_user(self, serial_interface):
        c = serial_interface._classify_line("echo:busy: paused for user")
        assert c.kind == SerialInterface.LineKind.STATUS_BUSY
        assert c.is_online_signal is True

    def test_error_response(self, serial_interface):
        c = serial_interface._classify_line("error:Invalid command")
        assert c.kind == SerialInterface.LineKind.STATUS_ERROR
        assert c.error_detail == "Invalid command"
        assert c.is_halt_signal is False
        assert c.is_online_signal is True

    def test_halt_message(self, serial_interface):
        c = serial_interface._classify_line("Error:Printer halted. kill() called!")
        assert c.kind == SerialInterface.LineKind.STATUS_ERROR
        assert c.is_halt_signal is True

    @pytest.mark.parametrize("greeting", ["start", "Marlin 2.1.2", "Grbl 1.1h"])
    def test_greetings_are_online_signals(self, serial_interface, greeting):
        c = serial_interface._classify_line(greeting)
        assert c.kind == SerialInterface.LineKind.PLAIN
        assert c.is_online_signal is True

    def test_temperature_line_is_online_signal(self, serial_interface):
        c = serial_interface._classify_line("T:200.0 /200.0")
        assert c.kind == SerialInterface.LineKind.PLAIN
        assert c.is_online_signal is True

    def test_unanchored_t_colon_does_not_false_positive(self, serial_interface):
        c = serial_interface._classify_line("format:something")
        assert c.kind == SerialInterface.LineKind.PLAIN
        assert c.is_online_signal is False

    def test_arbitrary_text(self, serial_interface):
        c = serial_interface._classify_line("hello world")
        assert c.kind == SerialInterface.LineKind.PLAIN
        assert c.is_online_signal is False


class TestDispatchClassifiedLine:
    @pytest.fixture
    def serial_interface(self, mocker):
        si = _new_interface()
        si._condition = mocker.MagicMock()
        si._waiting_for_response = False
        si._response_string = ""
        si._response_status = None
        si._response_error_msg = None
        si._connection_status = ConnectionStatus.DISCONNECTED
        si.log_message_callback = None
        si.unsolicited_msg_callback = None
        return si

    def test_log_kind_invokes_callback_only(self, serial_interface):
        callback_mock = Mock()
        serial_interface.log_message_callback = callback_mock
        c = SerialInterface.ClassifiedLine(
            "I)hi",
            SerialInterface.LineKind.LOG,
            log_level=SerialInterface.LogLevel.INFO,
            log_text="hi",
        )

        serial_interface._dispatch_classified_line(c)

        callback_mock.assert_called_once_with(SerialInterface.LogLevel.INFO, "hi")
        assert serial_interface._connection_status == ConnectionStatus.DISCONNECTED

    def test_status_ok_while_waiting_resolves_command_and_marks_online(
        self, serial_interface
    ):
        serial_interface._waiting_for_response = True
        c = SerialInterface.ClassifiedLine(
            "ok", SerialInterface.LineKind.STATUS_OK, is_online_signal=True
        )

        serial_interface._dispatch_classified_line(c)

        assert serial_interface._response_status == SerialInterface.ReplyStatus.OK
        assert serial_interface._connection_status == ConnectionStatus.ONLINE
        serial_interface._condition.notify.assert_called_once()

    def test_status_busy_while_waiting(self, serial_interface):
        serial_interface._waiting_for_response = True
        c = SerialInterface.ClassifiedLine(
            "echo:busy: processing", SerialInterface.LineKind.STATUS_BUSY
        )

        serial_interface._dispatch_classified_line(c)

        assert serial_interface._response_status == SerialInterface.ReplyStatus.BUSY
        serial_interface._condition.notify.assert_called_once()

    def test_status_error_while_waiting(self, serial_interface):
        serial_interface._waiting_for_response = True
        c = SerialInterface.ClassifiedLine(
            "error:Invalid command",
            SerialInterface.LineKind.STATUS_ERROR,
            error_detail="Invalid command",
        )

        serial_interface._dispatch_classified_line(c)

        assert serial_interface._response_status == SerialInterface.ReplyStatus.ERROR
        assert serial_interface._response_error_msg == "Invalid command"
        assert serial_interface._connection_status == ConnectionStatus.DISCONNECTED
        serial_interface._condition.notify.assert_called_once()

    def test_halt_signal_while_waiting_sets_halted_and_resolves_error(
        self, serial_interface
    ):
        serial_interface._waiting_for_response = True
        log_callback = Mock()
        serial_interface.log_message_callback = log_callback
        c = SerialInterface.ClassifiedLine(
            "Error:Printer halted. kill() called!",
            SerialInterface.LineKind.STATUS_ERROR,
            error_detail="Printer halted. kill() called!",
            is_halt_signal=True,
        )

        serial_interface._dispatch_classified_line(c)

        assert serial_interface._connection_status == ConnectionStatus.HALTED
        assert serial_interface._response_status == SerialInterface.ReplyStatus.ERROR
        log_callback.assert_called_once_with(
            SerialInterface.LogLevel.ERROR, "Error:Printer halted. kill() called!"
        )

    def test_halt_signal_while_not_waiting_still_notifies_and_forwards_unsolicited(
        self, serial_interface
    ):
        log_callback = Mock()
        unsolicited_callback = Mock()
        serial_interface.log_message_callback = log_callback
        serial_interface.unsolicited_msg_callback = unsolicited_callback
        c = SerialInterface.ClassifiedLine(
            "Error:Printer halted. kill() called!",
            SerialInterface.LineKind.STATUS_ERROR,
            error_detail="Printer halted. kill() called!",
            is_halt_signal=True,
        )

        serial_interface._dispatch_classified_line(c)

        assert serial_interface._connection_status == ConnectionStatus.HALTED
        log_callback.assert_called_once()
        unsolicited_callback.assert_called_once_with(
            "Error:Printer halted. kill() called!"
        )

    def test_plain_online_signal_while_not_waiting(self, serial_interface):
        callback_mock = Mock()
        serial_interface.unsolicited_msg_callback = callback_mock
        c = SerialInterface.ClassifiedLine(
            "start", SerialInterface.LineKind.PLAIN, is_online_signal=True
        )

        serial_interface._dispatch_classified_line(c)

        assert serial_interface._connection_status == ConnectionStatus.ONLINE
        callback_mock.assert_called_once_with("start")

    def test_plain_not_online_signal_still_forwarded_unsolicited(
        self, serial_interface
    ):
        callback_mock = Mock()
        serial_interface.unsolicited_msg_callback = callback_mock
        c = SerialInterface.ClassifiedLine(
            "Unsolicited message", SerialInterface.LineKind.PLAIN
        )

        serial_interface._dispatch_classified_line(c)

        assert serial_interface._connection_status == ConnectionStatus.DISCONNECTED
        callback_mock.assert_called_once_with("Unsolicited message")

    def test_plain_data_while_waiting_is_appended_to_response(self, serial_interface):
        serial_interface._waiting_for_response = True
        c = SerialInterface.ClassifiedLine(
            "X:10.5 Y:20.3 Z:5.1", SerialInterface.LineKind.PLAIN
        )

        serial_interface._dispatch_classified_line(c)

        assert serial_interface._response_string == "X:10.5 Y:20.3 Z:5.1\n"


class TestHandleLineEndToEnd:
    """End-to-end through _handle_line (classify + dispatch), matching the
    original test file's convention of exercising the public entry point."""

    @pytest.fixture
    def serial_interface(self, mocker):
        si = _new_interface()
        si._lock = mocker.MagicMock()
        si._condition = mocker.MagicMock()
        si._greetings = ("start", "marlin", "grbl ")
        si._connection_status = ConnectionStatus.DISCONNECTED
        si.log_message_callback = None
        si.unsolicited_msg_callback = None
        return si

    def test_handle_line_log_message(self, serial_interface):
        serial_interface._waiting_for_response = False
        callback_mock = Mock()
        serial_interface.log_message_callback = callback_mock

        serial_interface._handle_line("I)Test log message")

        callback_mock.assert_called_once_with(
            SerialInterface.LogLevel.INFO, "Test log message"
        )

    def test_handle_line_ok_response(self, serial_interface):
        serial_interface._waiting_for_response = True
        serial_interface._response_status = None
        serial_interface._response_string = ""

        serial_interface._handle_line("ok")

        assert serial_interface._response_status == SerialInterface.ReplyStatus.OK
        serial_interface._condition.notify.assert_called_once()

    def test_handle_line_error_response(self, serial_interface):
        serial_interface._waiting_for_response = True
        serial_interface._response_status = None
        serial_interface._response_string = ""
        serial_interface._response_error_msg = None

        serial_interface._handle_line("error: Invalid command")

        assert serial_interface._response_status == SerialInterface.ReplyStatus.ERROR
        assert serial_interface._response_error_msg == "Invalid command"
        serial_interface._condition.notify.assert_called_once()

    def test_handle_line_busy_response(self, serial_interface):
        serial_interface._waiting_for_response = True
        serial_interface._response_status = None
        serial_interface._response_string = ""

        serial_interface._handle_line("echo:busy: processing")

        assert serial_interface._response_status == SerialInterface.ReplyStatus.BUSY
        serial_interface._condition.notify.assert_called_once()

    def test_handle_line_data_before_status(self, serial_interface):
        serial_interface._waiting_for_response = True
        serial_interface._response_status = None
        serial_interface._response_string = ""

        serial_interface._handle_line("X:10.5 Y:20.3 Z:5.1")
        serial_interface._handle_line("ok")

        assert serial_interface._response_string == "X:10.5 Y:20.3 Z:5.1\n"
        assert serial_interface._response_status == SerialInterface.ReplyStatus.OK

    def test_handle_line_unsolicited_message(self, serial_interface):
        serial_interface._waiting_for_response = False
        callback_mock = Mock()
        serial_interface.unsolicited_msg_callback = callback_mock

        serial_interface._handle_line("Unsolicited message")

        callback_mock.assert_called_once_with("Unsolicited message")

    def test_handle_line_halt_then_greeting_recovers_online(self, serial_interface):
        serial_interface._waiting_for_response = False
        serial_interface._connection_status = ConnectionStatus.ONLINE

        serial_interface._handle_line("Error:Printer halted. kill() called!")
        assert serial_interface._connection_status == ConnectionStatus.HALTED

        serial_interface._handle_line("start")
        assert serial_interface._connection_status == ConnectionStatus.ONLINE


class TestConnectionStatus:
    def test_set_connection_status_no_op_when_unchanged(self, mocker):
        serial_interface = _new_interface()
        serial_interface._condition = mocker.MagicMock()
        serial_interface._connection_status = ConnectionStatus.ONLINE

        serial_interface._set_connection_status(ConnectionStatus.ONLINE)

        serial_interface._condition.notify_all.assert_not_called()

    def test_set_connection_status_notifies_on_transition(self, mocker):
        serial_interface = _new_interface()
        serial_interface._condition = mocker.MagicMock()
        serial_interface._connection_status = ConnectionStatus.DISCONNECTED

        serial_interface._set_connection_status(ConnectionStatus.ONLINE)

        assert serial_interface._connection_status == ConnectionStatus.ONLINE
        serial_interface._condition.notify_all.assert_called_once()

    def test_set_connection_status_can_transition_from_halted(self, mocker):
        serial_interface = _new_interface()
        serial_interface._condition = mocker.MagicMock()
        serial_interface._connection_status = ConnectionStatus.HALTED

        serial_interface._set_connection_status(ConnectionStatus.ONLINE)

        assert serial_interface._connection_status == ConnectionStatus.ONLINE

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (ConnectionStatus.ONLINE, True),
            (ConnectionStatus.DISCONNECTED, False),
            (ConnectionStatus.CONNECTING, False),
            (ConnectionStatus.HALTED, False),
        ],
    )
    def test_is_online_property(self, status, expected):
        serial_interface = _new_interface()
        serial_interface._connection_status = status
        assert serial_interface.is_online is expected

    def test_connection_status_property(self):
        serial_interface = _new_interface()
        serial_interface._connection_status = ConnectionStatus.HALTED
        assert serial_interface.connection_status == ConnectionStatus.HALTED


class TestReaderLoopIteration:
    def test_accumulates_line_and_calls_handle_line_once(self, mocker):
        serial_interface = _new_interface()
        mock_ser = MagicMock()
        mock_ser.is_open = True
        type(mock_ser).in_waiting = mocker.PropertyMock(side_effect=[1, 1, 1])
        mock_ser.read.side_effect = [b"o", b"k", b"\n"]
        serial_interface.serial = mock_ser
        serial_interface._handle_line = Mock()

        buffer = ""
        buffer = serial_interface._reader_loop_iteration(buffer)
        buffer = serial_interface._reader_loop_iteration(buffer)
        buffer = serial_interface._reader_loop_iteration(buffer)

        serial_interface._handle_line.assert_called_once_with("ok")
        assert buffer == ""

    def test_handle_line_exception_is_isolated_from_connection_handling(self, mocker):
        # A bug in classify/dispatch must not be mistaken for a lost
        # connection - it should be logged and the reader kept running,
        # not trigger a reconnect.
        serial_interface = _new_interface()
        serial_interface._connection_status = ConnectionStatus.ONLINE
        mock_ser = MagicMock()
        mock_ser.is_open = True
        type(mock_ser).in_waiting = mocker.PropertyMock(side_effect=[1, 1])
        mock_ser.read.side_effect = [b"x", b"\n"]
        serial_interface.serial = mock_ser
        serial_interface._handle_line = Mock(side_effect=RuntimeError("boom"))
        serial_interface.connect = Mock()
        serial_interface._trigger_online_handshake = Mock()

        buffer = ""
        buffer = serial_interface._reader_loop_iteration(buffer)
        buffer = serial_interface._reader_loop_iteration(buffer)

        assert buffer == ""
        assert serial_interface._connection_status == ConnectionStatus.ONLINE
        assert serial_interface.serial is mock_ser
        serial_interface.connect.assert_not_called()
        serial_interface._trigger_online_handshake.assert_not_called()

    def test_exception_sets_disconnected_and_nulls_serial(self, mocker):
        serial_interface = _new_interface()
        serial_interface._lock = mocker.MagicMock()
        serial_interface._condition = mocker.MagicMock()
        serial_interface._connection_status = ConnectionStatus.ONLINE
        mock_ser = MagicMock()
        mock_ser.is_open = True
        type(mock_ser).in_waiting = mocker.PropertyMock(
            side_effect=serial.SerialException("boom")
        )
        serial_interface.serial = mock_ser
        serial_interface.reconnect_timeout = 5
        serial_interface.connect = Mock(return_value=False)
        serial_interface._trigger_online_handshake = Mock()

        result = serial_interface._reader_loop_iteration("partial")

        assert serial_interface._connection_status == ConnectionStatus.DISCONNECTED
        assert serial_interface.serial is None
        serial_interface.connect.assert_called_once_with(5)
        assert result == ""

    def test_exception_reconnect_success_triggers_handshake(self, mocker):
        serial_interface = _new_interface()
        serial_interface._lock = mocker.MagicMock()
        serial_interface._condition = mocker.MagicMock()
        serial_interface._connection_status = ConnectionStatus.ONLINE
        mock_ser = MagicMock()
        mock_ser.is_open = True
        type(mock_ser).in_waiting = mocker.PropertyMock(side_effect=OSError("boom"))
        serial_interface.serial = mock_ser
        serial_interface.reconnect_timeout = 5
        serial_interface.connect = Mock(return_value=True)
        serial_interface._trigger_online_handshake = Mock()

        serial_interface._reader_loop_iteration("")

        assert serial_interface._connection_status == ConnectionStatus.CONNECTING
        serial_interface._trigger_online_handshake.assert_called_once()

    def test_exception_reconnect_failure_does_not_trigger_handshake(self, mocker):
        serial_interface = _new_interface()
        serial_interface._lock = mocker.MagicMock()
        serial_interface._condition = mocker.MagicMock()
        serial_interface._connection_status = ConnectionStatus.ONLINE
        mock_ser = MagicMock()
        mock_ser.is_open = True
        type(mock_ser).in_waiting = mocker.PropertyMock(side_effect=OSError("boom"))
        serial_interface.serial = mock_ser
        serial_interface.reconnect_timeout = 5
        serial_interface.connect = Mock(return_value=False)
        serial_interface._trigger_online_handshake = Mock()

        serial_interface._reader_loop_iteration("")

        assert serial_interface._connection_status == ConnectionStatus.DISCONNECTED
        serial_interface._trigger_online_handshake.assert_not_called()


class TestTriggerOnlineHandshake:
    def test_starts_thread_when_none_running(self):
        serial_interface = _new_interface()
        serial_interface._lock = threading.Lock()
        serial_interface._online_wait_thread = None
        serial_interface._wait_until_online = Mock()

        serial_interface._trigger_online_handshake()
        serial_interface._online_wait_thread.join(timeout=1.0)

        serial_interface._wait_until_online.assert_called_once()

    def test_skips_when_thread_already_alive(self):
        serial_interface = _new_interface()
        serial_interface._lock = threading.Lock()
        fake_thread = Mock()
        fake_thread.is_alive.return_value = True
        serial_interface._online_wait_thread = fake_thread
        serial_interface._wait_until_online = Mock()

        serial_interface._trigger_online_handshake()

        assert serial_interface._online_wait_thread is fake_thread
        serial_interface._wait_until_online.assert_not_called()


class TestDrainUntilQuiet:
    def test_returns_promptly_when_already_quiet(self):
        serial_interface = _new_interface()
        serial_interface._lock = threading.Lock()
        serial_interface._condition = threading.Condition(serial_interface._lock)
        serial_interface._last_line_at = time.time() - 1.0

        start = time.time()
        serial_interface._drain_until_quiet(quiet=0.3, timeout=2.0)
        elapsed = time.time() - start

        assert elapsed < 0.2

    def test_respects_timeout_when_never_quiet(self):
        serial_interface = _new_interface()
        serial_interface._lock = threading.Lock()
        serial_interface._condition = threading.Condition(serial_interface._lock)
        serial_interface._last_line_at = time.time()

        start = time.time()
        serial_interface._drain_until_quiet(quiet=5.0, timeout=0.2)
        elapsed = time.time() - start

        assert 0.15 <= elapsed <= 0.6


class TestSendCommand:
    def test_send_command_serial_not_open(self, mocker):
        serial_interface = _new_interface()
        serial_interface._lock = mocker.MagicMock()
        serial_interface.serial = None

        status, response = serial_interface.send_command("G28")

        assert status == SerialInterface.ReplyStatus.ERROR
        assert response == "Serial not open"

    def test_send_command_timeout(self, mocker):
        serial_interface = _new_interface()
        serial_interface._lock = mocker.MagicMock()
        serial_interface._condition = mocker.MagicMock()
        mock_serial_obj = MagicMock()
        mock_serial_obj.is_open = True
        serial_interface.serial = mock_serial_obj
        serial_interface.command_msg_callback = None
        serial_interface._response_status = None
        serial_interface._waiting_for_response = False
        serial_interface._connection_status = ConnectionStatus.ONLINE

        serial_interface._condition.wait.side_effect = lambda timeout: time.sleep(
            timeout
        )

        status, response = serial_interface.send_command("G28", timeout=0.1)

        assert status == SerialInterface.ReplyStatus.TIMEOUT
        assert serial_interface._waiting_for_response is False

    def test_send_command_returns_error_immediately_when_halted(self, mocker):
        serial_interface = _new_interface()
        serial_interface._lock = mocker.MagicMock()
        serial_interface._condition = mocker.MagicMock()
        mock_serial_obj = MagicMock()
        mock_serial_obj.is_open = True
        serial_interface.serial = mock_serial_obj
        serial_interface._connection_status = ConnectionStatus.HALTED

        status, msg = serial_interface.send_command("G28")

        assert status == SerialInterface.ReplyStatus.ERROR
        assert "halted" in msg.lower()
        mock_serial_obj.write.assert_not_called()
        serial_interface._condition.wait.assert_not_called()

    def test_send_command_flushes_input_before_write(self, mocker):
        serial_interface = _new_interface()
        serial_interface._lock = mocker.MagicMock()
        serial_interface._condition = mocker.MagicMock()
        serial_interface._condition.wait.side_effect = lambda timeout: time.sleep(
            timeout
        )
        mock_serial_obj = MagicMock()
        mock_serial_obj.is_open = True
        serial_interface.serial = mock_serial_obj
        serial_interface.command_msg_callback = None
        serial_interface._connection_status = ConnectionStatus.ONLINE

        serial_interface.send_command("G28", timeout=0.05)

        calls = mock_serial_obj.mock_calls
        reset_index = calls.index(mocker.call.reset_input_buffer())
        write_index = calls.index(mocker.call.write(b"G28\n"))
        assert reset_index < write_index

    def test_send_command_reset_input_buffer_failure_does_not_block_write(self, mocker):
        serial_interface = _new_interface()
        serial_interface._lock = mocker.MagicMock()
        serial_interface._condition = mocker.MagicMock()
        serial_interface._condition.wait.side_effect = lambda timeout: time.sleep(
            timeout
        )
        mock_serial_obj = MagicMock()
        mock_serial_obj.is_open = True
        mock_serial_obj.reset_input_buffer.side_effect = serial.SerialException("boom")
        serial_interface.serial = mock_serial_obj
        serial_interface.command_msg_callback = None
        serial_interface._connection_status = ConnectionStatus.ONLINE

        serial_interface.send_command("G28", timeout=0.05)

        mock_serial_obj.write.assert_called_once_with(b"G28\n")
