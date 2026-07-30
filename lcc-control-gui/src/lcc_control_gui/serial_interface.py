from __future__ import annotations

import contextlib
import platform
import re
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from glob import glob
from typing import TYPE_CHECKING, NamedTuple

import serial
from colorama import Fore, Style
from serial.tools import list_ports

if TYPE_CHECKING:
    from collections.abc import Callable

# Marlin serial protocol reference (what this module parses for)
# ----------------------------------------------------------------------
# Marlin (and Marlin-compatible firmware) speaks a line-based, free-text
# protocol over serial - just '\n'/'\r'-terminated lines, no framing, no
# command IDs. Replies are matched to the command that triggered them
# purely by arrival order. The line shapes this module cares about:
#
#   Boot greeting (once, right after reset/power-up/DTR-toggle):
#       start
#       Marlin 2.1.2
#       echo: Last Updated: ... | Author: ...
#       echo:Compiled: ...
#       echo: Free Memory: ...  PlannerBufferBytes: ...
#       echo:Hardcoded Default Settings Loaded
#
#   Command acknowledgement (after (almost) every processed line):
#       ok
#       ok N123 P15 B3                            (M110 line numbering / planner info)
#       ok T:210.00 /210.00 B:60.00 /60.00 @:127 B@:0   (e.g. reply to M105)
#
#   Temperature report (M105), independent of the "ok" that follows it:
#       T:210.00 /210.00 B:60.00 /60.00 @:127 B@:0
#
#   Busy/keepalive (HOST_KEEPALIVE_FEATURE; sent every couple of seconds
#   while a long-running command blocks the queue - no "ok" until done):
#       echo:busy: processing
#       echo:busy: paused for user
#       echo:busy: paused for input
#   Real Marlin always includes the "echo:" prefix and the colon after
#   "busy" - there is no bare "busy" line. This matches Printrun/
#   Pronterface's own line-ignore regex: `.*busy: ?processing|.*busy: ?heating`.
#
#   Recoverable command error (that one command failed, next one is fine):
#       Error:Invalid command
#       Error:Unknown command: "FOO"
#
#   Halt / kill (M112, thermal runaway, hard fault - device stops
#   responding to *everything*, including further status queries, until
#   a physical reset re-triggers the boot greeting above):
#       Error:Printer halted. kill() called!
#
#   Line-number/checksum resend request (only relevant if the host uses
#   M110 + line numbers + checksums; this class currently does not):
#       Resend: 123


class SerialInterface:
    class ReplyStatus(StrEnum):
        OK = "ok"
        ERROR = "error"
        TIMEOUT = "timeout"
        BUSY = "busy"

    class LogLevel(StrEnum):
        DEBUG = "debug"
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"

    class ConnectionStatus(StrEnum):
        DISCONNECTED = "disconnected"
        CONNECTING = "connecting"
        ONLINE = "online"
        HALTED = "halted"

    class LineKind(StrEnum):
        LOG = "log"
        STATUS_OK = "status_ok"
        STATUS_BUSY = "status_busy"
        STATUS_ERROR = "status_error"
        PLAIN = "plain"

    @dataclass(frozen=True, slots=True)
    class ClassifiedLine:
        raw: str
        kind: SerialInterface.LineKind
        log_level: SerialInterface.LogLevel | None = None
        log_text: str = ""
        error_detail: str = ""
        is_online_signal: bool = False
        is_halt_signal: bool = False

    log_level_prefix_map = {
        "D)": LogLevel.DEBUG,
        "I)": LogLevel.INFO,
        "W)": LogLevel.WARNING,
        "E)": LogLevel.ERROR,
    }

    _temp_line_re = re.compile(r"\bt:")

    def __init__(
        self,
        port: str,
        baud_rate: int = 115200,
        command_msg_callback: Callable | None = None,
        log_msg_callback: Callable | None = None,
        unsolicited_msg_callback: Callable | None = None,
        reconnect_timeout: int = 5,
    ):
        self.port = port
        self.baud_rate = baud_rate
        self.reconnect_timeout = reconnect_timeout
        self.serial = None

        self.command_msg_callback = command_msg_callback
        self.log_message_callback = log_msg_callback
        self.unsolicited_msg_callback = unsolicited_msg_callback

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._waiting_for_response = False
        self._response_string = ""
        self._response_status = None
        self._response_error_msg = None
        self._connection_status = SerialInterface.ConnectionStatus.DISCONNECTED
        self._online_wait_thread: threading.Thread | None = None
        self._last_line_at = time.time()
        self._running = True

        # Greetings that indicate printer is ready (like Pronterface)
        self._greetings = ("start", "marlin", "grbl ")

        self.connect(self.reconnect_timeout)

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        self._wait_until_online()

    def connect(self, timeout: int):
        deadline = time.time() + timeout
        print(Fore.MAGENTA, end="")
        print(f"[SerialInterface] Connecting to port '{self.port}'...", end="")
        while time.time() < deadline:
            try:
                self.serial = serial.Serial(self.port, self.baud_rate, timeout=2)
                print(" [OK]")
                print(Style.RESET_ALL, end="")
                return True
            except (serial.SerialException, OSError):
                print(".", end="")
                time.sleep(0.2)

        print(f" [FAILED] Timeout after {timeout} seconds.")
        print("[SerialInterface] Connection is permanently closed")
        print(Style.RESET_ALL, end="")
        self.serial = None
        return False

    def _reader_loop(self):
        buffer = ""
        while self._running:
            buffer = self._reader_loop_iteration(buffer)

    def _reader_loop_iteration(self, buffer: str) -> str:
        try:
            ser = self.serial
            if ser is not None and ser.is_open:
                if ser.in_waiting:
                    char = ser.read(1).decode("ascii", errors="ignore")
                    if char in ["\n", "\r"]:
                        if len(buffer) > 0:
                            try:
                                self._handle_line(buffer)
                            except Exception as e:
                                print(
                                    f"{Fore.MAGENTA}[SerialInterface] Error handling "
                                    f"line {buffer!r}: {e}{Style.RESET_ALL}"
                                )
                            buffer = ""
                    else:
                        buffer += char
                else:
                    time.sleep(0.001)
            else:
                time.sleep(0.001)
        except (serial.SerialException, OSError, TypeError) as e:
            print(
                f"{Fore.MAGENTA}[SerialInterface] Lost connection: {e}{Style.RESET_ALL}"
            )
            with self._lock:
                self._set_connection_status(
                    SerialInterface.ConnectionStatus.DISCONNECTED
                )
            try:
                if self.serial is not None and self.serial.is_open:
                    self.serial.close()
            except Exception:
                pass

            self.serial = None
            if self.connect(self.reconnect_timeout):
                with self._lock:
                    self._set_connection_status(
                        SerialInterface.ConnectionStatus.CONNECTING
                    )
                self._trigger_online_handshake()
            buffer = ""

        return buffer

    def _set_connection_status(self, status: ConnectionStatus) -> None:
        """Update connection status. Caller must hold self._lock."""
        if status == self._connection_status:
            return
        self._connection_status = status
        self._condition.notify_all()

    def _trigger_online_handshake(self) -> None:
        with self._lock:
            thread = self._online_wait_thread
            if thread is not None and thread.is_alive():
                return
            self._online_wait_thread = threading.Thread(
                target=self._wait_until_online, daemon=True
            )
        self._online_wait_thread.start()

    @property
    def is_online(self) -> bool:
        return self._connection_status == SerialInterface.ConnectionStatus.ONLINE

    @property
    def connection_status(self) -> ConnectionStatus:
        return self._connection_status

    def _classify_line(self, line: str) -> ClassifiedLine:
        log_level, log_text = self._check_log_msg(line)
        if log_level is not None:
            return SerialInterface.ClassifiedLine(
                line,
                SerialInterface.LineKind.LOG,
                log_level=log_level,
                log_text=log_text,
            )

        line_lower = line.lower()
        if line_lower.startswith("ok"):
            return SerialInterface.ClassifiedLine(
                line, SerialInterface.LineKind.STATUS_OK, is_online_signal=True
            )
        if "busy:" in line_lower:
            return SerialInterface.ClassifiedLine(
                line, SerialInterface.LineKind.STATUS_BUSY, is_online_signal=True
            )
        if line_lower.startswith("error"):
            parts = line.split(":", 1)
            detail = parts[1].strip() if len(parts) > 1 else ""
            return SerialInterface.ClassifiedLine(
                line,
                SerialInterface.LineKind.STATUS_ERROR,
                error_detail=detail,
                is_online_signal=True,
                is_halt_signal="printer halted" in line_lower,
            )

        is_online = line_lower.startswith(self._greetings) or bool(
            self._temp_line_re.search(line_lower)
        )
        return SerialInterface.ClassifiedLine(
            line, SerialInterface.LineKind.PLAIN, is_online_signal=is_online
        )

    def _dispatch_classified_line(self, c: ClassifiedLine) -> None:
        self._last_line_at = time.time()
        self._condition.notify_all()

        if c.is_online_signal:
            self._set_connection_status(SerialInterface.ConnectionStatus.ONLINE)
        if c.is_halt_signal:
            self._set_connection_status(SerialInterface.ConnectionStatus.HALTED)
            if self.log_message_callback:
                self.log_message_callback(SerialInterface.LogLevel.ERROR, c.raw)

        if c.kind == SerialInterface.LineKind.LOG:
            if self.log_message_callback:
                self.log_message_callback(c.log_level, c.log_text)
            return

        if self._waiting_for_response:
            if c.kind == SerialInterface.LineKind.STATUS_OK:
                self._response_status = SerialInterface.ReplyStatus.OK
                self._condition.notify()
            elif c.kind == SerialInterface.LineKind.STATUS_BUSY:
                self._response_status = SerialInterface.ReplyStatus.BUSY
                self._condition.notify()
            elif c.kind == SerialInterface.LineKind.STATUS_ERROR:
                self._response_status = SerialInterface.ReplyStatus.ERROR
                self._response_error_msg = c.error_detail
                self._condition.notify()
            else:
                self._response_string += c.raw + "\n"
            return

        if self.unsolicited_msg_callback:
            self.unsolicited_msg_callback(c.raw)

    def _handle_line(self, line: str):
        classified = self._classify_line(line)
        with self._lock:
            self._dispatch_classified_line(classified)

    def _check_log_msg(self, msg: str):
        if len(msg) < 2:
            return None, ""
        log_level = self.log_level_prefix_map.get(msg[:2])
        return log_level, msg[2:] if log_level else msg

    def _wait_until_online(self, timeout: float = 10.0):
        """Wait for printer to come online using active polling (Pronterface-style).

        Sends M105 repeatedly and monitors for valid responses (ok, T:, or
        greeting messages). Tolerates empty lines and timing variations during
        bootloader/startup phase.
        """
        print(f"{Fore.YELLOW}[SerialInterface] Waiting for device...{Style.RESET_ALL}")

        deadline = time.time() + timeout

        while time.time() < deadline and not self.is_online:
            # Send M105 temperature query - Marlin always responds to this
            self._send_raw("M105")

            # Wait for response, checking periodically
            poll_deadline = time.time() + 2.0
            while time.time() < poll_deadline and not self.is_online:
                with self._lock:
                    # Check if we got a valid response
                    if self.is_online:
                        break
                    self._condition.wait(timeout=0.1)

            if self.is_online:
                break

        if self.is_online:
            self._drain_until_quiet()
            print(f"{Fore.GREEN}[SerialInterface] Device ready{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}[SerialInterface] Device not responding{Style.RESET_ALL}")

    def _drain_until_quiet(self, quiet: float = 0.3, timeout: float = 2.0) -> None:
        """Wait until no new line has been dispatched for `quiet` seconds.

        Used right after the online handshake to let any extra in-flight
        replies (e.g. from redundant M105 probes sent while the device was
        still booting) settle before normal command traffic begins, so they
        don't get misattributed as the response to the first real command.
        """
        with self._lock:
            deadline = time.time() + timeout
            while True:
                remaining_quiet = quiet - (time.time() - self._last_line_at)
                if remaining_quiet <= 0:
                    return
                remaining_total = deadline - time.time()
                if remaining_total <= 0:
                    return
                self._condition.wait(timeout=min(remaining_quiet, remaining_total))

    def _send_raw(self, cmd: str):
        """Send a command without waiting for response (for startup polling)."""
        if self.serial and self.serial.is_open:
            cmd = cmd.strip() + "\n"
            try:
                self.serial.write(cmd.encode("ascii"))
                self.serial.flush()
            except (serial.SerialException, OSError):
                pass

    def send_command(self, cmd: str, timeout: int = 30) -> tuple[ReplyStatus, str]:
        with self._lock:
            if not self.serial or not self.serial.is_open:
                return SerialInterface.ReplyStatus.ERROR, "Serial not open"

            if self._connection_status == SerialInterface.ConnectionStatus.HALTED:
                return (
                    SerialInterface.ReplyStatus.ERROR,
                    "Device halted (kill() called) - power-cycle or reset required",
                )

            self._waiting_for_response = True
            self._response_string = ""
            self._response_error_msg = ""
            self._response_status = None

            cmd = cmd.strip() + "\n"
            print(
                f"{Fore.CYAN}[{self.__class__.__qualname__}] TX: {cmd.strip()}"
                f"{Style.RESET_ALL}"
            )
            if self.command_msg_callback:
                self.command_msg_callback(cmd, None, "")

            with contextlib.suppress(serial.SerialException, OSError):
                self.serial.reset_input_buffer()

            self.serial.write(cmd.encode("ascii"))
            self.serial.flush()

            end_time = time.time() + timeout
            while (
                self._response_status is None
                or self._response_status == SerialInterface.ReplyStatus.BUSY
            ):
                # Reset timeout on BUSY responses (like Pronterface)
                if self._response_status == SerialInterface.ReplyStatus.BUSY:
                    end_time = time.time() + timeout
                    self._response_status = None

                remaining = end_time - time.time()
                if remaining <= 0:
                    self._waiting_for_response = False
                    msg = (
                        f"[{self.__class__.__qualname__}] Command "
                        "timeout, device didn't reply"
                    )
                    print(f"{Fore.MAGENTA}{msg}{Style.RESET_ALL}")
                    return (
                        SerialInterface.ReplyStatus.TIMEOUT,
                        self._response_string,
                    )
                self._condition.wait(timeout=remaining)

            self._waiting_for_response = False
            if self.command_msg_callback:
                self.command_msg_callback(
                    self._response_string,
                    self._response_status,
                    self._response_error_msg,
                )
            return self._response_status, self._response_string

    def close(self):
        print(
            f"{Fore.MAGENTA}[SerialInterface] Disconnecting from port '{self.port}'..."
            f"{Style.RESET_ALL}"
        )
        self._running = False
        if self.serial and self.serial.is_open:
            self.serial.close()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        print(
            f"{Fore.GREEN}[SerialInterface] Disconnected successfully{Style.RESET_ALL}"
        )


def scan_ports():
    """
    Scan for available serial ports on the current platform

    Returns:
        list: List of available serial port names
    """
    print(
        f"{Fore.WHITE}[SerialInterface] Scanning for serial ports...{Style.RESET_ALL}"
    )

    # Get a list of all port objects
    if hasattr(list_ports, "comports"):
        # Use the more detailed comports() function if available
        port_objects = list_ports.comports()

        # Extract port names from the objects and include description in the result
        ports: list[PortInfo] = []
        for port in port_objects:
            if port.device:
                port_info = PortInfo(
                    name=port.name,
                    device=port.device,
                    description=port.description,
                )
                ports.append(port_info)
                print(f"Found port: {port.device} - {port.description}")

        if ports:
            return ports

    # Fall back to platform-specific device globbing if comports() didn't work
    if platform.system() == "Windows":
        ports = [
            PortInfo(name=f"COM{i + 1}", device=f"COM{i + 1}", description="")
            for i in range(256)
        ]
    elif platform.system() in ("Linux", "Darwin") or platform.system().startswith(
        "CYGWIN"
    ):
        if platform.system() == "Darwin":
            # macOS
            ports = [
                PortInfo(name=port, device=port, description="")
                for port in glob("/dev/tty.*")
            ]
        else:
            # Linux or Cygwin
            ports = [
                PortInfo(name=port, device=port, description="")
                for port in glob("/dev/tty[A-Za-z]*")
            ]

    else:
        print(f"Unsupported platform: {platform.system()}")
        return []

    # Test each port to see if it's available
    result = []
    for port in ports:
        try:
            s = serial.Serial(port.name)
            s.close()
            result.append(port)
            print(f"[SerialInterface] Found port: {port}")
        except (OSError, serial.SerialException):
            pass

    print(f"[SerialInterface] Found {len(result)} ports")
    return result


def is_port_available(port: str):
    """
    Check if a specific port is available

    Args:
        port (str): The port name to check

    Returns:
        bool: True if the port is available, False otherwise
    """
    try:
        s = serial.Serial(port)
        s.close()
        return True
    except (OSError, serial.SerialException):
        return False


class PortInfo(NamedTuple):
    name: str
    device: str
    description: str
