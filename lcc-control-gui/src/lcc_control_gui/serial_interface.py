from __future__ import annotations

import platform
import threading
import time
from enum import Enum
from glob import glob
from typing import NamedTuple

import serial
from colorama import Fore, Style
from serial.tools import list_ports


class SerialInterface:
    class ReplyStatus(Enum):
        OK = "ok"
        ERROR = "error"
        TIMEOUT = "timeout"
        BUSY = "busy"

    class LogLevel(Enum):
        DEBUG = "debug"
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"

    log_level_prefix_map = {
        "D)": LogLevel.DEBUG,
        "I)": LogLevel.INFO,
        "W)": LogLevel.WARNING,
        "E)": LogLevel.ERROR,
    }

    def __init__(
        self,
        port: str,
        baud_rate: int = 115200,
        command_msg_callback=None,
        log_msg_callback=None,
        unsolicited_msg_callback=None,
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
        self._startup_complete = False
        self._running = True

        self.connect(self.reconnect_timeout)

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        self._wait_for_startup()

    def connect(self, timeout):
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
            try:
                ser = self.serial
                if ser is not None and ser.is_open:
                    if ser.in_waiting:
                        char = ser.read(1).decode("ascii", errors="ignore")
                        if char in ["\n", "\r"]:
                            if len(buffer) > 0:
                                self._handle_line(buffer)
                                buffer = ""
                        else:
                            buffer += char
                    else:
                        time.sleep(0.001)
                else:
                    time.sleep(0.001)
            except (serial.SerialException, OSError, TypeError) as e:
                print(
                    Fore.MAGENTA
                    + f"[SerialInterface] Lost connection: {e}"
                    + Style.RESET_ALL
                )
                try:
                    if self.serial is not None and self.serial.is_open:
                        self.serial.close()
                except Exception:
                    pass

                self.serial = None
                self.connect(self.reconnect_timeout)

    def _handle_line(self, line: str):
        with self._lock:
            log_level, log_msg = self._check_log_msg(line)

            if log_level is not None:
                if self.log_message_callback:
                    self.log_message_callback(log_level, log_msg)
            elif self._waiting_for_response:
                line_lower = line.lower()
                if line_lower.startswith("ok"):
                    self._response_status = SerialInterface.ReplyStatus.OK
                    self._condition.notify()
                elif line_lower.startswith("busy"):
                    self._response_status = SerialInterface.ReplyStatus.BUSY
                    self._condition.notify()
                elif line_lower.startswith("error"):
                    self._response_status = SerialInterface.ReplyStatus.ERROR
                    parts = line.split(":", 1)
                    self._response_error_msg = (
                        parts[1].strip() if len(parts) > 1 else ""
                    )
                    self._condition.notify()
                else:
                    self._response_string += line + "\n"
            else:
                if line.lower().startswith("start") or line.startswith("Marlin"):
                    self._startup_complete = True
                    self._condition.notify_all()
                if self.unsolicited_msg_callback:
                    self.unsolicited_msg_callback(line)

    def _check_log_msg(self, msg: str):
        if len(msg) < 2:
            return None, ""
        log_level = self.log_level_prefix_map.get(msg[:2])
        return log_level, msg[2:] if log_level else msg

    def _wait_for_startup(self, timeout=3):
        with self._lock:
            deadline = time.time() + timeout
            while not self._startup_complete:
                remaining = deadline - time.time()
                if remaining <= 0:
                    print(
                        Fore.YELLOW + "[SerialInterface] No startup banner received, "
                        "checking if device is ready..." + Style.RESET_ALL
                    )
                    break
                self._condition.wait(timeout=remaining)

        if not self._startup_complete:
            status, response = self.send_command("M115", timeout=2)
            if status == SerialInterface.ReplyStatus.OK:
                print(
                    Fore.GREEN
                    + "[SerialInterface] Device ready (via M115)"
                    + Style.RESET_ALL
                )
                with self._lock:
                    self._startup_complete = True
            else:
                print(
                    Fore.RED
                    + "[SerialInterface] Device not responding"
                    + Style.RESET_ALL
                )
        else:
            print(Fore.GREEN + "[SerialInterface] Device ready" + Style.RESET_ALL)

    def send_command(self, cmd: str, timeout=2) -> tuple[ReplyStatus, str]:
        with self._lock:
            if not self.serial or not self.serial.is_open:
                return SerialInterface.ReplyStatus.ERROR, "Serial not open"

            self._waiting_for_response = True
            self._response_string = ""
            self._response_error_msg = ""
            self._response_status = None

            cmd = cmd.strip() + "\n"
            print(
                Fore.CYAN
                + f"[{self.__class__.__qualname__}] TX: {cmd.strip()}"
                + Style.RESET_ALL
            )
            if self.command_msg_callback:
                self.command_msg_callback(cmd, None, "")

            self.serial.write(cmd.encode("ascii"))
            self.serial.flush()

            end_time = time.time() + timeout
            while self._response_status is None:
                remaining = end_time - time.time()
                if remaining <= 0:
                    self._waiting_for_response = False
                    msg = (
                        f"[{self.__class__.__qualname__}] Command "
                        + "timeout, device didn't reply"
                    )
                    print(Fore.MAGENTA + msg + Style.RESET_ALL)
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
            Fore.MAGENTA
            + f"[SerialInterface] Disconnecting from port '{self.port}'..."
            + Style.RESET_ALL
        )
        self._running = False
        if self.serial and self.serial.is_open:
            self.serial.close()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        print(
            Fore.GREEN + "[SerialInterface] Disconnected successfully" + Style.RESET_ALL
        )


def scan_ports():
    """
    Scan for available serial ports on the current platform

    Returns:
        list: List of available serial port names
    """
    print(
        Fore.WHITE + "[SerialInterface] Scanning for serial ports..." + Style.RESET_ALL
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


def is_port_available(port):
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
