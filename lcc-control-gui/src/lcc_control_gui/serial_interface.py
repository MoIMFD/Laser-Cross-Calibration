from __future__ import annotations

import threading
import time
from enum import Enum

import serial
from colorama import Fore, Style


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

        self.connect(self.reconnect_timeout)

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

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
        while True:
            try:
                if self.serial is not None and self.serial.in_waiting:
                    char = self.serial.read(1).decode("ascii", errors="ignore")
                    if char in ["\n", "\r"]:
                        if len(buffer) > 0:
                            self._handle_line(buffer)
                            buffer = ""
                    else:
                        buffer += char
                else:
                    time.sleep(0.001)
            except (serial.SerialException, OSError) as e:
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
                elif line_lower.startswith("busy"):
                    self._response_status = SerialInterface.ReplyStatus.BUSY
                elif line_lower.startswith("error"):
                    self._response_status = SerialInterface.ReplyStatus.ERROR
                    parts = line.split(":", 1)
                    self._response_error_msg = (
                        parts[1].strip() if len(parts) > 1 else ""
                    )

                if self._response_status is not None:
                    self._condition.notify()
                else:
                    self._response_string += line + "\n"
            else:
                if self.unsolicited_msg_callback:
                    self.unsolicited_msg_callback(line)

    def _check_log_msg(self, msg: str):
        if len(msg) < 2:
            return None, ""
        log_level = self.log_level_prefix_map.get(msg[:2])
        return log_level, msg[2:] if log_level else msg

    def send_command(self, cmd: str, timeout=2) -> tuple[ReplyStatus, str]:
        with self._lock:
            if not self.serial or not self.serial.is_open:
                return SerialInterface.ReplyStatus.ERROR, "Serial not open"

            self._waiting_for_response = True
            self._response_string = ""
            self._response_error_msg = ""
            self._response_status = None

            cmd = cmd.strip() + "\n"
            if self.command_msg_callback:
                self.command_msg_callback(cmd, None, "")

            self.serial.write(cmd.encode("ascii"))
            self.serial.flush()

            end_time = time.time() + timeout
            while self._response_status is None:
                remaining = end_time - time.time()
                if remaining <= 0:
                    self._waiting_for_response = False
                    msg = "[SerialInterface] Command timeout, device didn't reply"
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
        if self.serial and self.serial.is_open:
            self.serial.close()
