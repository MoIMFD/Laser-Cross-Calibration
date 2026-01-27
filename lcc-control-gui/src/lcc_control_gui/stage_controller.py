"""
Interface to the physical stage running Marlin. Marlins GCode commands
are listed here: https://marlinfw.org/docs/gcode/G000-G001.html

2025 Moritz Kluwe
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Literal

from lcc_control_gui.serial_interface import SerialInterface

NUMBER_REGEX = r"[-+]?\d*\.?\d+"


class StageController:
    def __init__(self, serial_interface: SerialInterface):
        self.serial = serial_interface
        self._gcode_thread = None
        self._stop_gcode = False

    def home_axes(
        self, axes: list[Literal["X", "Y", "Z"]] | None = None
    ) -> SerialInterface.ReplyStatus:
        if axes is None:
            axes = ["X", "Y", "Z"]

        cmd = "G28"
        for axis in axes:
            if axis.upper() in ["X", "Y", "Z"]:
                cmd += f" {axis.upper()}"

        status, _ = self.serial.send_command(cmd, timeout=30)
        return status

    def go_to_home(self, feedrate: float = 1000.0) -> SerialInterface.ReplyStatus:
        return self.move_absolute(0.0, 0.0, 0.0, feedrate)

    def move_relative(
        self, axis: str, distance: float, feedrate: float
    ) -> SerialInterface.ReplyStatus:
        axis = axis.upper()
        if axis not in ["X", "Y", "Z"]:
            return SerialInterface.ReplyStatus.ERROR

        self.serial.send_command("G91")
        cmd = f"G0 {axis}{distance:.3f} F{feedrate:.3f}"
        status, _ = self.serial.send_command(cmd)
        return status

    def move_absolute(
        self, x: float, y: float, z: float, feedrate: float
    ) -> SerialInterface.ReplyStatus:
        cmd = f"G90\nG0 X{x:.3f} Y{y:.3f} Z{z:.3f} F{feedrate:.3f}"
        status, _ = self.serial.send_command(cmd)
        return status

    def set_acceleration(self, accel: float) -> SerialInterface.ReplyStatus:
        cmd = f"M204 P{accel:.3f}"
        status, _ = self.serial.send_command(cmd)
        return status

    def set_max_feedrate(
        self, x: float, y: float, z: float
    ) -> SerialInterface.ReplyStatus:
        cmd = f"M203 X{x:.3f} Y{y:.3f} Z{z:.3f}"
        status, _ = self.serial.send_command(cmd)
        return status

    def read_position(self) -> tuple[float, float, float] | None:
        status, response = self.serial.send_command("M114")
        if status != SerialInterface.ReplyStatus.OK or len(response) == 0:
            return None

        match = re.search(
            rf"^X:({NUMBER_REGEX})\s*" rf"Y:({NUMBER_REGEX})\s*" rf"Z:({NUMBER_REGEX})",
            response,
            re.MULTILINE,
        )
        if not match:
            return None

        x, y, z = match.groups()
        return float(x), float(y), float(z)

    def run_gcode_file(
        self,
        filepath: str | Path,
        progress_callback=None,
        completion_callback=None,
    ) -> None:
        if self._gcode_thread and self._gcode_thread.is_alive():
            return

        self._stop_gcode = False
        self._gcode_thread = threading.Thread(
            target=self._run_gcode_thread,
            args=(filepath, progress_callback, completion_callback),
            daemon=True,
        )
        self._gcode_thread.start()

    def _run_gcode_thread(
        self, filepath: str | Path, progress_callback=None, completion_callback=None
    ):
        filepath = Path(filepath)
        if not filepath.exists():
            if completion_callback:
                completion_callback(SerialInterface.ReplyStatus.ERROR)
            return

        with filepath.open() as f:
            lines = f.readlines()

        total_lines = len(lines)
        for i, line in enumerate(lines):
            if self._stop_gcode:
                if completion_callback:
                    completion_callback(SerialInterface.ReplyStatus.ERROR)
                return

            line = line.strip()

            if not line or line.startswith(";"):
                continue

            status = self.send_gcode_line(line)
            if status == SerialInterface.ReplyStatus.ERROR:
                if completion_callback:
                    completion_callback(status)
                return

            if progress_callback:
                progress_callback(i + 1, total_lines)

        if completion_callback:
            completion_callback(SerialInterface.ReplyStatus.OK)

    def stop_gcode_execution(self):
        self._stop_gcode = True

    def send_gcode_line(self, line: str) -> SerialInterface.ReplyStatus:
        status, _ = self.serial.send_command(line)
        return status

    def emergency_stop(self) -> SerialInterface.ReplyStatus:
        self.stop_gcode_execution()
        status, _ = self.serial.send_command("M410")
        return status

    def trigger_recording(self) -> SerialInterface.ReplyStatus:
        gcode = """
G4 P250
M42 P67 S255
G4 P10
M42 P67 S0
G4 S2
        """.strip()

        for line in gcode.split("\n"):
            status = self.send_gcode_line(line.strip())
            if status == SerialInterface.ReplyStatus.ERROR:
                return status

        return SerialInterface.ReplyStatus.OK

    def reset_to_defaults(self) -> SerialInterface.ReplyStatus:
        status, _ = self.serial.send_command("M502")
        return status
