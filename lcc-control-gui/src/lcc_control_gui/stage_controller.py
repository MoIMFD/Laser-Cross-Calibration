from __future__ import annotations

import re
from pathlib import Path

from lcc_control_gui.serial_interface import SerialInterface


class StageController:
    def __init__(self, serial_interface: SerialInterface):
        self.serial = serial_interface

    def home(self, axes: list[str] | None = None) -> SerialInterface.ReplyStatus:
        if axes is None:
            axes = ["X", "Y", "Z"]

        cmd = "G28"
        for axis in axes:
            if axis.upper() in ["X", "Y", "Z"]:
                cmd += f" {axis.upper()}"

        status, _ = self.serial.send_command(cmd, timeout=30)
        return status

    def move_relative(
        self, axis: str, distance: float, feedrate: float
    ) -> SerialInterface.ReplyStatus:
        axis = axis.upper()
        if axis not in ["X", "Y", "Z"]:
            return SerialInterface.ReplyStatus.ERROR

        self.serial.send_command("G91")
        cmd = f"G0 {axis}{distance:.3f} F{feedrate:.3f}"
        status, _ = self.serial.send_command(cmd)
        self.serial.send_command("G90")

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
            r"X([-+]?\d*\.?\d+)\s*Y([-+]?\d*\.?\d+)\s*Z([-+]?\d*\.?\d+)", response
        )
        if not match:
            return None

        x, y, z = match.groups()
        return float(x), float(y), float(z)

    def run_gcode_file(
        self, filepath: str | Path, progress_callback=None
    ) -> SerialInterface.ReplyStatus:
        filepath = Path(filepath)
        if not filepath.exists():
            return SerialInterface.ReplyStatus.ERROR

        with filepath.open() as f:
            lines = f.readlines()

        total_lines = len(lines)
        for i, line in enumerate(lines):
            line = line.strip()

            if not line or line.startswith(";"):
                continue

            status = self.send_gcode_line(line)
            if status == SerialInterface.ReplyStatus.ERROR:
                return status

            if progress_callback:
                progress_callback(i + 1, total_lines)

        return SerialInterface.ReplyStatus.OK

    def send_gcode_line(self, line: str) -> SerialInterface.ReplyStatus:
        status, _ = self.serial.send_command(line)
        return status

    def emergency_stop(self) -> SerialInterface.ReplyStatus:
        status, _ = self.serial.send_command("M112")
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
