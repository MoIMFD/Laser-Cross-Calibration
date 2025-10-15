from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from .serial_interface import SerialInterface
from .stage_controller import StageController
from .widgets import GCodeWidget, LogWidget, MovementWidget, StatusWidget

if TYPE_CHECKING:
    from pathlib import Path


class ConnectionWorker(QObject):
    finished = Signal(bool, object, object)
    error = Signal(str)

    def __init__(
        self,
        port: str,
        baudrate: int,
        command_callback=None,
        log_callback=None,
        unsolicited_callback=None,
    ):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.command_callback = command_callback
        self.log_callback = log_callback
        self.unsolicited_callback = unsolicited_callback

    def run(self):
        try:
            serial_interface = SerialInterface(
                self.port,
                baud_rate=self.baudrate,
                command_msg_callback=self.command_callback,
                log_msg_callback=self.log_callback,
                unsolicited_msg_callback=self.unsolicited_callback,
            )
            stage_controller = StageController(serial_interface)
            self.finished.emit(True, serial_interface, stage_controller)
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(False, None, None)


class MainWindow(QMainWindow):
    log_command_signal = Signal(str, object, str)
    log_message_signal = Signal(object, str)
    log_unsolicited_signal = Signal(str)
    position_update_signal = Signal(float, float, float)

    def __init__(self):
        super().__init__()
        self.serial_interface: SerialInterface | None = None
        self.stage_controller: StageController | None = None
        self.position_timer = None

        self.setWindowTitle("LCC Control GUI")
        self.setMinimumSize(600, 700)

        self._init_statusbar()
        self._init_ui()

    def _init_statusbar(self):
        statusbar = self.statusBar()

        self.status_connection_label = QLabel("Disconnected")
        self.status_connection_label.setStyleSheet("color: red; padding: 0 10px;")

        self.status_position_label = QLabel("Position: (0.0, 0.0, 0.0)")
        self.status_position_label.setStyleSheet("padding: 0 10px;")

        statusbar.addPermanentWidget(self.status_connection_label)
        statusbar.addPermanentWidget(self.status_position_label)

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.connection_widget = StatusWidget()
        self.movement_widget = MovementWidget()
        self.gcode_widget = GCodeWidget()
        self.log_widget = LogWidget()

        layout.addWidget(self.connection_widget)
        layout.addWidget(self.movement_widget)
        layout.addWidget(self.gcode_widget)
        layout.addWidget(self.log_widget)
        layout.addStretch()

        self.connection_widget.connect_requested.connect(self._connect_serial)
        self.connection_widget.disconnect_requested.connect(self._disconnect_serial)
        self.movement_widget.move_requested.connect(self._handle_move)
        self.movement_widget.home_requested.connect(self._handle_home)
        self.movement_widget.home_axes_requested.connect(self._handle_home_axes)
        self.gcode_widget.run_gcode_requested.connect(self._handle_run_gcode)
        self.gcode_widget.acceleration_changed.connect(self._handle_acceleration_change)
        self.gcode_widget.emergency_stop_requested.connect(self._handle_emergency_stop)

        self.log_command_signal.connect(self.log_widget.log_command)
        self.log_message_signal.connect(self._log_message_slot)
        self.log_unsolicited_signal.connect(self.log_widget.log_unsolicited)
        self.position_update_signal.connect(self._update_position_display)

    def _connect_serial(self, port: str, baudrate: int):
        self.connection_worker = ConnectionWorker(
            port,
            baudrate,
            command_callback=self._log_command,
            log_callback=self._log_message,
            unsolicited_callback=self._log_unsolicited,
        )
        self.connection_thread = threading.Thread(
            target=self.connection_worker.run, daemon=True
        )
        self.connection_worker.finished.connect(self._on_connection_finished)
        self.connection_worker.error.connect(self._on_connection_error)
        self.connection_thread.start()

    def _on_connection_finished(self, success: bool, serial_if, stage_ctrl):
        if success:
            self.serial_interface = serial_if
            self.stage_controller = stage_ctrl
            self.connection_widget.set_connection_status(True)
            self.gcode_widget.set_enabled(True)
            self.status_connection_label.setText("Connected")
            self.status_connection_label.setStyleSheet("color: green; padding: 0 10px;")

            self.position_timer = QTimer()
            self.position_timer.timeout.connect(self._update_position)
            self.position_timer.start(1000)
        else:
            self.connection_widget.set_connection_status(False)
            self.gcode_widget.set_enabled(False)
            self.status_connection_label.setText("Disconnected")
            self.status_connection_label.setStyleSheet("color: red; padding: 0 10px;")

    def _on_connection_error(self, error_msg: str):
        print(f"Failed to connect: {error_msg}")

    def _disconnect_serial(self):
        if self.position_timer:
            self.position_timer.stop()
            self.position_timer = None
        if self.serial_interface:
            self.serial_interface.close()
            self.serial_interface = None
        if self.stage_controller:
            self.stage_controller = None

        self.connection_widget.set_connection_status(False)
        self.gcode_widget.set_enabled(False)
        self.status_connection_label.setText("Disconnected")
        self.status_connection_label.setStyleSheet("color: red; padding: 0 10px;")
        self.status_position_label.setText("Position: (0.0, 0.0, 0.0)")

    def _log_command(self, command: str, status, response: str):
        self.log_command_signal.emit(command, status, response)

    def _log_message(self, level, message: str):
        self.log_message_signal.emit(level, message)

    def _log_message_slot(self, level, message: str):
        level_str = level.value if hasattr(level, "value") else str(level)
        self.log_widget.log_info(message, level_str)

    def _log_unsolicited(self, message: str):
        self.log_unsolicited_signal.emit(message)

    def _update_position(self):
        if self.stage_controller:
            threading.Thread(target=self._read_position_thread, daemon=True).start()

    def _read_position_thread(self):
        position = self.stage_controller.read_position()
        if position:
            self.position_update_signal.emit(*position)

    def _update_position_display(self, x: float, y: float, z: float):
        self.status_position_label.setText(f"Position: ({x:.1f}, {y:.1f}, {z:.1f})")

    def _handle_move(self, axis: str, distance: float):
        if self.stage_controller:
            feedrate = 1000.0
            threading.Thread(
                target=lambda: self.stage_controller.move_relative(
                    axis, distance, feedrate
                ),
                daemon=True,
            ).start()

    def _handle_home(self):
        if self.stage_controller:
            threading.Thread(
                target=self.stage_controller.go_to_home, daemon=True
            ).start()

    def _handle_home_axes(self):
        if self.stage_controller:
            threading.Thread(
                target=self.stage_controller.home_axes, daemon=True
            ).start()

    def _handle_run_gcode(self, file_path: Path):
        if self.stage_controller:
            self.stage_controller.run_gcode_file(
                file_path,
                progress_callback=self._on_gcode_progress,
                completion_callback=self._on_gcode_completed,
            )

    def _on_gcode_progress(self, current: int, total: int):
        self.gcode_widget.gcode_progress.emit(current, total)

    def _on_gcode_completed(self, status: int):
        self.gcode_widget.gcode_completed.emit(status)

    def _handle_acceleration_change(self, acceleration: float):
        if self.stage_controller:
            self.stage_controller.set_acceleration(acceleration)

    def _handle_emergency_stop(self):
        if self.stage_controller:
            self.stage_controller.emergency_stop()

    def closeEvent(self, event):
        if self.position_timer:
            self.position_timer.stop()
        if self.serial_interface:
            self.serial_interface.close()
        super().closeEvent(event)
