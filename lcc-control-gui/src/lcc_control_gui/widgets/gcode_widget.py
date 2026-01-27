from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class GCodeWidget(QWidget):
    run_gcode_requested = Signal(Path)
    acceleration_changed = Signal(float)
    emergency_stop_requested = Signal()
    gcode_progress = Signal(int, int)
    gcode_completed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self.set_enabled(False)

    def _init_ui(self):
        layout = QVBoxLayout(self)

        accel_group = QGroupBox("Acceleration")
        accel_layout = QHBoxLayout()

        accel_label = QLabel("Acceleration:")
        self.accel_spinbox = QDoubleSpinBox()
        self.accel_spinbox.setRange(0.01, 1000.0)
        self.accel_spinbox.setValue(0.1)
        self.accel_spinbox.setSingleStep(0.1)
        self.accel_spinbox.setDecimals(2)
        self.accel_spinbox.valueChanged.connect(self.acceleration_changed.emit)

        accel_layout.addWidget(accel_label)
        accel_layout.addWidget(self.accel_spinbox)
        accel_layout.addStretch()

        self.btn_emergency_stop = QPushButton("Emergency Stop")
        self.btn_emergency_stop.setStyleSheet(
            "QPushButton { "
            "background-color: #ff4444; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #cc0000; }"
        )
        self.btn_emergency_stop.setMinimumHeight(40)
        self.btn_emergency_stop.clicked.connect(self.emergency_stop_requested.emit)
        accel_layout.addWidget(self.btn_emergency_stop)

        accel_group.setLayout(accel_layout)
        layout.addWidget(accel_group)

        file_group = QGroupBox("GCode File")
        file_layout = QVBoxLayout()

        file_select_layout = QHBoxLayout()
        file_label = QLabel("File:")
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Path/to/file.gcode")
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.clicked.connect(self._browse_file)

        file_select_layout.addWidget(file_label)
        file_select_layout.addWidget(self.file_path_edit)
        file_select_layout.addWidget(self.btn_browse)

        self.btn_run = QPushButton("Run GCode")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.clicked.connect(self._run_gcode)

        file_layout.addLayout(file_select_layout)
        file_layout.addWidget(self.btn_run)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

    def _browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select GCode File", "", "GCode Files (*.gcode *.nc);;All Files (*)"
        )
        if file_path:
            self.file_path_edit.setText(file_path)

    def _run_gcode(self):
        file_path = self.file_path_edit.text()
        if file_path:
            self.run_gcode_requested.emit(Path(file_path))

    def set_enabled(self, enabled: bool):
        self.btn_run.setEnabled(enabled)
        self.btn_emergency_stop.setEnabled(enabled)
