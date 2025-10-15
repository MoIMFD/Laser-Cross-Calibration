from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class MovementWidget(QWidget):
    move_requested = Signal(str, float)
    home_requested = Signal()
    home_axes_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.step_size = 1.0
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        movement_group = QGroupBox("Movement Control")
        movement_layout = QGridLayout()

        self.btn_y_plus = QPushButton("Y+")
        self.btn_y_minus = QPushButton("Y-")
        self.btn_x_plus = QPushButton("X+")
        self.btn_x_minus = QPushButton("X-")
        self.btn_z_plus = QPushButton("Z+")
        self.btn_z_minus = QPushButton("Z-")
        self.btn_home = QPushButton("HOME")
        self.btn_home_axes = QPushButton("HOME AXES")

        for btn in [
            self.btn_y_plus,
            self.btn_y_minus,
            self.btn_x_plus,
            self.btn_x_minus,
            self.btn_z_plus,
            self.btn_z_minus,
        ]:
            btn.setMinimumSize(80, 60)

        self.btn_home.setMinimumSize(60, 60)
        self.btn_home_axes.setMinimumSize(80, 60)

        movement_layout.addWidget(self.btn_y_plus, 0, 1)
        movement_layout.addWidget(self.btn_x_minus, 1, 0)
        movement_layout.addWidget(self.btn_home, 1, 1)
        movement_layout.addWidget(self.btn_x_plus, 1, 2)
        movement_layout.addWidget(self.btn_y_minus, 2, 1)
        movement_layout.addWidget(self.btn_z_plus, 0, 3)
        movement_layout.addWidget(self.btn_z_minus, 2, 3)
        movement_layout.addWidget(self.btn_home_axes, 1, 3)

        movement_group.setLayout(movement_layout)
        layout.addWidget(movement_group)

        step_group = QGroupBox("Step Size (mm)")
        step_layout = QHBoxLayout()

        self.step_buttons = []
        for size in [100, 10, 1, 0.1]:
            btn = QPushButton(str(size))
            btn.setCheckable(True)
            btn.setMinimumSize(80, 40)
            btn.clicked.connect(lambda checked, s=size: self._set_step_size(s))
            step_layout.addWidget(btn)
            self.step_buttons.append(btn)

        self.step_buttons[2].setChecked(True)

        step_group.setLayout(step_layout)
        layout.addWidget(step_group)

        self.btn_x_plus.clicked.connect(lambda: self._move("X", self.step_size))
        self.btn_x_minus.clicked.connect(lambda: self._move("X", -self.step_size))
        self.btn_y_plus.clicked.connect(lambda: self._move("Y", self.step_size))
        self.btn_y_minus.clicked.connect(lambda: self._move("Y", -self.step_size))
        self.btn_z_plus.clicked.connect(lambda: self._move("Z", self.step_size))
        self.btn_z_minus.clicked.connect(lambda: self._move("Z", -self.step_size))
        self.btn_home.clicked.connect(self.home_requested.emit)
        self.btn_home_axes.clicked.connect(self.home_axes_requested.emit)

    def _set_step_size(self, size: float):
        self.step_size = size
        for btn in self.step_buttons:
            btn.setChecked(False)
        for btn in self.step_buttons:
            if float(btn.text()) == size:
                btn.setChecked(True)
                break

    def _move(self, axis: str, distance: float):
        self.move_requested.emit(axis, distance)
