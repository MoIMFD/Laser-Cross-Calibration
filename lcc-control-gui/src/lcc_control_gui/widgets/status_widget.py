from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

if TYPE_CHECKING:
    from lcc_control_gui.serial_interface import PortInfo


class StatusWidget(QWidget):
    connect_requested = Signal(str, int)
    disconnect_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._connected = False
        self._init_ui()
        self._cached_ports: list[PortInfo] = []
        self._refresh_ports()

    def _init_ui(self):
        connection_group = QGroupBox("Serial Connection")
        connection_layout = QHBoxLayout()

        connection_label = QLabel("Port:")
        self.port_combo_box = QComboBox()
        self.btn_rescan_ports = QPushButton("Scan")
        self.btn_rescan_ports.clicked.connect(self._on_button_rescan_clicked)

        baudrate_label = QLabel("Baudrate:")
        self.baudrate_combo = QComboBox()
        self.baudrate_combo.addItems(
            ["9600", "19200", "38400", "57600", "115200", "250000"]
        )
        self.baudrate_combo.setCurrentText("115200")

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self._on_button_connect_clicked)

        connection_layout.addWidget(connection_label)
        connection_layout.addWidget(self.port_combo_box)
        connection_layout.addWidget(self.btn_rescan_ports)
        connection_layout.addWidget(baudrate_label)
        connection_layout.addWidget(self.baudrate_combo)
        connection_layout.addWidget(self.btn_connect)

        connection_group.setLayout(connection_layout)

        layout = QHBoxLayout(self)
        layout.addWidget(connection_group)
        layout.setContentsMargins(0, 0, 0, 0)

    def _on_button_connect_clicked(self):
        if self._connected:
            self.disconnect_requested.emit()
        else:
            port = self.port_combo_box.currentData()
            baudrate = int(self.baudrate_combo.currentText())
            if port:
                self.connect_requested.emit(port, baudrate)

    def _on_button_rescan_clicked(self):
        print("[GUI] Refreshing ports")
        self._refresh_ports()

    def set_connection_status(self, connected: bool):
        self._connected = connected
        if connected:
            self.btn_connect.setText("Disconnect")
            self.port_combo_box.setEnabled(False)
            self.baudrate_combo.setEnabled(False)
        else:
            self.btn_connect.setText("Connect")
            self.port_combo_box.setEnabled(True)
            self.baudrate_combo.setEnabled(True)

    def _refresh_ports(self):
        from lcc_control_gui.serial_interface import scan_ports

        self._cached_ports = scan_ports()
        self._update_port_combo()

    def _update_port_combo(self):
        self.port_combo_box.clear()
        if not self._cached_ports:
            self.port_combo_box.setEnabled(False)
            self.port_combo_box.addItem("No ports found")
            self.btn_connect.setEnabled(False)

        for port in self._cached_ports:
            display_text = (
                f"{port.device} - {port.description}"
                if port.description
                else port.device
            )
            self.port_combo_box.addItem(display_text, userData=port.device)
        else:
            self.port_combo_box.setEnabled(True)
            self.btn_connect.setEnabled(True)
