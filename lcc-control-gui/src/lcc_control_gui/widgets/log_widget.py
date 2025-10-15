from __future__ import annotations

from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


class LogWidget(QGroupBox):
    def __init__(self):
        super().__init__("Serial Log")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMaximumHeight(200)
        self.log_display.setStyleSheet("font-family: monospace;")

        button_layout = QHBoxLayout()
        self.clear_button = QPushButton("Clear Log")
        self.clear_button.clicked.connect(self.log_display.clear)
        button_layout.addStretch()
        button_layout.addWidget(self.clear_button)

        layout.addWidget(self.log_display)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def log_command(self, command: str, response_status, response: str):
        cursor = self.log_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#2196F3"))
        cursor.setCharFormat(fmt)
        cursor.insertText(f"TX: {command.strip()}\n")

        if response_status:
            status_color = {
                "ok": "#4CAF50",
                "error": "#F44336",
                "timeout": "#FF9800",
                "busy": "#FFC107",
            }
            color = status_color.get(response_status.value, "#FFFFFF")

            fmt.setForeground(QColor(color))
            cursor.setCharFormat(fmt)
            cursor.insertText(f"RX: {response_status.value}")

            if response:
                fmt.setForeground(QColor("#9E9E9E"))
                cursor.setCharFormat(fmt)
                cursor.insertText(f" - {response.strip()}")

            cursor.insertText("\n")

        self.log_display.setTextCursor(cursor)
        self.log_display.ensureCursorVisible()

    def log_unsolicited(self, message: str):
        cursor = self.log_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#9C27B0"))
        cursor.setCharFormat(fmt)
        cursor.insertText(f"<< {message}\n")

        self.log_display.setTextCursor(cursor)
        self.log_display.ensureCursorVisible()

    def log_info(self, message: str, level: str = "info"):
        cursor = self.log_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        color_map = {
            "debug": "#607D8B",
            "info": "#00BCD4",
            "warning": "#FF9800",
            "error": "#F44336",
        }

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color_map.get(level, "#FFFFFF")))
        cursor.setCharFormat(fmt)
        cursor.insertText(f"[{level.upper()}] {message}\n")

        self.log_display.setTextCursor(cursor)
        self.log_display.ensureCursorVisible()
