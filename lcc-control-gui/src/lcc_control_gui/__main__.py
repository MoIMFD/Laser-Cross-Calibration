from __future__ import annotations

import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()

    def signal_handler(sig, frame):
        print("\n[LCC Control GUI] Received interrupt signal, shutting down...")
        window.close()
        app.quit()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
