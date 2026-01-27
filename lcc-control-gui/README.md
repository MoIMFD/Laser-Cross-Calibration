# Laser Cross Calibration - Control GUI

## Description

PySide6 GUI application for controlling a Marlin-based laser cross calibration stage via serial communication and G-code commands.

**Architecture:**
```
GUI → StageController → SerialInterface → Serial Port (Marlin Firmware)
```

The `StageController` can also be used directly in Python scripts for automated control without the GUI.

### Features

- Manual movement control with configurable step sizes (0.1, 1, 10, 100 mm)
- Acceleration configuration
- Emergency stop (M410 G-code)
- Automatic serial port scanning
- G-code file execution with progress tracking
- Real-time position monitoring
- Multi-threaded serial communication (non-blocking GUI)

## Installation

**Requirements:**
- Python ≥ 3.12
- PySide6 >= 6.9.2
- pyserial >= 3.5
- colorama >= 0.4.6

**Install from main project:**
```bash
cd ..  # Navigate to main project directory
uv pip install -e ./lcc-control-gui
```

**Or install standalone:**
```bash
cd lcc-control-gui
uv pip install -e .
```

## Usage

### Running the GUI

```bash
# From anywhere after installation
python -m lcc_control_gui

# Or from this directory
python -m lcc_control_gui
```

### First-time Setup

1. Connect the stage controller via USB
2. Launch the GUI
3. Select the serial port from the dropdown (e.g., `/dev/ttyUSB0` on Linux, `COM3` on Windows)
4. Click **Connect**
5. Wait for connection status: ● Connected
6. Click **Home Axes** to initialize the stage position

### Manual Movement

1. Select step size (100, 10, 1, or 0.1 mm) using the radio buttons
2. Click the direction buttons (X+, X-, Y+, Y-, Z+, Z-) to move
3. Current position updates automatically in the status bar
4. Use **Go Home** to return to (0, 0, 0)

### G-code File Execution

1. Click **Browse...** to select a `.gcode` file
2. Click **Run GCode**
3. Monitor the progress bar
4. Click **Emergency Stop** if needed

### Scripting with StageController

The `StageController` can be used directly in Python for automated tasks:

```python
from lcc_control_gui.serial_interface import SerialInterface
from lcc_control_gui.stage_controller import StageController

# Initialize serial connection
serial_if = SerialInterface(port="/dev/ttyUSB0", baud_rate=115200)

# Create stage controller
stage = StageController(serial_if)

# Home all axes
stage.home_axes()

# Move to absolute position
stage.move_absolute(x=10.0, y=20.0, z=5.0, feedrate=1000.0)

# Read current position
position = stage.read_position()
print(f"Current position: {position}")

# Run G-code file
stage.run_gcode_file("path/to/file.gcode")

# Clean up
serial_if.close()
```

## Development

### Running Tests

```bash
pytest tests/
```

### Test Device

A simple test device firmware for ESP32 (also compatible with Arduino) is provided in `firmware/`:

```bash
cd firmware
platformio run --target upload
```

This creates a mock Marlin controller for testing the GUI without physical hardware.

## Technical Notes

- **Serial Baud Rate:** 115200 (standard for Marlin)
- **Position Polling:** 1000ms interval
- **Threading:** Serial communication runs in a background thread to prevent GUI blocking
- **G-code Standard:** Marlin-compatible commands (see [Marlin G-code Documentation](https://marlinfw.org/docs/gcode/G000-G001.html))

## Future Enhancements

- Logging via loguru
- Text field for custom G-code commands
- Save/load movement presets
- Position history and visualization
