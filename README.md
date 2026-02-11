# Laser Cross Calibration

A precision laser cross-calibration system for volumetric PIV (Particle Image Velocimetry) camera calibration using non-sequential ray tracing through complex optical geometries.

## Concept

The laser cross calibration works by moving the intersection point of two laser beams through a volume. By taking pictures and knowing the world-space location of the intersection, this information can be used for camera calibration. This method has the advantage of being non-invasive. For simple configurations, e.g. a beam at an angle through a flat surface, the intersection can be calculated easily. However, more complex systems require the aid of non-sequential ray tracing. The goal is to provide a good estimation of the intersection point in world coordinates to create an initial calibration which can subsequently be refined using _Volume-Self-Calibration_ or similar methods.

## Package Structure

The project is a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) containing:

- **`laser-cross-calibration`** -- the core Python library (`src/laser_cross_calibration/`)
- **`lcc-control-gui`** -- a PySide6 GUI for controlling a Marlin-based laser stage (`lcc-control-gui/`)

### Core Library Modules

| Module | Description |
|---|---|
| `surfaces` | Surface primitives for ray-surface intersection: `Plane`, `InfiniteCylinder`, `FiniteCylinder`, `EllipticCylinder`, and `TriSurface` (STL meshes) |
| `materials` | Optical material definitions with refractive index models: `ConstantMaterial`, `SellmeierGlass`, `WaterGlycerolMixture`. Includes a registry with common pre-defined materials (Air, Water, PMMA, BK7, Fused Silica, ...) |
| `tracing` | Non-sequential ray tracer: `OpticalRay`, `OpticalInterface`, `OpticalSystem`, `RayTracer`. Handles Snell's law refraction at material interfaces and finds beam crossings |
| `sources` | Laser source abstractions: `SingleLaserSource` and `DualLaserStageSource` for modeling physical laser stage setups with two beam arms |
| `optimization` | Inverse kinematics solver: given a desired intersection point, find the stage origin. Uses a `GradientBoostingEstimator` for initial guess + Nelder-Mead refinement |
| `geometry` | `PipeCenterFinder` for determining pipe center position and geometry from chord measurements, with full uncertainty propagation via the `uncertainties` package |
| `visualization` | Interactive 3D scene viewer built on Plotly for visualizing optical systems, traced rays, and intersection points |

### Control GUI

The `lcc-control-gui` sub-package provides a PySide6 desktop application for controlling the physical laser stage via serial/G-code. See its own [README](lcc-control-gui/README.md) for details.

## Installation

Requires Python >= 3.12. Managed with [uv](https://docs.astral.sh/uv/).

```bash
# Install the core library in development mode
uv sync

# Optionally install the control GUI
uv pip install -e ./lcc-control-gui

# Install example notebook dependencies
uv sync --extra examples
```

## Usage

Jupyter notebook tutorials are provided in `examples/`:

- **`introduction-tutorial.ipynb`** -- Getting started with the ray tracing framework
- **`laser-cross-stage.ipynb`** -- Full dual-laser stage setup with optimization

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/
```
