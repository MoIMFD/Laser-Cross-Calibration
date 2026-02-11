"""Pipe center finding using chord geometry from laser cross measurements.

This module provides tools to find pipe centers by analyzing chord lengths
and midpoints from laser cross intersections at different z positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
import uncertainties.unumpy
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit
from uncertainties import correlated_values, ufloat
from uncertainties import unumpy as unp

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from numpy.typing import NDArray
    from uncertainties import UFloat as UFloatType

UFloat = Union[float, "UFloatType"]


@dataclass
class MidpointLine:
    """Line fit through chord midpoints.

    Attributes:
        slope: [slope_x, slope_y] - may contain ufloats for uncertainty.
        intercept: [intercept_x, intercept_y] - may contain ufloats for uncertainty.
    """

    slope: NDArray
    intercept: NDArray

    def at_z(self, z: UFloat) -> NDArray:
        """Evaluate (x, y) at given z, propagating all uncertainties.

        Args:
            z: The z coordinate at which to evaluate the line.

        Returns:
            Array [x, y] at the given z position.
        """
        x = self.slope[0] * z + self.intercept[0]
        y = self.slope[1] * z + self.intercept[1]
        return np.array([x, y])


@dataclass
class PipeGeometry:
    """Full geometric solution for pipe geometry.

    Attributes:
        center_xy: (x, y) center position in image coordinates, may contain ufloats.
        z0: z position of the pipe center.
        a: Semi-axis corresponding to max chord / 2.
        b: Semi-axis along z direction.
        alpha: Perspective coefficient for foreshortening correction.
    """

    center_xy: NDArray
    z0: UFloat
    a: UFloat
    b: UFloat
    alpha: UFloat

    def nominal(self) -> PipeGeometry:
        """Return a copy with all nominal values (no uncertainties)."""

        def nom(x: UFloat) -> float:
            if hasattr(x, "nominal_value"):
                return x.nominal_value
            return x

        return PipeGeometry(
            center_xy=np.array([nom(self.center_xy[0]), nom(self.center_xy[1])]),
            z0=nom(self.z0),
            a=nom(self.a),
            b=nom(self.b),
            alpha=nom(self.alpha),
        )


class PipeCenterFinder:
    """Find pipe center using chord geometry.

    This class analyzes laser cross chord measurements to determine pipe
    center position and geometry. The key insight is that L^2 (chord length
    squared) is parabolic in z, allowing direct estimation of z0 from the
    vertex without degeneracy.

    Attributes:
        points: Input points array of shape (N, 3) with columns [x, y, z].
        geometry: Fitted PipeGeometry after calling fit().
        midpoint_line: Fitted MidpointLine after calling fit().

    Example:
        >>> points = np.array([...])  # chord endpoints with z values
        >>> finder = PipeCenterFinder(points)
        >>> geometry = finder.fit()
        >>> print(finder.summary())
    """

    def __init__(self, points: NDArray, uncertainty: float = 0.0) -> None:
        """Initialize the pipe center finder.

        Args:
            points: Array of shape (N, 3) containing chord endpoint coordinates.
                Each row is [x, y, z]. Points at the same z form chord pairs.
        """
        self.points = uncertainties.unumpy.uarray(
            points, np.ones_like(points) * uncertainty
        )
        self.geometry: PipeGeometry | None = None
        self.midpoint_line: MidpointLine | None = None

    def _chord_data(self) -> tuple[NDArray, NDArray, NDArray]:
        """Extract z values, chord lengths, and midpoints from points.

        Returns:
            Tuple of (z_values, lengths, midpoints) where:
                - z_values: unique z coordinates
                - lengths: chord length at each z
                - midpoints: (x, y) midpoint at each z
        """
        z_values = np.unique(uncertainties.unumpy.nominal_values(self.points)[:, 2])
        lengths = []
        midpoints = []

        for z in z_values:
            pts = self.points[
                uncertainties.unumpy.nominal_values(self.points)[:, 2] == z
            ][:, :2]
            distance = pts[1] - pts[0]
            lengths.append((distance[0] ** 2 + distance[1] ** 2) ** 0.5)
            midpoints.append(pts.mean(axis=0))

        return z_values, np.array(lengths), np.array(midpoints)

    def _fit_z0_from_parabola(
        self, z: NDArray, L: NDArray
    ) -> tuple[UFloat, UFloat, UFloat]:
        """Fit parabola to L^2 to find z0 directly.

        The model is:
            L^2 = 4a^2 - (4a^2/b^2)(z - z0)^2
                = c0 + c1*z + c2*z^2

        The vertex occurs at z0 = -c1 / (2*c2).

        Args:
            z: Array of z coordinates.
            L: Array of chord lengths.

        Returns:
            Tuple of (z0, a, b) with uncertainties.
        """

        def polynom(z, c0, c1, c2):
            return c0 + c1 * z + c2 * z**2

        ydata = uncertainties.unumpy.nominal_values(L**2)
        yerr = uncertainties.unumpy.std_devs(L**2)

        popt, pcov = curve_fit(
            polynom, xdata=z, ydata=ydata, sigma=yerr, absolute_sigma=True
        )
        c0, c1, c2 = correlated_values(popt, pcov)

        p = Polynomial([c0, c1, c2])
        z0 = p.deriv().roots().item()

        L_sq_max = p(z0)
        a = L_sq_max**0.5 / 2

        # b from curvature: c2 = -4a^2/b^2 -> b = sqrt(-4a^2/c2) = sqrt(-L_sq_max/c2)
        b = (-L_sq_max / c2) ** 0.5 if c2.nominal_value < 0 else ufloat(float("inf"), 0)

        return z0, a, b

    def _fit_alpha(
        self, z: NDArray, L: NDArray, z0: UFloat, a: UFloat, b: UFloat
    ) -> UFloat:
        """Fit perspective coefficient with uncertainty propagation."""

        dz = z - z0  # array of ufloats (z0 uncertainty propagates)

        # Use ufloat-compatible math
        true_L = 2 * a * (1 - (dz / b) ** 2) ** 0.5  # array of ufloats

        # ratio is now array of ufloats with propagated uncertainty from z0, a, b, and L
        ratio = L / true_L - 1

        # Filter valid points
        dz_nom = unp.nominal_values(dz)
        true_L_nom = unp.nominal_values(true_L)
        valid = (true_L_nom > 1e-6) & (np.abs(dz_nom) > 1e-6)

        if valid.sum() < 2:
            return ufloat(0.0, 0.0)

        # Extract for fitting
        ratio_valid = ratio[valid]
        dz_valid = dz[valid]

        ydata = unp.nominal_values(ratio_valid)
        yerr = unp.std_devs(ratio_valid)
        xdata = unp.nominal_values(dz_valid)

        # Linear through origin: ratio = alpha * dz
        def model(x, alpha):
            return alpha * x

        popt, pcov = curve_fit(model, xdata, ydata, sigma=yerr, absolute_sigma=True)
        alpha = ufloat(popt[0], np.sqrt(pcov[0, 0]))

        return alpha

    def _fit_midpoint_line(self, z: NDArray, midpoints: NDArray) -> MidpointLine:
        """Fit line through chord midpoints with uncertainties.

        Args:
            z: Array of z coordinates.
            midpoints: Array of (x, y) midpoints, can be ufloats.

        Returns:
            MidpointLine with slope and intercept including uncertainties.
        """
        # Extract nominal values and uncertainties
        x_nom = unp.nominal_values(midpoints[:, 0])
        y_nom = unp.nominal_values(midpoints[:, 1])
        x_err = unp.std_devs(midpoints[:, 0])
        y_err = unp.std_devs(midpoints[:, 1])

        # Handle zero uncertainties (replace with small value or use unweighted)
        x_err = np.where(x_err > 0, x_err, 1e-10)
        y_err = np.where(y_err > 0, y_err, 1e-10)

        def line(z, slope, intercept):
            return slope * z + intercept

        # Fit x(z)
        popt_x, pcov_x = curve_fit(line, z, x_nom, sigma=x_err, absolute_sigma=True)
        slope_x, intercept_x = correlated_values(popt_x, pcov_x)

        # Fit y(z)
        popt_y, pcov_y = curve_fit(line, z, y_nom, sigma=y_err, absolute_sigma=True)
        slope_y, intercept_y = correlated_values(popt_y, pcov_y)

        return MidpointLine(
            slope=np.array([slope_x, slope_y]),
            intercept=np.array([intercept_x, intercept_y]),
        )

    def fit(self) -> PipeGeometry:
        """Find pipe center and geometry with uncertainties.

        Returns:
            PipeGeometry containing center position and ellipse parameters.
        """
        z, L, midpoints = self._chord_data()

        # 1. Fit z0, a, b from parabolic chord model
        z0, a, b = self._fit_z0_from_parabola(z, L)

        # 2. Fit perspective coefficient
        alpha = self._fit_alpha(z, L, z0, a, b)

        # 3. Fit midpoint line and evaluate at z0
        self.midpoint_line = self._fit_midpoint_line(z, midpoints)
        center_xy = self.midpoint_line.at_z(z0)

        self.geometry = PipeGeometry(
            center_xy=center_xy,
            z0=z0,
            a=a,
            b=b,
            alpha=alpha,
        )

        return self.geometry

    def summary(self) -> str:
        """Generate a human-readable summary of the fitted geometry.

        Returns:
            Formatted string with center position and ellipse parameters.
        """
        if self.geometry is None:
            return "Not fitted yet."

        g = self.geometry

        def fmt(val: UFloat, decimals: int = 2) -> str:
            if hasattr(val, "nominal_value"):
                return (
                    f"{val.nominal_value:.{decimals}f} +/- {val.std_dev:.{decimals}f}"
                )
            return f"{val:.{decimals}f}"

        return (
            f"Pipe Geometry:\n"
            f"  center (x, y): ({fmt(g.center_xy[0], 1)}, {fmt(g.center_xy[1], 1)})\n"
            f"  center z0:     {fmt(g.z0, 3)} mm\n"
            f"  semi-axis a:   {fmt(g.a, 1)} px\n"
            f"  semi-axis b:   {fmt(g.b, 2)} mm\n"
            f"  alpha:         {fmt(g.alpha, 6)} /mm"
        )

    def plot(
        self, axes: tuple[plt.Axes, plt.Axes] | None = None
    ) -> tuple[plt.Axes, plt.Axes]:
        """Visualize fit with uncertainty bands.

        Args:
            axes: Optional tuple of (ax1, ax2) matplotlib axes. If None,
                creates a new figure.

        Returns:
            Tuple of (ax1, ax2) axes used for plotting.

        Raises:
            ValueError: If fit() has not been called yet.
        """
        if self.geometry is None:
            raise ValueError("Call fit() first")

        import matplotlib.pyplot as plt

        if axes is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        else:
            ax1, ax2 = axes

        z, L, midpoints = self._chord_data()
        g = self.geometry.nominal()

        # Left: L^2 vs z with parabola fit
        z_fit = np.linspace(
            z.min() - 0.1 * (z.max() - z.min()),
            z.max() + 0.1 * (z.max() - z.min()),
            100,
        )
        L_fit = 2 * g.a * (1 - ((z_fit - g.z0) / g.b) ** 2) ** 0.5

        ax1.scatter(
            z,
            uncertainties.unumpy.nominal_values(L**2),
            s=80,
            c="blue",
            label="Observed L^2",
        )
        ax1.plot(z_fit, L_fit**2, "g-", linewidth=2, label="Fitted parabola")
        ax1.axvline(g.z0, color="red", linestyle="--", label=f"z0 = {self.geometry.z0}")
        ax1.set_xlabel("z (mm)")
        ax1.set_ylabel("L^2 (px^2)")
        ax1.legend()
        ax1.set_title("Chord Length^2 vs z")

        # Right: image space
        ax2.scatter(
            uncertainties.unumpy.nominal_values(self.points[:, 0]),
            uncertainties.unumpy.nominal_values(self.points[:, 1]),
            c=uncertainties.unumpy.nominal_values(self.points[:, 2]),
            cmap="coolwarm",
            s=80,
            label="Endpoints",
        )
        ax2.scatter(
            uncertainties.unumpy.nominal_values(midpoints[:, 0]),
            uncertainties.unumpy.nominal_values(midpoints[:, 1]),
            c=z,
            cmap="coolwarm",
            s=60,
            marker="s",
            edgecolors="black",
            label="Midpoints",
        )

        # Midpoint line
        z_range = np.array([z.min(), z.max()])
        line_pts = np.array([self.midpoint_line.at_z(zi) for zi in z_range])
        ax2.plot(
            unp.nominal_values(line_pts[:, 0]),
            unp.nominal_values(line_pts[:, 1]),
            "b--",
            label="Midpoint line",
        )

        # Center with marker
        ax2.plot(
            g.center_xy[0],
            g.center_xy[1],
            "r+",
            markersize=20,
            markeredgewidth=3,
            label=f"Center (z0={self.geometry.z0})",
        )

        ax2.set_aspect("equal")
        ax2.invert_yaxis()
        ax2.legend()
        ax2.set_title("Image Space")

        plt.tight_layout()
        return ax1, ax2

    def plot_ellipse(self, ax: plt.Axes | None = None, sigma: float = 3.0) -> plt.Axes:
        """Plot the fitted ellipse in the u-z plane with uncertainty bands.

        Shows the ellipse cross-section with the center position and
        uncertainty bands for both the ellipse shape and center z0.

        Args:
            ax: Optional matplotlib axes. If None, creates a new figure.
            sigma: Number of standard deviations for uncertainty bands.

        Returns:
            The axes used for plotting.

        Raises:
            ValueError: If fit() has not been called yet.
        """
        if self.geometry is None:
            raise ValueError("Call fit() first")

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        g = self.geometry

        # Parametric ellipse
        theta = np.linspace(0, 2 * np.pi, 100)
        u = g.a * np.cos(theta)
        z_ellipse = g.z0 + g.b * np.sin(theta)

        ellipse_nom = unp.nominal_values(z_ellipse)
        ellipse_std = unp.std_devs(z_ellipse)

        ax.plot(
            unp.nominal_values(u),
            ellipse_nom,
            "g-",
            linewidth=2,
            label="Fitted ellipse",
        )
        ax.fill_between(
            unp.nominal_values(u),
            ellipse_nom - sigma * ellipse_std,
            ellipse_nom + sigma * ellipse_std,
            color="green",
            alpha=0.2,
        )

        # Center z0 with uncertainty band
        z0_nom = g.z0.nominal_value
        z0_std = g.z0.std_dev
        ax.axhline(z0_nom, color="red", linestyle="--", alpha=0.5)
        ax.axhspan(
            z0_nom - sigma * z0_std,
            z0_nom + sigma * z0_std,
            color="red",
            alpha=0.3,
        )
        ax.plot(
            0,
            z0_nom,
            "r+",
            markersize=15,
            markeredgewidth=2,
            label=f"Center z0 = {g.z0}",
        )

        ax.set_xlabel("u (px)")
        ax.set_ylabel("z (mm)")
        ax.legend()
        ax.set_title("Ellipse in u-z Plane")

        return ax
