"""Optimization module for finding stage positions that align laser intersections with
targets.

This module provides tools for solving the inverse kinematics problem of a dual-laser
stage system: given a desired intersection point, find the stage origin that achieves
it.

The approach combines machine learning (for initial guess estimation) with numerical
optimization (for precise refinement).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

if TYPE_CHECKING:
    from collections.abc import Callable

    from hazy import Frame, Point
    from numpy.typing import NDArray

    from laser_cross_calibration.sources import DualLaserStageSource
    from laser_cross_calibration.tracing import RayTracer


class Estimator(Protocol):
    """Protocol for initial guess estimators.

    Estimators provide a fast approximation of the stage origin needed to hit
    a target point. This initial guess is then refined by numerical optimization.
    """

    def __call__(self, point: Point) -> Point:
        """Estimate the stage origin needed to hit the given target point.

        Args:
            point: The target intersection point.

        Returns:
            Estimated stage origin position.
        """
        ...


class GradientBoostingEstimator(Estimator):
    """ML-based estimator using gradient boosting regression.

    Uses a trained GradientBoostingRegressor to predict stage origins from target
    points. Should be trained on calibration data mapping target positions to
    known stage origins.

    Attributes:
        model: The underlying multi-output gradient boosting model.
        frame: The coordinate frame for predictions.
    """

    def __init__(
        self,
        frame: Frame,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
    ):
        """Initialize the gradient boosting estimator.

        Args:
            frame: Coordinate frame for input/output points.
            n_estimators: Number of boosting stages.
            learning_rate: Shrinks the contribution of each tree.
            max_depth: Maximum depth of individual trees.
        """
        self.frame = frame
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self.model = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
            )
        )

    def fit[T: np.floating](self, X: NDArray[T], y: NDArray[T]) -> Self:
        """Train the estimator on calibration data.

        Args:
            X: Target points (N x 3 array).
            y: Corresponding stage origins (N x 3 array).

        Returns:
            Self for method chaining.
        """
        self.model.fit(X=X, y=y)
        return self

    def predict(self, point: Point) -> Point:
        """Predict the stage origin for a target point.

        Args:
            point: Target intersection point.

        Returns:
            Predicted stage origin in the original point's frame.
        """
        point_frame = point.frame
        point_local = np.array(point.to_frame(self.frame))
        prediction = self.model.predict([point_local])
        return self.frame.point(prediction).to_frame(point_frame)

    def __call__(self, point: Point) -> Point:
        """Estimate stage origin for a target point.

        Args:
            point: Target intersection point.

        Returns:
            Predicted stage origin.
        """
        return self.predict(point=point)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"frame={self.frame!r}, "
            f"n_estimators={self._n_estimators}, "
            f"learning_rate={self._learning_rate}, "
            f"max_depth={self._max_depth})"
        )


class Optimizer:
    """Finds stage origins that align laser intersections with target points.

    Combines an initial guess estimator with Nelder-Mead optimization to solve
    the inverse problem: given a desired intersection point, find the stage
    origin that achieves it.

    The optimization is configured for noisy physical systems (stepper-driven
    stages) with tolerances in the micrometer range.

    Attributes:
        estimator: Provides initial guesses for optimization.
        tracer: Ray tracer for computing laser intersections.
        loss_function: Function to compute error between points.
    """

    # Optimization tolerances (in meters)
    POSITION_TOLERANCE = 1e-6  # 1 µm - stage position convergence
    ERROR_TOLERANCE = 1e-7  # 0.1 µm - intersection error convergence

    # Penalty for invalid configurations (no intersection found)
    INVALID_PENALTY = 1e12

    def __init__(
        self,
        tracer: RayTracer,
        estimator: Estimator,
        loss_function: Callable[[Point, Point], float] | None = None,
    ):
        """Initialize the optimizer.

        Args:
            tracer: Ray tracer for computing laser intersections.
            estimator: Provides initial guess for stage origin.
            loss_function: Optional custom loss function. Defaults to Euclidean
                distance.
        """
        self.estimator = estimator
        self.tracer = tracer
        self.loss_function = loss_function or self.default_loss_function

    def find_source_origin(
        self, target: Point, source: DualLaserStageSource
    ) -> OptimizeResult:
        """Find the stage origin that places the laser intersection at the target.

        Uses Nelder-Mead optimization (gradient-free) which is robust to the
        measurement noise inherent in physical stage systems.

        Args:
            target: Desired intersection point.
            source: The dual-laser stage source to optimize.

        Returns:
            Optimization result containing:
                - x: Optimal stage origin coordinates
                - fun: Final intersection error (meters)
                - success: Whether convergence criteria were met
        """
        original_origin = source.origin.copy()
        target_frame = target.frame
        target = target.to_frame(source.origin.frame)

        def objective(new_origin: NDArray) -> float:
            """Compute intersection error for a given stage origin."""
            source.set_origin(source.origin.frame.point(new_origin))
            _, intersections = self.tracer.trace_and_find_crossings(
                sources=[source], threshold=1e-6
            )
            source.set_origin(original_origin)

            if len(intersections) == 1:
                return self.loss_function(intersections[0], target)
            return self.INVALID_PENALTY

        initial_guess = np.array(self.estimator(point=target))

        result = minimize(
            fun=objective,
            x0=initial_guess,
            method="Nelder-Mead",
            options={
                "xatol": self.POSITION_TOLERANCE,
                "fatol": self.ERROR_TOLERANCE,
            },
        )
        result.x = target_frame.point(result.x)
        return result

    @staticmethod
    def default_loss_function(prediction: Point, target: Point) -> float:
        """Compute Euclidean distance between prediction and target.

        Args:
            prediction: Predicted point coordinates.
            target: Target point coordinates.

        Returns:
            Euclidean distance between the points.
        """
        return (prediction - target).magnitude

    def __repr__(self) -> str:
        loss_name = (
            getattr(self.loss_function, "__name__", None)
            or getattr(self.loss_function, "__qualname__", None)
            or type(self.loss_function).__name__
        )
        return (
            f"{self.__class__.__qualname__}("
            f"tracer={self.tracer!r}, "
            f"estimator={self.estimator!r}, "
            f"loss_function={loss_name})"
        )
