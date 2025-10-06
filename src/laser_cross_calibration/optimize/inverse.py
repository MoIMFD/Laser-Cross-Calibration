"""Inverse optimization for finding stage positions given target intersections."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import warnings

import numpy as np
from scipy.optimize import Bounds, minimize

from laser_cross_calibration.ray_tracing import OpticalSystem
from laser_cross_calibration.types import POINT3


class OptimizationStatus(Enum):
    """Optimization result status codes."""

    SUCCESS = "success"
    MAX_ITERATIONS = "max_iterations"
    TOLERANCE_REACHED = "tolerance_reached"
    FAILED = "failed"
    NO_INTERSECTION = "no_intersection"


@dataclass
class OptimizationOperand:
    """
    Optimization operand defining a target or constraint.

    Inspired by Optiland's operand system for flexible optimization targets.

    Attributes:
        operand_type: Type of operand (e.g., "intersection_target", "ray_intersection")
        target_value: Target value for the operand
        current_value: Current computed value
        weight: Weighting factor for this operand in merit function
        tolerance: Acceptable tolerance for this operand
        bounds: Optional bounds for constraint operands
    """

    operand_type: str
    target_value: Any
    current_value: Any = None
    weight: float = 1.0
    tolerance: float = 1e-6
    bounds: Optional[tuple] = None

    def compute_error(self) -> float:
        """Compute weighted error contribution."""
        if self.current_value is None:
            return float("inf")

        if self.operand_type == "intersection_target":
            # 3D position error
            target = np.array(self.target_value)
            current = np.array(self.current_value)
            error = np.linalg.norm(current - target)
            return self.weight * error

        elif self.operand_type == "intersection_exists":
            # Boolean: intersection should exist
            exists = self.current_value is not None and len(self.current_value) > 0
            error = 0.0 if exists == self.target_value else 1.0
            return self.weight * error

        else:
            # Generic scalar error
            error = abs(self.current_value - self.target_value)
            return self.weight * error


@dataclass
class OptimizationVariable:
    """
    Optimization variable with bounds and scaling.

    Attributes:
        name: Variable identifier
        initial_value: Starting value for optimization
        min_bound: Lower bound
        max_bound: Upper bound
        scale_factor: Scaling factor for numerical conditioning
    """

    name: str
    initial_value: float
    min_bound: float = -np.inf
    max_bound: float = np.inf
    scale_factor: float = 1.0


@dataclass
class OptimizationResult:
    """
    Optimization result container.

    Attributes:
        success: Whether optimization succeeded
        status: Detailed status code
        optimal_position: Optimal stage position found
        final_intersection: Final intersection point achieved
        target_intersection: Original target intersection point
        final_error: Final merit function value
        num_iterations: Number of optimization iterations
        convergence_info: Additional convergence information
    """

    success: bool
    status: OptimizationStatus
    optimal_position: Optional[POINT3] = None
    final_intersection: Optional[POINT3] = None
    target_intersection: Optional[POINT3] = None
    final_error: float = float("inf")
    num_iterations: int = 0
    convergence_info: Dict[str, Any] = field(default_factory=dict)


class StagePositionOptimizer:
    """
    Optimizer for finding stage positions given target intersection points.

    Inspired by Optiland's optimization framework with operands and variables.
    Solves the inverse problem: given desired intersection → find stage position.
    """

    def __init__(self, optical_system: OpticalSystem):
        """
        Initialize optimizer with optical system.

        Args:
            optical_system: Complete optical system for ray tracing
        """
        self.optical_system = optical_system
        self.operands: List[OptimizationOperand] = []
        self.variables: List[OptimizationVariable] = []

        # Optimization settings
        self.max_iterations = 1000
        self.tolerance = 1e-6
        self.method = "L-BFGS-B"  # Good for bounded problems

        # History tracking
        self.iteration_history: List[Dict[str, Any]] = []

    def add_operand(
        self,
        operand_type: str,
        target_value: Any,
        weight: float = 1.0,
        tolerance: float = 1e-6,
    ) -> None:
        """Add optimization operand (target or constraint)."""
        operand = OptimizationOperand(
            operand_type=operand_type,
            target_value=target_value,
            weight=weight,
            tolerance=tolerance,
        )
        self.operands.append(operand)

    def add_variable(
        self,
        name: str,
        initial_value: float,
        min_bound: float = -np.inf,
        max_bound: float = np.inf,
        scale_factor: float = 1.0,
    ) -> None:
        """Add optimization variable with bounds."""
        variable = OptimizationVariable(
            name=name,
            initial_value=initial_value,
            min_bound=min_bound,
            max_bound=max_bound,
            scale_factor=scale_factor,
        )
        self.variables.append(variable)

    def setup_stage_position_variables(
        self, initial_position: POINT3, position_bounds: Optional[tuple] = None
    ) -> None:
        """
        Setup standard 3-DOF stage position variables.

        Args:
            initial_position: Starting stage position [x, y, z]
            position_bounds: (min_pos, max_pos) bounds for each axis
        """
        if position_bounds is None:
            # Use stage limits if available
            if hasattr(self.optical_system.stage, "limits"):
                limits = self.optical_system.stage.limits
                min_pos = limits[0]
                max_pos = limits[1]
                # Convert from pint quantities if needed
                if hasattr(min_pos, "magnitude"):
                    min_pos = np.array(min_pos.magnitude)
                if hasattr(max_pos, "magnitude"):
                    max_pos = np.array(max_pos.magnitude)
            else:
                min_pos = np.array([-100, -100, -100])  # Default bounds
                max_pos = np.array([100, 100, 100])
        else:
            min_pos, max_pos = position_bounds

        # Add variables for each axis
        for i, axis in enumerate(["x", "y", "z"]):
            self.add_variable(
                name=f"stage_{axis}",
                initial_value=initial_position[i],
                min_bound=min_pos[i],
                max_bound=max_pos[i],
                scale_factor=1.0,  # mm units
            )

    def _evaluate_operands(self, stage_position: POINT3) -> None:
        """Evaluate all operands at current stage position."""
        # Set stage position and compute intersection
        self.optical_system.set_stage_position(stage_position)

        try:
            intersections = self.optical_system.find_intersections()
        except Exception:
            intersections = []

        # Update operand current values
        for operand in self.operands:
            if operand.operand_type == "intersection_target":
                if intersections:
                    operand.current_value = intersections[0]  # Take first intersection
                else:
                    operand.current_value = None

            elif operand.operand_type == "intersection_exists":
                operand.current_value = intersections

    def _merit_function(self, variables: np.ndarray) -> float:
        """
        Merit function for optimization.

        Args:
            variables: Current variable values [stage_x, stage_y, stage_z]

        Returns:
            Merit function value (sum of weighted operand errors)
        """
        # Reconstruct stage position from variables
        stage_position = variables[:3]  # Assume first 3 are x,y,z

        # Evaluate all operands
        self._evaluate_operands(stage_position)

        # Compute total merit function
        total_error = 0.0
        for operand in self.operands:
            error = operand.compute_error()
            total_error += error

        # Store iteration history
        self.iteration_history.append(
            {
                "stage_position": stage_position.copy(),
                "merit_function": total_error,
                "intersections": [
                    op.current_value
                    for op in self.operands
                    if op.operand_type.startswith("intersection")
                ],
            }
        )

        return total_error

    def optimize(self) -> OptimizationResult:
        """
        Run optimization to find optimal stage position.

        Returns:
            OptimizationResult with optimal position and convergence info
        """
        if not self.variables:
            raise ValueError(
                "No optimization variables defined. Call setup_stage_position_variables() first."
            )

        if not self.operands:
            raise ValueError(
                "No optimization operands defined. Call add_operand() first."
            )

        # Setup optimization problem
        initial_values = np.array([var.initial_value for var in self.variables])
        bounds = Bounds(
            lb=np.array([var.min_bound for var in self.variables]),
            ub=np.array([var.max_bound for var in self.variables]),
        )

        # Clear history
        self.iteration_history = []

        # Run optimization
        try:
            result = minimize(
                fun=self._merit_function,
                x0=initial_values,
                method=self.method,
                bounds=bounds,
                options={
                    "maxiter": self.max_iterations,
                    "ftol": self.tolerance,
                    "disp": False,
                },
            )

            # Process results
            optimal_position = result.x[:3]  # Extract stage position
            final_error = result.fun

            # Get final intersection
            self._evaluate_operands(optimal_position)
            final_intersections = [
                op.current_value
                for op in self.operands
                if op.operand_type == "intersection_target"
                and op.current_value is not None
            ]
            final_intersection = final_intersections[0] if final_intersections else None

            # Determine success based on final error, not just scipy success
            # scipy may report ABNORMAL when starting very close to solution
            error_converged = final_error < self.tolerance
            scipy_success = result.success

            # Consider successful if error is small, even if scipy reports issues
            success = error_converged or (
                scipy_success and final_error < 10 * self.tolerance
            )

            if success:
                status = OptimizationStatus.SUCCESS
            elif error_converged:
                status = OptimizationStatus.TOLERANCE_REACHED
            else:
                status = OptimizationStatus.FAILED

            # Get target intersection for comparison
            target_intersections = [
                op.target_value
                for op in self.operands
                if op.operand_type == "intersection_target"
            ]
            target_intersection = (
                target_intersections[0] if target_intersections else None
            )

            return OptimizationResult(
                success=success,
                status=status,
                optimal_position=optimal_position,
                final_intersection=final_intersection,
                target_intersection=target_intersection,
                final_error=final_error,
                num_iterations=result.nit,
                convergence_info={
                    "scipy_result": result,
                    "iteration_history": self.iteration_history,
                },
            )

        except Exception as e:
            return OptimizationResult(
                success=False,
                status=OptimizationStatus.FAILED,
                final_error=float("inf"),
                convergence_info={"error": str(e)},
            )


def find_stage_position_for_intersection(
    optical_system: OpticalSystem,
    target_intersection: POINT3,
    initial_guess: Optional[POINT3] = None,
    position_bounds: Optional[tuple] = None,
    method="L-BFGS-B",
) -> OptimizationResult:
    """
    Convenience function to find stage position for a target intersection.

    Args:
        optical_system: Complete optical system
        target_intersection: Desired 3D intersection point
        initial_guess: Initial stage position guess
        position_bounds: (min_pos, max_pos) bounds for stage position

    Returns:
        OptimizationResult with optimal stage position
    """
    # Setup optimizer
    optimizer = StagePositionOptimizer(optical_system)
    if method in ["L-BFGS-B", "Powell"]:
        optimizer.method = method
    else:
        warnings.warn(
            f"Unsupported optimization method {method}. Falling back to default 'L-BFGS-B'."
        )

    # Default initial guess from current stage position
    if initial_guess is None:
        current_pos = optical_system.stage.position_local
        if hasattr(current_pos, "magnitude"):
            initial_guess = np.array(current_pos.magnitude)
        else:
            initial_guess = np.array(current_pos)

    # Setup optimization problem
    optimizer.setup_stage_position_variables(initial_guess, position_bounds)
    optimizer.add_operand("intersection_target", target_intersection, weight=1.0)
    optimizer.add_operand(
        "intersection_exists", True, weight=10.0
    )  # Ensure intersection exists

    # Run optimization
    return optimizer.optimize()
