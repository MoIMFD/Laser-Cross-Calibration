from enum import StrEnum
from pydantic import BaseModel, model_validator, Field
from typing import Dict, List, Optional, Any, Union
from .logger import logger

from collections import ChainMap


class GCodeCommandType(StrEnum):
    RAPID_MOVE = "G0"
    LINEAR_MOVE = "G1"
    CW_ARC = "G2"
    CCW_ARC = "G3"
    PAUSE = "G4"
    SET_PIN = "M42"
    COMMENT = ";"


MOVEMENT_COMMANDS = {GCodeCommandType.RAPID_MOVE, GCodeCommandType.LINEAR_MOVE}
PAUSE_COMMANDS = {GCodeCommandType.PAUSE}


class GCodeCommand(BaseModel):
    """Template definition for a G-code command."""

    type: GCodeCommandType
    description: str
    required_params: List[str] = Field(default_factory=list)
    optional_params: List[str] = Field(default_factory=list)
    default_params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_command(self) -> "GCodeCommand":
        if self.type in MOVEMENT_COMMANDS:
            # Movement commands should have at least one axis parameter
            axis_params = [
                p
                for p in self.required_params + self.optional_params
                if p.lower() in ["x", "y", "z"]
            ]
            if not axis_params:
                raise ValueError(
                    f"Movement command {self.type} must have at least one axis parameter (X, Y, Z)"
                )

        # Validate parameter lists don't have duplicates
        all_params = self.required_params + self.optional_params
        if len(all_params) != len(set(all_params)):
            raise ValueError(
                "Duplicate parameters found in required_params or optional_params"
            )

        return self

    def create_instance(self, **kwargs) -> "GCodeInstance":
        """Create an instance of this command with specific parameter values."""
        return GCodeInstance(command=self, params=kwargs)

    def __call__(self, **kwargs) -> "GCodeInstance":
        """Create an instance of this command with specific parameter values."""
        return self.create_instance(**kwargs)


class GCodeInstance(BaseModel):
    """An actual G-code command instance with parameter values."""

    command: GCodeCommand
    params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_instance(self) -> "GCodeInstance":
        # Check required parameters
        missing_params = [
            p for p in self.command.required_params if p not in self.params
        ]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        # Check for invalid parameters
        allowed_params = set(
            self.command.required_params + self.command.optional_params
        )
        invalid_params = [p for p in self.params.keys() if p not in allowed_params]
        if invalid_params:
            raise ValueError(f"Invalid parameters: {invalid_params}")

        return self

    def render(self) -> str:
        """Render this command instance to G-code string."""
        # Merge with defaults, instance params take precedence
        all_params = ChainMap(self.params, self.command.default_params)

        # Special handling for comments
        if self.command.type == GCodeCommandType.COMMENT:
            return f"; {all_params.get('comment', '')}"

        # Special handling for pause commands (S takes precedence over P)
        if self.command.type in PAUSE_COMMANDS:
            has_s = "s" in all_params and all_params["s"] is not None
            has_p = "p" in all_params and all_params["p"] is not None

            if has_s and has_p:
                logger.warning(
                    f"Both S and P parameters provided for {self.command.type}. S takes precedence, but both will be included in G-code."
                )
            elif not (has_s or has_p):
                raise ValueError(
                    "Pause command must have either S (seconds) or P (milliseconds) parameter"
                )

        # Format parameters (exclude 'comment' from regular parameters)
        formatted_params = []
        for key, value in all_params.items():
            if value is not None and key != "comment":
                formatted_params.append(f"{key.upper()}{value}")

        parts = [self.command.type] + formatted_params
        result = " ".join(parts)

        # Add inline comment if provided (but not for COMMENT commands)
        if self.command.type != GCodeCommandType.COMMENT:
            comment = all_params.get("comment")
            if comment:
                result += f" ; {comment}"

        return result

    def __add__(self, other: Union["GCodeInstance", "GCodeProgram"]) -> "GCodeProgram":
        """Combine commands/programs using the + operator."""
        if isinstance(other, GCodeInstance):
            return GCodeProgram(commands=[self, other])
        elif isinstance(other, GCodeProgram):
            return GCodeProgram(commands=[self] + other.commands)
        else:
            raise TypeError(f"Cannot add {type(other)} to GCodeInstance")


class GCodeProgram(BaseModel):
    """A program consisting of multiple G-code command instances and/or sub-programs."""

    commands: List[Union["GCodeInstance", "GCodeProgram"]] = Field(default_factory=list)

    def add_command(self, command: Union["GCodeInstance", "GCodeProgram"]) -> None:
        """Add a command instance or sub-program to the program."""
        self.commands.append(command)

    def __add__(self, other: Union["GCodeInstance", "GCodeProgram"]) -> "GCodeProgram":
        """Combine with commands/programs using the + operator."""
        if isinstance(other, GCodeInstance):
            return GCodeProgram(commands=self.commands + [other])
        elif isinstance(other, GCodeProgram):
            return GCodeProgram(commands=self.commands + other.commands)
        else:
            raise TypeError(f"Cannot add {type(other)} to GCodeProgram")

    def render(self) -> str:
        """Render the entire program to G-code string, expanding sub-programs."""
        rendered_lines = []

        for item in self.commands:
            if isinstance(item, GCodeProgram):
                # Recursively render sub-programs
                sub_rendered = item.render()
                if sub_rendered:  # Only add non-empty renders
                    rendered_lines.append(sub_rendered)
            else:
                # Render individual commands
                rendered_lines.append(item.render())

        return "\n".join(rendered_lines)


# Command definitions (templates)
RAPID_MOVE = GCodeCommand(
    type=GCodeCommandType.RAPID_MOVE,
    description="Rapid positioning movement",
    optional_params=["x", "y", "z", "f", "comment"],
)

LINEAR_MOVE = GCodeCommand(
    type=GCodeCommandType.LINEAR_MOVE,
    description="Linear interpolation movement",
    optional_params=["x", "y", "z", "f", "comment"],
)

PAUSE = GCodeCommand(
    type=GCodeCommandType.PAUSE,
    description="Pause the command queue for S (seconds) or P (milliseconds). S takes precedence if both provided.",
    optional_params=["s", "p", "comment"],
)

SET_PIN = GCodeCommand(
    type=GCodeCommandType.SET_PIN,
    description="Set pin state (M42 P<pin> S<value>)",
    required_params=["p", "s"],
    optional_params=["comment"],
)

COMMENT = GCodeCommand(
    type=GCodeCommandType.COMMENT,
    description="Add a comment",
    required_params=["comment"],
)
