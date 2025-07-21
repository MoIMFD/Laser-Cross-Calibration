from abc import ABC, abstractmethod
from enum import StrEnum
from warnings import warn

from pydantic import BaseModel, Field, model_validator


class GCodes(StrEnum):
    RAPID_POSITIONING = "G00"
    LINEAR_TRAVEL = "G01"
    HOME_AXIS = "G28"
    PAUSE = "G04"


class MCodes(StrEnum):
    RESET_DEFAULTS = "M502"
    SET_PIN = "M42"
    SET_TRAVEL_SPEED = "M203"


class OtherCodes(StrEnum):
    COMMENT = ";"


COMMAND_TYPES = GCodes | MCodes | OtherCodes


# Base interface
class BaseCommand(BaseModel, ABC):
    type: COMMAND_TYPES
    comment: str | None = None

    def _get_parameter_names(self) -> list[str]:
        """Override in subclasses to specify parameter order"""
        return []

    def to_gcode(self) -> str:
        """Default implementation - can be overridden for special cases"""
        parts: list[str] = [self.type]

        for param in self._get_parameter_names():
            value = getattr(self, param, None)
            if value is not None:
                parts.append(f"{param.upper()}{value}")

        if self.comment:
            parts.append(f"; {self.comment}")

        return " ".join(parts)

    @classmethod
    @abstractmethod
    def from_params(cls, **kwargs) -> "BaseCommand":
        """Create command from parameters"""
        pass

    def __str__(self) -> str:
        return self.to_gcode()


class CommentCommand(BaseCommand):
    type: COMMAND_TYPES = OtherCodes.COMMENT

    def to_gcode(self) -> str:
        comment = self.comment if self.comment is not None else ""
        return f"{OtherCodes.COMMENT} {comment}"

    @classmethod
    def from_params(cls, **kwargs) -> "CommentCommand":
        return cls(**kwargs)


class G01Command(BaseCommand):
    type: COMMAND_TYPES = GCodes.LINEAR_TRAVEL
    x: float | None = None
    y: float | None = None
    z: float | None = None
    e: float | None = None
    f: float | None = None

    @model_validator(mode="after")
    def require_axis(self):
        if not any([self.x, self.y, self.z, self.e, self.f]):
            raise ValueError("G01 requires at least one axis")
        return self

    def _get_parameter_names(self) -> list[str]:
        return ["x", "y", "z", "e", "f"]

    @classmethod
    def from_params(cls, **kwargs) -> "G01Command":
        return cls(**kwargs)


class G00Command(BaseCommand):
    type: COMMAND_TYPES = GCodes.RAPID_POSITIONING
    x: float | None = None
    y: float | None = None
    z: float | None = None
    e: float | None = None
    f: float | None = None

    @model_validator(mode="after")
    def require_axis(self):
        if not any([self.x, self.y, self.z, self.e, self.f]):
            raise ValueError("G00 requires at least one axis")
        return self

    def _get_parameter_names(self) -> list[str]:
        return ["x", "y", "z", "e", "f"]

    @classmethod
    def from_params(cls, **kwargs) -> "G00Command":
        return cls(**kwargs)


class M203Command(BaseCommand):
    type: COMMAND_TYPES = MCodes.SET_TRAVEL_SPEED
    x: float | None = Field(default=None, ge=0.0)
    y: float | None = Field(default=None, ge=0.0)
    z: float | None = Field(default=None, ge=0.0)
    e: float | None = Field(default=None, ge=0.0)
    f: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def require_axis(self):
        if not any([self.x, self.y, self.z, self.e, self.f]):
            raise ValueError("M203 requires at least one axis")
        return self

    def _get_parameter_names(self) -> list[str]:
        return ["x", "y", "z", "e", "f"]

    @classmethod
    def from_params(cls, **kwargs) -> "M203Command":
        return cls(**kwargs)


class M42Command(BaseCommand):
    type: COMMAND_TYPES = MCodes.SET_PIN
    p: int = Field(..., ge=0, description="pin to modify")
    s: int = Field(..., ge=0, description="value to set")

    def _get_parameter_names(self) -> list[str]:
        return ["p", "s"]

    @classmethod
    def from_params(cls, **kwargs) -> "M42Command":
        return cls(**kwargs)


class M502Command(BaseCommand):
    type: COMMAND_TYPES = MCodes.RESET_DEFAULTS

    @classmethod
    def from_params(cls, **kwargs) -> "M502Command":
        return cls(**kwargs)


class G04Command(BaseCommand):
    type: COMMAND_TYPES = GCodes.PAUSE
    s: float | None = Field(None, ge=0.0, description="Pause time in milliseconds")
    p: float | None = Field(None, ge=0.0, description="Pause time in seconds")

    @model_validator(mode="after")
    def require_s_or_p(self):
        if not any([self.s, self.p]):
            raise ValueError("G4 requires at least s or p")
        return self

    @model_validator(mode="after")
    def check_if_both_present(self):
        if all([self.p, self.s]):
            warn("Both s and p are specified in a G04 command. S takes precedence.")
        return self

    def _get_parameter_names(self) -> list[str]:
        return ["p", "s"]

    @classmethod
    def from_params(cls, **kwargs) -> "G04Command":
        return cls(**kwargs)
