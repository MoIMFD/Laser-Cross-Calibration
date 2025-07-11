from typing import Any
from pathlib import Path

from pydantic import ConfigDict, Field, BaseModel
from pydantic_settings import BaseSettings

from .constants import DeviceType


class DeviceConstraints(BaseModel):
    device_type: DeviceType

    x_min: float = Field(..., description="Minimum position of x axis")
    x_max: float = Field(..., description="Maximum position of x axis")
    y_min: float = Field(..., description="Minimum position of y axis")
    y_max: float = Field(..., description="Maximum position of y axis")
    z_min: float = Field(..., description="Minimum position of z axis")
    z_max: float = Field(..., description="Maximum position of z axis")

    max_travel_velocity: float = Field(
        ..., description="Maximum moving velocity of the device"
    )
    max_travel_acceleration: float = Field(
        ..., description="Maximum moving acceleration of the device"
    )


class Config(BaseSettings):
    """Main configuration class that loads and manages all configurations."""

    model_config = ConfigDict(
        extra="ignore", env_file=".env", env_file_encoding="utf-8"
    )

    device: DeviceConstraints = Field(default_factory=DeviceConstraints)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """Create Config instance from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            "preprocessing": self.preprocessing.model_dump(),
        }
