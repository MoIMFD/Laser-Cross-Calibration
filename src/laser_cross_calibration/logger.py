"""Simple logging configuration using loguru."""

from loguru import logger
import sys

# Configure loguru with a nice format
logger.remove()  # Remove default handler

# Add console handler with custom format
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# Export the logger for use in other modules
__all__ = ["logger"]