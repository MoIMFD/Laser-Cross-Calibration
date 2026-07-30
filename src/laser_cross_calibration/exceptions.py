"""Domain-specific exceptions.

Errors raised by the underlying `hazy-frames` geometry library (mismatched
frames, zero-length vectors, ...) are wrapped into `InvalidGeometryError` at
object-construction boundaries, so callers and tests depend on this
package's own exception type and messages instead of `hazy-frames`
internals, which may change wording between versions.
"""

from __future__ import annotations


class InvalidGeometryError(Exception):
    """Raised when geometric inputs used to construct an object are invalid."""
