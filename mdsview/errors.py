"""User-facing errors for CLI and library callers."""

from __future__ import annotations


class MdsViewError(Exception):
    """Expected failure (bad path, missing field, invalid iteration)."""

    exit_code = 1

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class DataDirError(MdsViewError):
    pass


class FieldError(MdsViewError):
    pass


class IterationError(MdsViewError):
    pass
