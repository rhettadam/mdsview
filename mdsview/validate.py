"""Input validation for CLI and scripts."""

from __future__ import annotations

import os

from . import io
from .errors import DataDirError, FieldError, IterationError


def abs_data_dir(path: str) -> str:
    data_dir = os.path.abspath(path)
    if not os.path.isdir(data_dir):
        raise DataDirError(f"Not a directory: {data_dir}")
    return data_dir


def require_prefix(data_dir: str, prefix: str) -> None:
    prefixes = io.list_prefixes(data_dir)
    if prefix not in prefixes:
        hint = ", ".join(prefixes[:8])
        suffix = " …" if len(prefixes) > 8 else ""
        raise FieldError(f"Unknown variable {prefix!r}. Available: {hint}{suffix}")


def require_iteration(data_dir: str, prefix: str, iteration: int) -> None:
    iters = io.list_iterations(data_dir, prefix)
    if not iters:
        raise IterationError(f"{prefix} has no time snapshots (static grid file?)")
    if iteration not in iters:
        raise IterationError(
            f"Iteration {iteration} not found for {prefix}. "
            f"Range: {iters[0]} … {iters[-1]} ({len(iters)} snapshots)"
        )


def resolve_iteration(data_dir: str, prefix: str, raw: str | int | None) -> int | None:
    """Parse iteration argument; None for grid files, int otherwise."""
    if raw is None:
        return None
    if isinstance(raw, int):
        iteration = raw
    elif str(raw).lower() == "last":
        iters = io.list_iterations(data_dir, prefix)
        if not iters:
            raise IterationError(f"No iterations found for {prefix}")
        return iters[-1]
    else:
        try:
            iteration = int(raw)
        except (TypeError, ValueError) as exc:
            raise IterationError(f"Invalid iteration {raw!r}") from exc
    require_iteration(data_dir, prefix, iteration)
    return iteration
