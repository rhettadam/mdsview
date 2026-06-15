"""Parse iteration, level, and region selections for CLI commands."""

from __future__ import annotations

from . import io
from .errors import IterationError, MdsViewError


def parse_int_list(spec: str, *, name: str = "value") -> list[int]:
    """Parse comma-separated integers or start:stop[:step] ranges (stop is exclusive)."""
    spec = spec.strip()
    if not spec:
        raise MdsViewError(f"Empty {name} list", exit_code=2)

    if ":" in spec:
        parts = [p.strip() for p in spec.split(":")]
        if len(parts) == 2:
            start, stop = (int(parts[0]), int(parts[1]))
            step = 1
        elif len(parts) == 3:
            start, stop, step = (int(parts[0]), int(parts[1]), int(parts[2]))
        else:
            raise MdsViewError(f"Invalid {name} range {spec!r}", exit_code=2)
        if step == 0:
            raise MdsViewError(f"{name} range step cannot be zero", exit_code=2)
        return list(range(start, stop, step))

    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def resolve_iterations(data_dir: str, prefix: str, spec: str) -> list[int]:
    """Resolve iteration selection: all, last, comma list, or range."""
    spec = spec.strip()
    available = io.list_iterations(data_dir, prefix)
    if not available:
        raise IterationError(f"{prefix} has no time snapshots (static grid file?)")

    lowered = spec.lower()
    if lowered == "all":
        return available
    if lowered == "last":
        return [available[-1]]

    chosen = parse_int_list(spec, name="iteration")
    available_set = set(available)
    for itr in chosen:
        if itr not in available_set:
            raise IterationError(
                f"Iteration {itr} not found for {prefix}. "
                f"Range: {available[0]} … {available[-1]} ({len(available)} snapshots)"
            )
    return chosen


def parse_level_list(spec: str | None, nz: int) -> list[int] | None:
    """Return level indices to export, or None for all levels."""
    if spec is None:
        return None
    if nz <= 1:
        return [0]
    levels = parse_int_list(spec, name="level")
    for lev in levels:
        if lev < 0 or lev >= nz:
            raise MdsViewError(f"Level {lev} out of range 0 … {nz - 1}", exit_code=2)
    return levels


def parse_region(spec: str | None) -> tuple[int, int, int, int] | None:
    """Parse I0,I1,J0,J1 region for io.read_field(region=...)."""
    if not spec:
        return None
    parts = [int(part.strip()) for part in spec.split(",")]
    if len(parts) != 4:
        raise MdsViewError("Region must be I0,I1,J0,J1 (four comma-separated indices)", exit_code=2)
    return tuple(parts)


def parse_prefix_list(spec: str) -> list[str]:
    """Parse comma-separated variable names."""
    names = [part.strip() for part in spec.split(",") if part.strip()]
    if not names:
        raise MdsViewError("Provide at least one variable name", exit_code=2)
    return names


def parse_index_pair(spec: str, *, name: str = "index") -> tuple[int, int]:
    """Parse I,J index pair."""
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 2:
        raise MdsViewError(f"{name} must be two comma-separated indices, e.g. 40,30", exit_code=2)
    return int(parts[0]), int(parts[1])
