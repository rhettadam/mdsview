"""Field arithmetic and combination operations."""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np

from . import io
from .slices import level_axis


def diff_fields(
    data_dir_a: str,
    prefix: str,
    iteration_a: int,
    iteration_b: int,
    *,
    data_dir_b: str | None = None,
    rec: int | None = None,
    lev: int | list[int] | None = None,
    region: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, dict]:
    """Subtract field at iteration_b from field at iteration_a (optional second run dir)."""
    if data_dir_b is None:
        data_dir_b = data_dir_a
    kwargs: dict = {"rec": rec, "region": region}
    if lev is not None:
        meta = io.read_field_meta(data_dir_a, prefix, iteration_a)
        if len(meta.shape) >= 3:
            kwargs["lev"] = lev
    a, _, meta_a = io.read_field(data_dir_a, prefix, iteration_a, **kwargs)
    b, _, _ = io.read_field(data_dir_b, prefix, iteration_b, **kwargs)
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {iteration_a}@{data_dir_a} {a.shape} vs "
            f"{iteration_b}@{data_dir_b} {b.shape}"
        )
    result = a.astype(np.float64) - b.astype(np.float64)
    meta = {
        "prefix": prefix,
        "iteration_a": iteration_a,
        "iteration_b": iteration_b,
        "data_dir_a": os.path.abspath(data_dir_a),
        "data_dir_b": os.path.abspath(data_dir_b),
        "operation": "subtract",
        "source_meta": meta_a,
    }
    return result, meta


def diff_slice(
    data_dir_a: str,
    prefix: str,
    iteration_a: int,
    iteration_b: int,
    level: int | None = None,
    *,
    data_dir_b: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Subtract two snapshots at one horizontal slice (always 2D)."""
    if data_dir_b is None:
        data_dir_b = data_dir_a
    meta_a = io.read_field_meta(data_dir_a, prefix, iteration_a)
    a2 = io.read_level_slice(data_dir_a, prefix, iteration_a, level)
    b2 = io.read_level_slice(data_dir_b, prefix, iteration_b, level)
    if a2.shape != b2.shape:
        raise ValueError(
            f"Slice shape mismatch: {iteration_a}@{data_dir_a} {a2.shape} vs "
            f"{iteration_b}@{data_dir_b} {b2.shape}"
        )
    result = a2.astype(np.float64) - b2.astype(np.float64)
    meta = {
        "prefix": prefix,
        "iteration_a": iteration_a,
        "iteration_b": iteration_b,
        "data_dir_a": os.path.abspath(data_dir_a),
        "data_dir_b": os.path.abspath(data_dir_b),
        "level": level,
        "operation": "subtract_slice",
        "source_meta": meta_a,
    }
    return result, meta


def combine_iterations(
    data_dir: str,
    prefix: str,
    iterations: Iterable[int],
    *,
    rec: int | None = None,
) -> tuple[np.ndarray, list[int], dict]:
    """
    Stack multiple iterations along a new leading time dimension.

    Returns (array, iterations, metadata).
    """
    iters = list(iterations)
    if not iters:
        raise ValueError("At least one iteration is required")

    slices = []
    meta_ref = None
    for itr in iters:
        arr, _, meta = io.read_field(data_dir, prefix, itr, rec=rec, squeeze=False)
        if meta_ref is None:
            meta_ref = meta
        elif arr.shape != slices[0].shape:
            raise ValueError(f"Inconsistent shape at iteration {itr}: {arr.shape}")
        slices.append(arr)

    stacked = np.stack(slices, axis=0)
    meta = {
        "prefix": prefix,
        "iterations": iters,
        "operation": "stack",
        "source_meta": meta_ref,
    }
    return stacked, iters, meta


def stats(arr: np.ndarray) -> dict[str, float]:
    """Basic summary statistics for a field."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan, "rms": np.nan}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "rms": float(np.sqrt(np.mean(finite**2))),
    }
