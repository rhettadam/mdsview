"""Field arithmetic and combination operations."""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Callable, Iterable

import numpy as np

from . import io
from .slices import level_axis, pick_2d_slice


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


def difference_of_differences(
    data_dir: str,
    prefix_a: str,
    prefix_b: str,
    iter_1: int,
    iter_2: int,
    *,
    rec: int | None = None,
    levels: list[int] | None = None,
    region: tuple[int, int, int, int] | None = None,
    progress: bool = True,
    on_level: Callable[[int, int], None] | None = None,
    mmap_path: str | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Difference-of-differences (DiD) between two variables at two times:

        (B(t1) - A(t1)) - (B(t2) - A(t2))

    Equivalent to the change in (B - A) from t1 to t2. Processes one vertical
    level at a time when the field is 3D so peak RAM stays near four 2D slabs,
    not four full 3D volumes (~2 GB each).

    Returns a memmap-backed float32 array unless the field is small enough to
    fit comfortably in memory as a regular ndarray.
    """
    meta_a1 = io.read_field_meta(data_dir, prefix_a, iter_1)
    meta_b1 = io.read_field_meta(data_dir, prefix_b, iter_1)
    meta_a2 = io.read_field_meta(data_dir, prefix_a, iter_2)
    meta_b2 = io.read_field_meta(data_dir, prefix_b, iter_2)

    for label, left, right in (
        ("iter_1", meta_a1, meta_b1),
        ("iter_2", meta_a2, meta_b2),
    ):
        if left.shape != right.shape:
            raise ValueError(
                f"Shape mismatch at {label}: {prefix_a} {left.shape} vs {prefix_b} {right.shape}"
            )
    if meta_a1.shape != meta_a2.shape:
        raise ValueError(
            f"Shape mismatch for {prefix_a}: iter {iter_1} {meta_a1.shape} vs iter {iter_2} {meta_a2.shape}"
        )

    shape = meta_a1.shape
    level_axis_idx, nz = level_axis(shape)
    level_indices = list(levels) if levels is not None else list(range(nz))

    out_shape = _output_shape(shape, level_indices, level_axis_idx)
    if mmap_path is None:
        mmap_path = tempfile.NamedTemporaryFile(prefix="mdsview_dod_", suffix=".data", delete=False).name
    out = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=out_shape)

    read_kwargs: dict = {"rec": rec, "region": region, "squeeze": False}
    total = len(level_indices)

    try:
        if level_axis_idx is not None and nz > 1:
            for out_k, level in enumerate(level_indices):
                if progress:
                    _print_progress(out_k + 1, total, prefix=f"level {level}")
                if on_level:
                    on_level(out_k, level)

                lev = [level]
                a1, _, _ = io.read_field(data_dir, prefix_a, iter_1, lev=lev, **read_kwargs)
                b1, _, _ = io.read_field(data_dir, prefix_b, iter_1, lev=lev, **read_kwargs)
                a2, _, _ = io.read_field(data_dir, prefix_a, iter_2, lev=lev, **read_kwargs)
                b2, _, _ = io.read_field(data_dir, prefix_b, iter_2, lev=lev, **read_kwargs)

                slab = (
                    b1.astype(np.float64)
                    - a1.astype(np.float64)
                    - b2.astype(np.float64)
                    + a2.astype(np.float64)
                )
                _assign_level(out, out_k, np.squeeze(slab))
        else:
            if progress:
                _print_progress(1, 1, prefix="field")
            a1, _, _ = io.read_field(data_dir, prefix_a, iter_1, **read_kwargs)
            b1, _, _ = io.read_field(data_dir, prefix_b, iter_1, **read_kwargs)
            a2, _, _ = io.read_field(data_dir, prefix_a, iter_2, **read_kwargs)
            b2, _, _ = io.read_field(data_dir, prefix_b, iter_2, **read_kwargs)
            out[...] = (
                b1.astype(np.float64)
                - a1.astype(np.float64)
                - b2.astype(np.float64)
                + a2.astype(np.float64)
            ).astype(np.float32)
    except Exception:
        out._mmap.close()  # type: ignore[attr-defined]
        raise

    out.flush()
    meta = {
        "prefix_a": prefix_a,
        "prefix_b": prefix_b,
        "iter_1": iter_1,
        "iter_2": iter_2,
        "operation": "difference_of_differences",
        "formula": "(B(t1)-A(t1)) - (B(t2)-A(t2))",
        "shape": tuple(out_shape),
        "levels": level_indices,
        "nbytes_per_input": meta_a1.nbytes_human,
        "mmap_path": mmap_path,
    }
    return out, meta


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


def streaming_stats(
    data_dir: str,
    prefix_a: str,
    prefix_b: str,
    iter_1: int,
    iter_2: int,
    *,
    rec: int | None = None,
) -> dict[str, float]:
    """Global stats for a DiD field without retaining the full array in memory."""
    meta_a1 = io.read_field_meta(data_dir, prefix_a, iter_1)
    level_axis_idx, nz = level_axis(meta_a1.shape)
    level_indices = list(range(nz))

    count = 0
    vmin = np.inf
    vmax = -np.inf
    sum_val = 0.0
    sumsq = 0.0

    read_kwargs: dict = {"rec": rec, "squeeze": False}
    for level in level_indices:
        if level_axis_idx is not None and nz > 1:
            lev = [level]
            a1, _, _ = io.read_field(data_dir, prefix_a, iter_1, lev=lev, **read_kwargs)
            b1, _, _ = io.read_field(data_dir, prefix_b, iter_1, lev=lev, **read_kwargs)
            a2, _, _ = io.read_field(data_dir, prefix_a, iter_2, lev=lev, **read_kwargs)
            b2, _, _ = io.read_field(data_dir, prefix_b, iter_2, lev=lev, **read_kwargs)
            slab = np.squeeze(
                b1.astype(np.float64) - a1.astype(np.float64) - b2.astype(np.float64) + a2.astype(np.float64)
            )
        else:
            a1, _, _ = io.read_field(data_dir, prefix_a, iter_1, **read_kwargs)
            b1, _, _ = io.read_field(data_dir, prefix_b, iter_1, **read_kwargs)
            a2, _, _ = io.read_field(data_dir, prefix_a, iter_2, **read_kwargs)
            b2, _, _ = io.read_field(data_dir, prefix_b, iter_2, **read_kwargs)
            slab = (
                b1.astype(np.float64) - a1.astype(np.float64) - b2.astype(np.float64) + a2.astype(np.float64)
            )

        finite = slab[np.isfinite(slab)]
        if finite.size == 0:
            continue
        count += int(finite.size)
        sum_val += float(np.sum(finite))
        sumsq += float(np.sum(finite**2))
        vmin = min(vmin, float(np.min(finite)))
        vmax = max(vmax, float(np.max(finite)))

        if level_axis_idx is None or nz <= 1:
            break

    if count == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan, "rms": np.nan}
    mean = sum_val / count
    variance = max(sumsq / count - mean**2, 0.0)
    return {
        "min": vmin,
        "max": vmax,
        "mean": mean,
        "std": float(np.sqrt(variance)),
        "rms": float(np.sqrt(sumsq / count)),
    }


def _output_shape(
    shape: tuple[int, ...],
    level_indices: list[int],
    level_axis_idx: int | None,
) -> tuple[int, ...]:
    if level_axis_idx is not None and len(level_indices) < level_axis(shape)[1]:
        if len(shape) == 4:
            return (shape[0], len(level_indices), shape[2], shape[3])
        if len(shape) == 3:
            return (len(level_indices), shape[1], shape[2])
    return shape


def _assign_level(out: np.ndarray, out_k: int, slab: np.ndarray) -> None:
    if out.ndim == slab.ndim + 1:
        out[out_k] = slab.astype(np.float32)
    elif out.ndim == slab.ndim:
        out[...] = slab.astype(np.float32)
    else:
        out[out_k] = slab.astype(np.float32)


def _print_progress(step: int, total: int, *, prefix: str) -> None:
    pct = 100.0 * step / total
    sys.stderr.write(f"\r  DiD {prefix}: {step}/{total} ({pct:.1f}%)")
    sys.stderr.flush()
    if step == total:
        sys.stderr.write("\n")
