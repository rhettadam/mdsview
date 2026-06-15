"""Extract spatial, vertical, and time subsets to new MDS files."""

from __future__ import annotations

import os
from typing import Any

from . import io
from .export_nc import _read_snapshot
from .selection import parse_level_list, parse_region, resolve_iterations
from .slices import level_axis


def extract_field(
    data_dir: str,
    prefix: str,
    output_dir: str,
    *,
    iterations: str = "all",
    levels: str | None = None,
    region: str | None = None,
    rec: int | None = None,
) -> dict[str, Any]:
    """
    Write subset snapshots as new .data/.meta files under output_dir.

    Reads one snapshot at a time so peak memory stays near one subset volume.
    """
    data_dir = os.path.abspath(data_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    iters = resolve_iterations(data_dir, prefix, iterations)
    region_t = parse_region(region)
    shape = io.field_info(data_dir, prefix).shape
    _, nz = level_axis(shape)
    level_list = parse_level_list(levels, nz)

    read_kwargs: dict[str, Any] = {"rec": rec, "region": region_t, "squeeze": False}
    bytes_written = 0
    out_paths: list[str] = []

    for itr in iters:
        arr = _read_snapshot(data_dir, prefix, itr, level_list, read_kwargs)
        meta = io.read_field_meta(data_dir, prefix, itr)
        out_base = os.path.join(output_dir, prefix)
        io.write_field(out_base, arr, iteration=itr, dataprec=meta.dataprec)
        bytes_written += arr.nbytes
        out_paths.append(f"{out_base}.{itr:010d}")

    return {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "prefix": prefix,
        "iterations": iters,
        "levels": level_list,
        "region": list(region_t) if region_t else None,
        "snapshots_written": len(iters),
        "bytes_written": bytes_written,
        "sample_path": out_paths[0] if out_paths else None,
    }
