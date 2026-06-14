"""Read and write MITgcm MDS (.data/.meta) files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from MITgcmutils import mds

from .catalog import get_catalog
from .meta import MetaSummary, read_meta_summary
from .slices import mitgcm_lev_kw, pick_2d_slice


@dataclass
class FieldInfo:
    prefix: str
    iterations: list[int]
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    nbytes_human: str
    meta: dict


def _prefix_path(data_dir: str, prefix: str) -> str:
    data_dir = os.path.abspath(data_dir)
    return os.path.join(data_dir, prefix)


def list_prefixes(data_dir: str) -> list[str]:
    """Return variable prefixes that have at least one .meta file."""
    return get_catalog(data_dir).prefixes


def list_iterations(data_dir: str, prefix: str) -> list[int]:
    """Return sorted iteration numbers available for a prefix."""
    return get_catalog(data_dir).iters_for(prefix)


def read_field(
    data_dir: str,
    prefix: str,
    iteration: int | Iterable[int] | Literal["all", "last"] | None = None,
    *,
    rec: int | list[int] | None = None,
    lev: int | list[int] | tuple | None = None,
    region: tuple[int, int, int, int] | None = None,
    fill_value: float = 0.0,
    squeeze: bool = True,
) -> tuple[np.ndarray, list[int], dict]:
    """
    Read one MITgcm field.

    Returns (array, iterations, metadata). When iteration is None, reads
    grid files or single-snapshot fields without an iteration suffix.
    """
    fname = _prefix_path(data_dir, prefix)
    if iteration is None:
        itrs = -1
    elif iteration == "all":
        itrs = np.nan
    elif iteration == "last":
        itrs = np.inf
    else:
        itrs = iteration

    kwargs: dict = dict(
        rec=rec,
        region=region,
        fill_value=fill_value,
        returnmeta=True,
        squeeze=squeeze,
    )
    if lev is not None:
        kwargs["lev"] = lev

    arr, its, meta = mds.rdmds(fname, itrs, **kwargs)
    return arr, list(its), meta


def read_level_slice(
    data_dir: str,
    prefix: str,
    iteration: int | None,
    level: int | None = None,
    *,
    rec: int | None = None,
) -> np.ndarray:
    """
    Read one horizontal slice with minimal I/O.

    For 3-D fields with a level index, passes lev= to MITgcmutils so only that
    slab is read from disk.
    """
    summary = read_meta_summary(data_dir, prefix, iteration)
    lev_kw = mitgcm_lev_kw(summary.shape, level)
    squeeze = lev_kw.pop("squeeze", True)
    arr, _, _ = read_field(data_dir, prefix, iteration, rec=rec, squeeze=squeeze, **lev_kw)
    return pick_2d_slice(arr, level)


def read_field_meta(
    data_dir: str,
    prefix: str,
    iteration: int | None = None,
) -> MetaSummary:
    """Read shape and size from .meta without loading .data."""
    return read_meta_summary(data_dir, prefix, iteration)


def field_info(data_dir: str, prefix: str) -> FieldInfo:
    """Summarize a field prefix using .meta only (no binary read)."""
    cat = get_catalog(data_dir)
    iters = cat.iters_for(prefix)
    summary = cat.meta_summary(prefix)
    return FieldInfo(
        prefix=prefix,
        iterations=iters,
        shape=summary.shape,
        dtype=summary.dataprec,
        nbytes=summary.nbytes,
        nbytes_human=summary.nbytes_human,
        meta={"dataprec": [summary.dataprec], "nrecords": [summary.nrecords]},
    )


def write_field(
    output_path: str,
    arr: np.ndarray,
    *,
    iteration: int | None = None,
    dataprec: str = "float32",
    times: list[float] | None = None,
    fields: list[str] | None = None,
) -> None:
    """Write array to MDS .data/.meta pair."""
    mds.wrmds(
        output_path,
        arr,
        itr=iteration,
        dataprec=dataprec,
        times=times,
        fields=fields,
    )
