"""Parse MITgcm .meta files without reading binary .data."""

from __future__ import annotations

import glob
import math
import os
from dataclasses import dataclass

from MITgcmutils import mds


@dataclass
class MetaSummary:
    prefix: str
    iteration: int | None
    meta_path: str
    gdims: tuple[int, ...]
    nrecords: int
    dataprec: str
    shape: tuple[int, ...]
    itemsize: int
    nbytes: int

    @property
    def nbytes_human(self) -> str:
        size = float(self.nbytes)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024 or unit == "TB":
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} TB"


def _prefix_path(data_dir: str, prefix: str) -> str:
    return os.path.join(os.path.abspath(data_dir), prefix)


def find_meta_file(data_dir: str, prefix: str, iteration: int | None = None) -> str:
    """Return one representative .meta path for a prefix (and optional iteration)."""
    fname = _prefix_path(data_dir, prefix)
    if iteration is not None:
        fname = f"{fname}.{int(iteration):010d}"
    patterns = [
        fname + "." + 3 * "[0-9]" + "." + 3 * "[0-9]" + ".meta",
        fname + ".meta",
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No .meta file found for {prefix}" + (f" iter {iteration}" if iteration else ""))


def read_meta_summary(
    data_dir: str,
    prefix: str,
    iteration: int | None = None,
) -> MetaSummary:
    """Read grid dimensions and size estimates from .meta only."""
    meta_path = find_meta_file(data_dir, prefix, iteration)
    gdims, _, _, timestep, _, _, meta = mds.readmeta(meta_path)

    dataprec = meta.get("dataprec", meta.get("format", ["float32"]))[0]
    nrecords = int(meta.get("nrecords", [1])[0])
    itemsize = 8 if dataprec == "float64" else 4

    spatial = tuple(int(d) for d in gdims)
    if nrecords > 1:
        shape = (nrecords,) + spatial
    else:
        shape = spatial

    iter_val = None
    if timestep:
        iter_val = int(timestep[0])
    elif iteration is not None:
        iter_val = int(iteration)

    nbytes = nrecords * itemsize * math.prod(spatial)

    return MetaSummary(
        prefix=prefix,
        iteration=iter_val,
        meta_path=meta_path,
        gdims=spatial,
        nrecords=nrecords,
        dataprec=dataprec,
        shape=shape,
        itemsize=itemsize,
        nbytes=nbytes,
    )
