"""2-D slice selection and MITgcm level read hints."""

from __future__ import annotations

import numpy as np


def level_axis(shape: tuple[int, ...]) -> tuple[int | None, int]:
    """Return (level_axis_index, n_levels) for MITgcm-style arrays."""
    if len(shape) == 4:
        return 1, int(shape[1])
    if len(shape) >= 3:
        return 0, int(shape[-3])
    return None, 1


def pick_2d_slice(arr: np.ndarray, level: int | None) -> np.ndarray:
    """Reduce an array to 2-D for plotting or differencing."""
    data = np.squeeze(arr)
    if data.ndim == 2:
        return data
    if data.ndim == 1:
        return data[np.newaxis, :]
    if data.ndim > 2:
        _, nz = level_axis(data.shape)
        if nz <= 1:
            return np.squeeze(data)
        if level is None:
            level = nz // 2
        level = max(0, min(nz - 1, int(level)))
        return np.squeeze(data[level])
    raise ValueError(f"Cannot slice array with shape {arr.shape}")


def mitgcm_lev_kw(shape: tuple[int, ...], level: int | None) -> dict:
    """
    Return kwargs for io.read_field so MITgcmutils reads one horizontal slab.

    For 2-D fields returns {} (read the whole field). For 3-D, passes lev= when
    a level index is given.
    """
    _, nz = level_axis(shape)
    if nz <= 1 or level is None:
        return {}
    level = max(0, min(nz - 1, int(level)))
    return {"lev": level, "squeeze": False}
