"""Timestep playback and GIF export helpers."""

from __future__ import annotations

import io as pyio
import os
import tempfile
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from . import io, plotting


def resolve_playback_clim(
    data_dir: str,
    prefix: str,
    iterations: list[int],
    level: int,
    *,
    user_vmin: float | None,
    user_vmax: float | None,
    lock_scale: bool,
    symmetric: bool = False,
    max_samples: int = 64,
    anchor_iteration: int | None = None,
) -> tuple[float | None, float | None]:
    """Colour limits for playback/GIF; optionally fixed across frames."""
    if not lock_scale:
        return user_vmin, user_vmax

    if user_vmin is not None and user_vmax is not None:
        return user_vmin, user_vmax
    if symmetric and user_vmax is not None:
        return -user_vmax, user_vmax

    if anchor_iteration is not None:
        field2d = io.read_level_slice(data_dir, prefix, anchor_iteration, level)
        return plotting.resolved_clim(
            field2d, user_vmin, user_vmax, symmetric=symmetric,
        )

    sample_iters = _subsample_iterations(iterations, max_samples)

    if symmetric:
        peak = 0.0
        for itr in sample_iters:
            sl = io.read_level_slice(data_dir, prefix, itr, level)
            peak = max(peak, float(np.nanmax(np.abs(sl))))
        limit = peak or 1.0
        return -limit, limit

    vmin, vmax = np.inf, -np.inf
    for itr in sample_iters:
        sl = io.read_level_slice(data_dir, prefix, itr, level)
        finite = sl[np.isfinite(sl)]
        if finite.size:
            vmin = min(vmin, float(np.min(finite)))
            vmax = max(vmax, float(np.max(finite)))
    if not np.isfinite(vmin):
        return user_vmin, user_vmax
    return vmin, vmax


def _subsample_iterations(iterations: list[int], max_samples: int) -> list[int]:
    if max_samples <= 0 or len(iterations) <= max_samples:
        return list(iterations)
    idx = np.linspace(0, len(iterations) - 1, max_samples, dtype=int)
    return [iterations[int(i)] for i in idx]


def save_playback_gif(
    data_dir: str,
    prefix: str,
    iterations: list[int],
    level: int,
    output_path: str,
    *,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    lock_scale: bool = True,
    fps: float = 2.0,
    coord_mode: str = "centers",
    overlay_grid: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Render a timestep loop to an animated GIF (one frame at a time on disk)."""
    if not iterations:
        raise ValueError("No iterations to animate")

    clim_vmin, clim_vmax = resolve_playback_clim(
        data_dir,
        prefix,
        iterations,
        level,
        user_vmin=vmin,
        user_vmax=vmax,
        lock_scale=lock_scale,
        symmetric=False,
    )

    duration_ms = max(int(1000 / max(fps, 0.1)), 50)
    total = len(iterations)
    field_shape = io.field_info(data_dir, prefix).shape

    with tempfile.TemporaryDirectory(prefix="mdsview_gif_") as tmpdir:
        frame_paths: list[str] = []
        fig, ax = plt.subplots(figsize=(7, 5))
        try:
            for i, iteration in enumerate(iterations):
                field2d = io.read_level_slice(data_dir, prefix, iteration, level)
                title = plotting.format_field_title(
                    prefix, iteration, level=level, shape=field_shape,
                )
                plotting.draw_slice_on_ax(
                    ax,
                    field2d,
                    data_dir,
                    title=title,
                    cmap=cmap,
                    vmin=clim_vmin,
                    vmax=clim_vmax,
                    clear=True,
                    colorbar_label=prefix,
                    coord_mode=coord_mode,
                    overlay_grid=overlay_grid,
                )
                path = os.path.join(tmpdir, f"frame_{i:06d}.png")
                fig.savefig(path, dpi=120, bbox_inches="tight")
                frame_paths.append(path)
                if progress:
                    progress(i + 1, total)
        finally:
            plt.close(fig)

        first = Image.open(frame_paths[0]).convert("P", palette=Image.ADAPTIVE, colors=256)
        rest = [
            Image.open(p).convert("P", palette=first.palette)
            for p in frame_paths[1:]
        ]
        first.save(
            output_path,
            save_all=True,
            append_images=rest,
            duration=duration_ms,
            loop=0,
            disposal=2,
        )
        for im in rest:
            im.close()
        first.close()
