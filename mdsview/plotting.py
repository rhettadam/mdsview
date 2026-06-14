"""Matplotlib helpers for MITgcm fields."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from . import io
from .colormaps import DEFAULT_DIFF_CMAP, DEFAULT_PLOT_CMAP, resolve_cmap
from .slices import level_axis, pick_2d_slice

# Backward-compatible alias used in tests
_pick_2d_slice = pick_2d_slice


def _resolve_clim(
    field2d: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    *,
    symmetric: bool,
) -> tuple[float | None, float | None]:
    """Resolve colour limits; blank inputs mean auto-scale from data."""
    if vmin is not None and vmax is not None:
        return vmin, vmax
    if symmetric:
        if vmax is not None:
            return -vmax, vmax
        if vmin is not None:
            return vmin, -vmin
        limit = float(np.nanmax(np.abs(field2d))) or 1.0
        return -limit, limit
    if vmin is not None or vmax is not None:
        return vmin, vmax
    return None, None


def resolved_clim(
    field2d: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    *,
    symmetric: bool = False,
) -> tuple[float, float]:
    """Return the colour limits that will be used for plotting."""
    plot_vmin, plot_vmax = _resolve_clim(field2d, vmin, vmax, symmetric=symmetric)
    finite = field2d[np.isfinite(field2d)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(plot_vmin) if plot_vmin is not None else float(np.nanmin(finite))
    hi = float(plot_vmax) if plot_vmax is not None else float(np.nanmax(finite))
    return lo, hi


def format_cmap_limits_meta(
    cmap: str,
    field2d: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    *,
    symmetric: bool = False,
) -> str:
    lo, hi = resolved_clim(field2d, vmin, vmax, symmetric=symmetric)
    return f"{cmap}  ·  {lo:.4g} to {hi:.4g}"


def draw_slice_on_ax(
    ax,
    field2d: np.ndarray,
    data_dir: str,
    *,
    title: str,
    cmap: str = DEFAULT_PLOT_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    use_coords: bool = True,
    coord_mode: str = "centers",
    overlay_grid: bool = False,
    clear: bool = True,
    colorbar_label: str = "",
) -> Any:
    """Draw a 2-D field slice on an existing axes (for GUI playback/GIF)."""
    if clear:
        ax.clear()

    from .grid import draw_field_mesh, overlay_cell_outlines, plot_coords_for_field

    mode = coord_mode if use_coords else "index"
    coords = plot_coords_for_field(data_dir, field2d.shape, mode)

    plot_vmin, plot_vmax = _resolve_clim(field2d, vmin, vmax, symmetric=symmetric)
    cmap_obj = resolve_cmap(cmap)

    mesh = draw_field_mesh(
        ax, field2d, coords, cmap=cmap_obj, vmin=plot_vmin, vmax=plot_vmax,
    )
    if overlay_grid:
        overlay_cell_outlines(ax, data_dir, field2d.shape)

    ax.set_title(title)
    if coords.note:
        ax.text(
            0.02, 0.98, coords.note, transform=ax.transAxes, fontsize=8,
            color="0.4", verticalalignment="top",
        )
    if colorbar_label:
        ax.figure.colorbar(mesh, ax=ax, label=colorbar_label)
    ax.set_xlabel(coords.xlabel)
    ax.set_ylabel(coords.ylabel)
    ax.figure.tight_layout()
    return mesh


def plot_field(
    data_dir: str,
    prefix: str,
    iteration: int | None = None,
    *,
    level: int | None = None,
    cmap: str = DEFAULT_PLOT_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    use_coords: bool = True,
    ax: Any | None = None,
    show: bool = True,
    save: str | None = None,
) -> tuple[Any, Any]:
    """
    Plot a 2D slice of a MITgcm field.

    Returns (figure, axes).
    """
    if iteration is not None:
        field2d = io.read_level_slice(data_dir, prefix, iteration, level)
        its: list[int] = []
    else:
        arr, its, _meta = io.read_field(data_dir, prefix, iteration)
        field2d = pick_2d_slice(arr, level)

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    if title is None:
        if iteration is not None:
            title = f"{prefix} @ iter {iteration}"
        elif its:
            title = f"{prefix} @ iter {its[-1]}"
        else:
            title = prefix
        _, nz = level_axis(field2d.shape if field2d.ndim == 2 else io.field_info(data_dir, prefix).shape)
        if nz > 1 and level is not None:
            title += f" (level {level})"
        elif nz > 1:
            title += f" (level {nz // 2})"

    draw_slice_on_ax(
        ax,
        field2d,
        data_dir,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        use_coords=use_coords,
        clear=False,
        colorbar_label=prefix,
    )

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    elif not show and created_fig:
        plt.close(fig)

    return fig, ax


def plot_array(
    arr: np.ndarray,
    data_dir: str,
    *,
    title: str = "field",
    level: int | None = None,
    cmap: str = DEFAULT_DIFF_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = True,
    coord_mode: str = "centers",
    overlay_grid: bool = False,
    ax: Any | None = None,
    save: str | None = None,
    show: bool = True,
) -> tuple[Any, Any]:
    """Plot a precomputed array (e.g. DiD result) as a 2D slice."""
    field2d = pick_2d_slice(arr, level)

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    draw_slice_on_ax(
        ax,
        field2d,
        data_dir,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        symmetric=symmetric,
        coord_mode=coord_mode,
        overlay_grid=overlay_grid,
        clear=not created_fig,
        colorbar_label="value",
    )

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    elif not show and created_fig:
        plt.close(fig)

    return fig, ax


def plot_diff(
    data_dir: str,
    prefix: str,
    iteration_a: int,
    iteration_b: int,
    *,
    level: int | None = None,
    cmap: str = DEFAULT_DIFF_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = True,
    diff2d: np.ndarray | None = None,
    ax: Any | None = None,
    save: str | None = None,
    show: bool = True,
) -> tuple[Any, Any]:
    """Plot difference between two iterations (2-D slice)."""
    from .ops import diff_slice

    if diff2d is None:
        diff2d, _ = diff_slice(data_dir, prefix, iteration_a, iteration_b, level)

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    title = f"{prefix}: iter {iteration_a} - {iteration_b}"
    if level is not None:
        title += f" (level {level})"

    draw_slice_on_ax(
        ax,
        diff2d,
        data_dir,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        symmetric=symmetric,
        clear=not created_fig,
        colorbar_label="difference",
    )

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    elif not show and created_fig:
        plt.close(fig)

    return fig, ax
