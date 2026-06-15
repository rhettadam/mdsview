"""MITgcm horizontal grid discovery, coordinates, and preview plotting."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from . import io
from .catalog import get_catalog

CoordMode = Literal["index", "centers", "faces"]

COORD_MODE_LABELS: dict[CoordMode, str] = {
    "index": "Array index (i, j)",
    "centers": "Cell centres (XC, YC)",
    "faces": "Face corners (XG, YG)",
}

GRID_HELP = """\
MITgcm C-grid fields are usually stored on tracer cell centres (same shape as T, S, …).

  XC, YC  — longitude/latitude or x/y at cell centres (ny × nx)
  XG, YG  — coordinates at cell corners (often ny+1 × nx+1) for pcolormesh edges
  RC      — vertical centre depths (1-D or 3-D)

Choose how the map axes are drawn:

  • Index — column/row numbers (no grid files required)
  • Centres — plot with XC/YC (typical for cell-centred fields)
  • Faces — plot with XG/YG corners (needs corner grids in the run folder)

Enable “Overlay grid lines” to sketch cell boundaries on field plots.\
"""


@dataclass
class GridCatalog:
  """Which grid variables exist in a run directory."""

  prefixes: set[str]
  xc_shape: tuple[int, ...] | None = None
  yc_shape: tuple[int, ...] | None = None
  xg_shape: tuple[int, ...] | None = None
  yg_shape: tuple[int, ...] | None = None
  rc_shape: tuple[int, ...] | None = None

  @property
  def has_centers(self) -> bool:
    return "XC" in self.prefixes and "YC" in self.prefixes

  @property
  def has_faces(self) -> bool:
    return "XG" in self.prefixes and "YG" in self.prefixes

  @property
  def has_rc(self) -> bool:
    return "RC" in self.prefixes

  def mode_available(self, mode: CoordMode) -> bool:
    if mode == "index":
      return True
    if mode == "centers":
      return self.has_centers
    if mode == "faces":
      return self.has_faces
    return False

  def summary_lines(self) -> list[str]:
    lines = ["Grid files in run folder:"]
    for name in ("XC", "YC", "XG", "YG", "RC", "RF"):
      if name in self.prefixes:
        shape = getattr(self, f"{name.lower()}_shape", None)
        lines.append(f"  {name:3s}  {shape}")
      else:
        lines.append(f"  {name:3s}  —")
    return lines


def catalog_grid(data_dir: str) -> GridCatalog:
  data_dir = os.path.abspath(data_dir)
  run = get_catalog(data_dir)
  prefixes = set(run.prefixes)
  cat = GridCatalog(prefixes=prefixes)

  def _shape(prefix: str) -> tuple[int, ...] | None:
    if prefix not in prefixes:
      return None
    try:
      return run.shape(prefix)
    except OSError:
      return None

  cat.xc_shape = _shape("XC")
  cat.yc_shape = _shape("YC")
  cat.xg_shape = _shape("XG")
  cat.yg_shape = _shape("YG")
  cat.rc_shape = _shape("RC")
  return cat


def _read_grid_2d(data_dir: str, prefix: str) -> np.ndarray | None:
  run = get_catalog(data_dir)
  if prefix not in run.prefixes:
    return None
  try:
    arr, _, _ = io.read_field(data_dir, prefix)
    return np.squeeze(arr).astype(np.float64)
  except OSError:
    return None


def resolve_coord_mode(mode: CoordMode, catalog: GridCatalog) -> CoordMode:
  """Fall back when requested coordinates are missing."""
  if catalog.mode_available(mode):
    return mode
  if mode == "faces" and catalog.has_centers:
    return "centers"
  return "index"


@dataclass
class PlotCoords:
  x: np.ndarray | None
  y: np.ndarray | None
  mode: CoordMode
  xlabel: str
  ylabel: str
  shading: str
  note: str = ""


def plot_coords_for_field(
  data_dir: str,
  field_shape: tuple[int, ...],
  mode: CoordMode,
) -> PlotCoords:
  """Resolve (x, y) arrays and axis labels for a 2-D field slice."""
  cat = catalog_grid(data_dir)
  mode = resolve_coord_mode(mode, cat)
  ny, nx = int(field_shape[-2]), int(field_shape[-1])

  if mode == "index":
    return PlotCoords(
      None, None, "index", "i (x index)", "j (y index)", "auto", ""
    )

  if mode == "centers":
    xc = _read_grid_2d(data_dir, "XC")
    yc = _read_grid_2d(data_dir, "YC")
    if xc is not None and yc is not None and xc.shape == (ny, nx) and yc.shape == (ny, nx):
      unit = _guess_unit(xc, yc)
      return PlotCoords(
        xc, yc, "centers", f"x ({unit})", f"y ({unit})", "auto", ""
      )
    return PlotCoords(
      None, None, "index", "i (x index)", "j (y index)", "auto",
      "XC/YC missing or shape mismatch — using index axes.",
    )

  xg = _read_grid_2d(data_dir, "XG")
  yg = _read_grid_2d(data_dir, "YG")
  if xg is not None and yg is not None and xg.shape == yg.shape:
    if xg.shape == (ny, nx):
      unit = _guess_unit(xg, yg)
      return PlotCoords(
        xg, yg, "faces", f"x ({unit})", f"y ({unit})", "auto",
        "XG/YG same shape as field — treated as centre coords.",
      )
    if xg.shape == (ny + 1, nx + 1):
      unit = _guess_unit(xg, yg)
      return PlotCoords(
        xg, yg, "faces", f"x ({unit})", f"y ({unit})", "flat", ""
      )
  if cat.has_centers:
    note = "XG/YG missing or incompatible — using XC/YC."
    sub = plot_coords_for_field(data_dir, field_shape, "centers")
    sub.note = note
    return sub
  return PlotCoords(
    None, None, "index", "i (x index)", "j (y index)", "auto",
    "Corner grids unavailable — using index axes.",
  )


def _guess_unit(xc: np.ndarray, yc: np.ndarray) -> str:
  span = max(float(np.nanmax(xc) - np.nanmin(xc)), float(np.nanmax(yc) - np.nanmin(yc)))
  if span > 1e5:
    return "m"
  if span > 180:
    return "deg"
  return "model units"


def draw_field_mesh(
  ax,
  field2d: np.ndarray,
  coords: PlotCoords,
  *,
  cmap,
  vmin: float | None,
  vmax: float | None,
) -> Any:
  if coords.x is not None and coords.y is not None:
    return ax.pcolormesh(
      coords.x, coords.y, field2d,
      cmap=cmap, vmin=vmin, vmax=vmax, shading=coords.shading,
    )
  return ax.imshow(
    field2d, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto",
  )


def overlay_cell_outlines(
  ax,
  data_dir: str,
  field_shape: tuple[int, ...],
  *,
  line_color: str | tuple = "k",
  line_alpha: float = 0.45,
  line_width: float = 0.35,
) -> None:
  """Sketch horizontal cell boundaries using the best available grid."""
  ny, nx = int(field_shape[-2]), int(field_shape[-1])
  xg = _read_grid_2d(data_dir, "XG")
  yg = _read_grid_2d(data_dir, "YG")
  if xg is not None and yg is not None and xg.shape == (ny + 1, nx + 1):
    for j in range(ny + 1):
      ax.plot(xg[j, :], yg[j, :], color=line_color, lw=line_width, alpha=line_alpha)
    for i in range(nx + 1):
      ax.plot(xg[:, i], yg[:, i], color=line_color, lw=line_width, alpha=line_alpha)
    return
  xc = _read_grid_2d(data_dir, "XC")
  yc = _read_grid_2d(data_dir, "YC")
  if xc is None or yc is None or xc.shape != (ny, nx):
    ax.grid(True, color="0.55", alpha=0.35, linewidth=0.4)
    return
  lw = max(line_width * 0.85, 0.25)
  for j in range(ny):
    ax.plot(xc[j, :], yc[j, :], color=line_color, lw=lw, alpha=line_alpha)
  for i in range(nx):
    ax.plot(xc[:, i], yc[:, i], color=line_color, lw=lw, alpha=line_alpha)


def plot_grid_preview(
  data_dir: str,
  *,
  mode: CoordMode = "centers",
  ax=None,
  show: bool = False,
) -> tuple[Any, Any]:
  """Demonstrate the horizontal grid in the map plane (unit field)."""
  cat = catalog_grid(data_dir)
  mode = resolve_coord_mode(mode, cat)

  if cat.xc_shape and len(cat.xc_shape) >= 2:
    ny, nx = int(cat.xc_shape[-2]), int(cat.xc_shape[-1])
  elif cat.xg_shape and len(cat.xg_shape) >= 2:
    sh = cat.xg_shape
    ny, nx = (int(sh[-2]) - 1, int(sh[-1]) - 1) if sh[-2] > 1 else (int(sh[-2]), int(sh[-1]))
  else:
    ny, nx = 64, 48

  demo = (np.add.outer(np.arange(ny), np.arange(nx)) % 2).astype(np.float64)
  coords = plot_coords_for_field(data_dir, demo.shape, mode)

  created = ax is None
  if ax is None:
    fig, ax = plt.subplots(figsize=(7, 5))
  else:
    fig = ax.figure

  ax.set_facecolor("#f4f5f7")
  draw_field_mesh(ax, demo, coords, cmap="Greys", vmin=0, vmax=1)
  overlay_cell_outlines(
    ax, data_dir, demo.shape,
    line_color="#2b6cb0", line_alpha=0.85, line_width=0.55,
  )
  ax.set_title(f"Grid preview ({COORD_MODE_LABELS.get(coords.mode, coords.mode)})")
  ax.set_xlabel(coords.xlabel)
  ax.set_ylabel(coords.ylabel)
  if coords.note:
    ax.text(
      0.02, 0.02, coords.note, transform=ax.transAxes, fontsize=8,
      color="0.35", verticalalignment="bottom",
    )
  ax.text(
    0.98, 0.02, "checkerboard demo", transform=ax.transAxes, fontsize=8,
    color="0.45", horizontalalignment="right", verticalalignment="bottom",
  )
  if created:
    fig.tight_layout()
  if show:
    plt.show()
  elif created:
    plt.close(fig)
  return fig, ax
