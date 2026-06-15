"""Point, box, and domain time series from MDS fields."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from . import io
from .errors import MdsViewError
from .selection import parse_index_pair, parse_region, resolve_iterations
from .slices import level_axis, pick_2d_slice

SeriesMode = Literal["point", "box", "domain"]


def compute_timeseries(
    data_dir: str,
    prefix: str,
    *,
    iterations: str = "all",
    level: int | None = None,
    at: str | None = None,
    box: str | None = None,
    rec: int | None = None,
) -> dict[str, Any]:
    """
    Build a time series by reading one 2-D slab per iteration.

    Modes (mutually exclusive):
      - --at I,J     value at one grid point
      - --box I0,I1,J0,J1   mean over a sub-domain
      - (default)    domain mean over the slice
    """
    if at and box:
        raise MdsViewError("Use only one of --at or --box", exit_code=2)

    data_dir = os.path.abspath(data_dir)
    iters = resolve_iterations(data_dir, prefix, iterations)
    shape = io.field_info(data_dir, prefix).shape
    _, nz = level_axis(shape)
    if nz > 1 and level is None:
        level = nz // 2

    point: tuple[int, int] | None = None
    box_region: tuple[int, int, int, int] | None = None

    if at:
        mode: SeriesMode = "point"
        point = parse_index_pair(at, name="--at")
        location: dict[str, Any] = {"i": point[0], "j": point[1]}
    elif box:
        mode = "box"
        box_region = parse_region(box)
        location = {"box": list(box_region)}
    else:
        mode = "domain"
        location = {}

    series: list[dict[str, Any]] = []
    for itr in iters:
        slab = io.read_level_slice(data_dir, prefix, itr, level, rec=rec)
        value = _reduce_slab(slab, mode=mode, at=point, box=box_region)
        series.append({"iteration": itr, "value": value})

    return {
        "data_dir": data_dir,
        "variable": prefix,
        "level": level if nz > 1 else None,
        "mode": mode,
        "location": location,
        "series": series,
    }


def _reduce_slab(
    slab: np.ndarray,
    *,
    mode: SeriesMode,
    at: tuple[int, int] | None,
    box: tuple[int, int, int, int] | None,
) -> float:
    field = pick_2d_slice(np.asarray(slab), None)
    if mode == "point":
        i, j = at or (0, 0)
        try:
            sample = field[j, i]
        except IndexError as exc:
            raise MdsViewError(
                f"Point ({i},{j}) outside field shape {field.shape} (y, x)"
            ) from exc
        if not np.isfinite(sample):
            return float("nan")
        return float(sample)

    if mode == "box":
        i0, i1, j0, j1 = box or (0, 0, 0, 0)
        patch = field[j0:j1, i0:i1]
        if patch.size == 0:
            raise MdsViewError(f"Empty box [{i0}:{i1}, {j0}:{j1}] for shape {field.shape}")
        return float(np.nanmean(patch))

    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def write_timeseries_csv(path: str, payload: dict[str, Any]) -> None:
    path = os.path.abspath(path)
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "value"])
        for row in payload["series"]:
            writer.writerow([row["iteration"], row["value"]])


def write_timeseries_json(path: str, payload: dict[str, Any]) -> None:
    path = os.path.abspath(path)
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def format_timeseries_title(payload: dict[str, Any]) -> str:
    parts = [payload["variable"]]
    level = payload.get("level")
    if level is not None:
        parts.append(f"level {level}")
    mode = payload["mode"]
    location = payload.get("location") or {}
    if mode == "point":
        parts.append(f"@ ({location['i']},{location['j']})")
    elif mode == "box":
        i0, i1, j0, j1 = location["box"]
        parts.append(f"box mean [{i0}:{i1}, {j0}:{j1}]")
    else:
        parts.append("domain mean")
    return " · ".join(parts)


def plot_timeseries(
    payload: dict[str, Any],
    *,
    save: str | None = None,
    show: bool = True,
) -> tuple[Any, Any]:
    """Plot iteration vs value as a line chart. Returns (figure, axes)."""
    series = payload["series"]
    if not series:
        raise MdsViewError("No time series points to plot")

    iterations = [int(row["iteration"]) for row in series]
    values = [float(row["value"]) for row in series]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iterations, values, marker="o", markersize=3, linewidth=1.2)
    ax.set_xlabel("iteration")
    ax.set_ylabel(payload["variable"])
    ax.set_title(format_timeseries_title(payload))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    elif not show:
        plt.close(fig)
    return fig, ax
