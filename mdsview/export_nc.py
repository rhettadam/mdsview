"""Export MITgcm MDS fields to NetCDF."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from . import io
from .selection import parse_level_list, parse_region, resolve_iterations
from .slices import level_axis


def export_to_netcdf(
    data_dir: str,
    prefixes: list[str],
    output_path: str,
    *,
    iterations: str = "all",
    levels: str | None = None,
    region: str | None = None,
    rec: int | None = None,
    compress: bool = True,
) -> dict[str, Any]:
    """
    Write selected variables and snapshots to a NetCDF file.

    Horizontal coordinates use y/x for model grids (metres or index) and
    lat/lon only when XC/YC look like geographic degrees.
    """
    try:
        import netCDF4 as nc
    except ImportError as exc:
        raise ImportError(
            "NetCDF export requires netCDF4. Install with: pip install netCDF4"
        ) from exc

    data_dir = os.path.abspath(data_dir)
    output_path = os.path.abspath(output_path)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    shapes = {prefix: io.field_info(data_dir, prefix).shape for prefix in prefixes}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        detail = ", ".join(f"{name} {shape}" for name, shape in shapes.items())
        raise ValueError(f"All variables must share the same shape; got {detail}")

    region_t = parse_region(region)
    iters = resolve_iterations(data_dir, prefixes[0], iterations)
    read_kwargs: dict[str, Any] = {"rec": rec, "region": region_t, "squeeze": False}

    ref_shape = shapes[prefixes[0]]
    _, nz = level_axis(ref_shape)
    level_list = parse_level_list(levels, nz)

    sample = _read_snapshot(data_dir, prefixes[0], iters[0], level_list, read_kwargs)
    has_depth = sample.ndim == 3
    if has_depth:
        depth_len, y_len, x_len = sample.shape
    else:
        y_len, x_len = sample.shape
        depth_len = 0

    for prefix in prefixes[1:]:
        other = _read_snapshot(data_dir, prefix, iters[0], level_list, read_kwargs)
        if other.shape != sample.shape:
            raise ValueError(
                f"Snapshot shape mismatch for {prefix}: {other.shape} vs {sample.shape} "
                f"(from {prefixes[0]})"
            )
        if (other.ndim == 3) != has_depth:
            raise ValueError(f"Inconsistent vertical rank for {prefix}")

    horiz = _resolve_horizontal_coords(data_dir, region_t, y_len, x_len)
    y_dim = horiz.y_dim
    x_dim = horiz.x_dim
    data_dims = _variable_dims(has_depth, y_dim, x_dim)
    expected_data_shape = _expected_data_shape(
        len(iters), has_depth, depth_len, y_len, x_len
    )

    ds = nc.Dataset(output_path, "w", format="NETCDF4")
    try:
        ds.title = "mdsview export"
        ds.source = data_dir
        ds.history = f"Created by mdsview export from {data_dir}"
        ds.Conventions = "CF-1.8"

        ds.createDimension("time", len(iters))
        ds.createDimension(y_dim, y_len)
        ds.createDimension(x_dim, x_len)
        if has_depth:
            ds.createDimension("depth", depth_len)

        _write_time_coord(ds, iters)
        if has_depth:
            _write_depth_coord(ds, data_dir, level_list, depth_len, region_t)
        _write_horizontal_coords(ds, horiz)

        var_kwargs = {"zlib": True, "complevel": 4} if compress else {}
        exported: list[dict[str, Any]] = []
        coord_attr = _coordinates_attr(has_depth, y_dim, x_dim)

        for prefix in prefixes:
            stacks = [
                _read_snapshot(data_dir, prefix, itr, level_list, read_kwargs)
                for itr in iters
            ]
            data = np.stack(stacks, axis=0).astype(np.float32)
            if data.shape != expected_data_shape:
                raise ValueError(
                    f"Export shape mismatch for {prefix}: got {data.shape}, "
                    f"expected {expected_data_shape}"
                )
            var = ds.createVariable(prefix, "f4", data_dims, **var_kwargs)
            var[:] = data
            var.long_name = prefix
            var.source = f"{prefix} from {data_dir}"
            var.coordinates = coord_attr
            exported.append(
                {
                    "variable": prefix,
                    "shape": list(data.shape),
                    "dimensions": list(data_dims),
                }
            )

        return {
            "output": output_path,
            "data_dir": data_dir,
            "iterations": iters,
            "levels": level_list,
            "region": list(region_t) if region_t else None,
            "coordinates": {
                "geographic": bool(horiz.geographic),
                "y_dim": y_dim,
                "x_dim": x_dim,
                "has_xc_yc": horiz.xc_2d is not None,
            },
            "variables": exported,
        }
    finally:
        ds.close()


class _HorizontalCoords:
    __slots__ = (
        "geographic",
        "y_dim",
        "x_dim",
        "y_1d",
        "x_1d",
        "xc_2d",
        "yc_2d",
        "y_units",
        "x_units",
    )

    def __init__(
        self,
        *,
        geographic: bool,
        y_dim: str,
        x_dim: str,
        y_1d: np.ndarray,
        x_1d: np.ndarray,
        xc_2d: np.ndarray | None,
        yc_2d: np.ndarray | None,
        y_units: str,
        x_units: str,
    ) -> None:
        self.geographic = geographic
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.y_1d = y_1d
        self.x_1d = x_1d
        self.xc_2d = xc_2d
        self.yc_2d = yc_2d
        self.y_units = y_units
        self.x_units = x_units


def _expected_data_shape(
    n_time: int,
    has_depth: bool,
    depth_len: int,
    y_len: int,
    x_len: int,
) -> tuple[int, ...]:
    if has_depth:
        return (n_time, depth_len, y_len, x_len)
    return (n_time, y_len, x_len)


def _variable_dims(has_depth: bool, y_dim: str, x_dim: str) -> tuple[str, ...]:
    if has_depth:
        return ("time", "depth", y_dim, x_dim)
    return ("time", y_dim, x_dim)


def _coordinates_attr(has_depth: bool, y_dim: str, x_dim: str) -> str:
    if has_depth:
        return f"time depth {y_dim} {x_dim}"
    return f"time {y_dim} {x_dim}"


def _read_snapshot(
    data_dir: str,
    prefix: str,
    iteration: int,
    level_list: list[int] | None,
    read_kwargs: dict[str, Any],
) -> np.ndarray:
    shape = io.field_info(data_dir, prefix).shape
    level_axis_idx, nz = level_axis(shape)
    kwargs = dict(read_kwargs)

    if level_axis_idx is not None and nz > 1:
        if level_list is None:
            arr, _, _ = io.read_field(data_dir, prefix, iteration, **kwargs)
        elif len(level_list) == 1:
            kwargs["lev"] = level_list[0]
            arr, _, _ = io.read_field(data_dir, prefix, iteration, **kwargs)
            arr = np.squeeze(arr)
        else:
            kwargs["lev"] = level_list
            arr, _, _ = io.read_field(data_dir, prefix, iteration, **kwargs)
    else:
        arr, _, _ = io.read_field(data_dir, prefix, iteration, **kwargs)
        arr = np.squeeze(arr)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2-D or 3-D snapshot for {prefix}, got shape {arr.shape}")
    return arr


def _resolve_horizontal_coords(
    data_dir: str,
    region: tuple[int, int, int, int] | None,
    y_len: int,
    x_len: int,
) -> _HorizontalCoords:
    y_1d = np.arange(y_len, dtype=np.float64)
    x_1d = np.arange(x_len, dtype=np.float64)
    xc_2d: np.ndarray | None = None
    yc_2d: np.ndarray | None = None
    y_units = "1"
    x_units = "1"
    geographic = False

    prefixes = set(io.list_prefixes(data_dir))
    if "XC" in prefixes and "YC" in prefixes:
        try:
            xc, _, _ = io.read_field(data_dir, "XC", None, squeeze=True, region=region)
            yc, _, _ = io.read_field(data_dir, "YC", None, squeeze=True, region=region)
            xc = np.asarray(xc, dtype=np.float64)
            yc = np.asarray(yc, dtype=np.float64)
            if xc.shape == (y_len, x_len) and yc.shape == (y_len, x_len):
                x_1d = xc[0, :].copy()
                y_1d = yc[:, 0].copy()
                xc_2d = xc
                yc_2d = yc
                geographic = _looks_like_geographic(y_1d, x_1d)
                if geographic:
                    y_units = "degrees_north"
                    x_units = "degrees_east"
                else:
                    y_units = "m"
                    x_units = "m"
        except OSError:
            pass

    if geographic:
        y_dim, x_dim = "lat", "lon"
    else:
        y_dim, x_dim = "y", "x"

    return _HorizontalCoords(
        geographic=geographic,
        y_dim=y_dim,
        x_dim=x_dim,
        y_1d=y_1d,
        x_1d=x_1d,
        xc_2d=xc_2d,
        yc_2d=yc_2d,
        y_units=y_units,
        x_units=x_units,
    )


def _write_time_coord(ds: Any, iters: list[int]) -> None:
    var = ds.createVariable("time", "f8", ("time",))
    var[:] = np.asarray(iters, dtype=np.float64)
    var.long_name = "time"
    var.standard_name = "time"
    var.axis = "T"
    var.calendar = "standard"
    var.units = "seconds since 1970-01-01 00:00:00"
    var.comment = f"MITgcm iteration numbers: {', '.join(str(i) for i in iters)}"

    iter_var = ds.createVariable("iteration", "i4", ("time",))
    iter_var[:] = np.asarray(iters, dtype=np.int32)
    iter_var.long_name = "MITgcm iteration number"
    iter_var.comment = "Model time step index from the MDS filename"


def _write_depth_coord(
    ds: Any,
    data_dir: str,
    level_list: list[int] | None,
    depth_len: int,
    region: tuple[int, int, int, int] | None,
) -> None:
    depth = np.arange(depth_len, dtype=np.float64)
    if "RC" in io.list_prefixes(data_dir):
        try:
            rc, _, _ = io.read_field(data_dir, "RC", None, squeeze=False, region=region)
            rc = np.asarray(rc)
            if rc.ndim >= 3:
                rc_1d = rc.reshape(rc.shape[0], -1).mean(axis=1)
            else:
                rc_1d = np.ravel(rc)
            if level_list is not None:
                rc_1d = rc_1d[level_list]
            if rc_1d.size >= depth_len:
                depth = rc_1d[:depth_len].astype(np.float64)
        except OSError:
            pass

    var = ds.createVariable("depth", "f8", ("depth",))
    var[:] = depth
    var.long_name = "depth"
    var.standard_name = "depth"
    var.axis = "Z"
    var.units = "meters"
    var.positive = "down"


def _write_horizontal_coords(ds: Any, horiz: _HorizontalCoords) -> None:
    y_var = ds.createVariable(horiz.y_dim, "f8", (horiz.y_dim,))
    x_var = ds.createVariable(horiz.x_dim, "f8", (horiz.x_dim,))
    y_var[:] = horiz.y_1d
    x_var[:] = horiz.x_1d
    y_var.axis = "Y"
    x_var.axis = "X"
    y_var.units = horiz.y_units
    x_var.units = horiz.x_units

    if horiz.geographic:
        y_var.standard_name = "latitude"
        x_var.standard_name = "longitude"
        y_var.long_name = "latitude"
        x_var.long_name = "longitude"
    else:
        y_var.long_name = "y coordinate"
        x_var.long_name = "x coordinate"
        y_var.comment = "Model row coordinate at cell centres"
        x_var.comment = "Model column coordinate at cell centres"

    if horiz.xc_2d is not None and horiz.yc_2d is not None:
        xc_var = ds.createVariable("xc", "f8", (horiz.y_dim, horiz.x_dim))
        yc_var = ds.createVariable("yc", "f8", (horiz.y_dim, horiz.x_dim))
        xc_var[:] = horiz.xc_2d
        yc_var[:] = horiz.yc_2d
        xc_var.long_name = "x coordinate of grid cell centre"
        yc_var.long_name = "y coordinate of grid cell centre"
        xc_var.units = horiz.x_units
        yc_var.units = horiz.y_units
        if horiz.geographic:
            xc_var.standard_name = "longitude"
            yc_var.standard_name = "latitude"


def _looks_like_geographic(lat: np.ndarray, lon: np.ndarray) -> bool:
    finite_lat = lat[np.isfinite(lat)]
    finite_lon = lon[np.isfinite(lon)]
    if finite_lat.size == 0 or finite_lon.size == 0:
        return False
    return (
        np.nanmin(finite_lat) >= -90.0
        and np.nanmax(finite_lat) <= 90.0
        and np.nanmin(finite_lon) >= -360.0
        and np.nanmax(finite_lon) <= 360.0
    )
