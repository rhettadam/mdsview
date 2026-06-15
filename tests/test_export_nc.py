import json
import subprocess
import sys

import numpy as np
import pytest

from mdsview.export_nc import export_to_netcdf
from mdsview.samples.generate import generate_sample_run
from mdsview.selection import parse_int_list, parse_level_list, parse_prefix_list, resolve_iterations


@pytest.fixture
def sample_dir(tmp_path):
    generate_sample_run(tmp_path, preset="tiny", progress=False)
    return tmp_path


def test_parse_int_list():
    assert parse_int_list("0,120,240") == [0, 120, 240]
    assert parse_int_list("0:360:120") == [0, 120, 240]


def test_resolve_iterations_all(sample_dir):
    iters = resolve_iterations(sample_dir, "T", "all")
    assert iters == [0, 120, 240, 360, 480]


def test_resolve_iterations_last(sample_dir):
    assert resolve_iterations(sample_dir, "T", "last") == [480]


def test_parse_level_list(sample_dir):
    assert parse_level_list(None, 12) is None
    assert parse_level_list("0,4,8", 12) == [0, 4, 8]


def test_parse_prefix_list():
    assert parse_prefix_list("T, S ,Eta") == ["T", "S", "Eta"]


def test_export_to_netcdf(sample_dir, tmp_path):
    nc_path = tmp_path / "tiny.nc"
    result = export_to_netcdf(
        sample_dir,
        ["T", "S"],
        str(nc_path),
        iterations="0,240,480",
        levels="0:4",
    )
    assert nc_path.is_file()
    assert result["iterations"] == [0, 240, 480]
    assert len(result["variables"]) == 2

    import netCDF4 as nc

    with nc.Dataset(nc_path) as ds:
        assert ds.variables["T"].dimensions == ("time", "depth", "y", "x")
        assert ds.variables["S"].dimensions == ("time", "depth", "y", "x")
        assert ds.variables["T"].shape == ds.variables["S"].shape == (3, 4, 60, 80)
        assert "y" in ds.variables
        assert "x" in ds.variables
        assert "xc" in ds.variables
        assert "yc" in ds.variables
        assert "depth" in ds.dimensions
        assert ds.variables["T"].coordinates == "time depth y x"
        assert ds.variables["time"].units.startswith("seconds since")
        assert np.nanstd(ds.variables["T"][0, 0]) > 0
        np.testing.assert_array_equal(ds.variables["iteration"][:], [0, 240, 480])


def test_export_surface_only(sample_dir, tmp_path):
    nc_path = tmp_path / "eta.nc"
    export_to_netcdf(
        sample_dir,
        ["Eta"],
        str(nc_path),
        iterations="last",
    )
    import netCDF4 as nc

    with nc.Dataset(nc_path) as ds:
        assert ds.variables["Eta"].dimensions == ("time", "y", "x")
        assert ds.variables["Eta"].shape == (1, 60, 80)
        assert "depth" not in ds.dimensions
        assert ds.variables["Eta"].coordinates == "time y x"


def test_cli_export(sample_dir, tmp_path):
    nc_path = tmp_path / "cli.nc"
    result = subprocess.run(
        [
            sys.executable, "-m", "mdsview.cli", "export",
            "-d", str(sample_dir),
            "-v", "T",
            "-o", str(nc_path),
            "--iterations", "0,480",
            "--levels", "2",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["iterations"] == [0, 480]
    assert nc_path.is_file()
