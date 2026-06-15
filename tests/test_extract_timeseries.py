import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from mdsview import io
from mdsview.extract import extract_field
from mdsview.samples.generate import generate_sample_run
from mdsview.timeseries import compute_timeseries, plot_timeseries, write_timeseries_csv


@pytest.fixture
def sample_dir(tmp_path):
    generate_sample_run(tmp_path, preset="tiny", progress=False)
    return tmp_path


def test_extract_iterations_and_levels(sample_dir, tmp_path):
    out = tmp_path / "subset"
    result = extract_field(
        sample_dir,
        "T",
        str(out),
        iterations="0,240,480",
        levels="0:3",
    )
    assert result["snapshots_written"] == 3
    assert io.list_iterations(str(out), "T") == [0, 240, 480]
    arr, _, _ = io.read_field(str(out), "T", 240)
    assert arr.shape == (3, 60, 80)


def test_extract_region(sample_dir, tmp_path):
    out = tmp_path / "patch"
    extract_field(
        sample_dir,
        "T",
        str(out),
        iterations="480",
        levels="2",
        region="10,30,20,40",
    )
    arr, _, _ = io.read_field(str(out), "T", 480)
    assert arr.shape == (20, 20)


def test_timeseries_point(sample_dir):
    payload = compute_timeseries(
        sample_dir,
        "T",
        iterations="0,480",
        level=2,
        at="10,20",
    )
    assert payload["mode"] == "point"
    assert len(payload["series"]) == 2
    full = io.read_level_slice(sample_dir, "T", 0, 2)
    expected = float(full[20, 10])
    assert payload["series"][0]["value"] == pytest.approx(expected)


def test_timeseries_domain_mean(sample_dir):
    payload = compute_timeseries(sample_dir, "Eta", iterations="0,480")
    assert payload["mode"] == "domain"
    assert len(payload["series"]) == 2
    slab = io.read_level_slice(sample_dir, "Eta", 0, None)
    expected = float(np.nanmean(slab))
    assert payload["series"][0]["value"] == pytest.approx(expected)


def test_timeseries_box(sample_dir):
    payload = compute_timeseries(
        sample_dir,
        "T",
        iterations="480",
        level=0,
        box="0,10,0,10",
    )
    slab = io.read_level_slice(sample_dir, "T", 480, 0)
    expected = float(np.nanmean(slab[0:10, 0:10]))
    assert payload["series"][0]["value"] == pytest.approx(expected)


def test_write_timeseries_csv(sample_dir, tmp_path):
    payload = compute_timeseries(sample_dir, "T", iterations="0,480", level=0, at="5,5")
    path = tmp_path / "ts.csv"
    write_timeseries_csv(str(path), payload)
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["iteration", "value"]
    assert len(rows) == 3


def test_plot_timeseries(sample_dir, tmp_path):
    payload = compute_timeseries(sample_dir, "T", iterations="0,480", level=0, at="5,5")
    out = tmp_path / "ts.png"
    plot_timeseries(payload, save=str(out), show=False)
    assert out.is_file()
    assert out.stat().st_size > 0


def test_cli_extract(sample_dir, tmp_path):
    out = tmp_path / "cli_subset"
    result = subprocess.run(
        [
            sys.executable, "-m", "mdsview.cli", "extract",
            "-d", str(sample_dir),
            "-v", "S",
            "-o", str(out),
            "--iterations", "0,480",
            "--levels", "1",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["snapshots_written"] == 2


def test_cli_timeseries(sample_dir, tmp_path):
    out = tmp_path / "ts.csv"
    result = subprocess.run(
        [
            sys.executable, "-m", "mdsview.cli", "timeseries",
            "-d", str(sample_dir),
            "-v", "T",
            "-l", "0",
            "--iterations", "0,480",
            "-o", str(out),
            "--no-plot",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert out.is_file()


def test_cli_timeseries_save_figure(sample_dir, tmp_path):
    out = tmp_path / "ts.png"
    result = subprocess.run(
        [
            sys.executable, "-m", "mdsview.cli", "timeseries",
            "-d", str(sample_dir),
            "-v", "T",
            "-l", "0",
            "--iterations", "0,480",
            "--save-figure", str(out),
            "--no-show",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    assert out.stat().st_size > 0
