import subprocess
import sys

import numpy as np
import pytest

from mdsview import io
from mdsview.slices import level_axis, mitgcm_lev_kw, pick_2d_slice
from mdsview.samples.generate import generate_sample_run


@pytest.fixture
def sample_dir(tmp_path):
    generate_sample_run(tmp_path, preset="tiny", progress=False)
    return tmp_path


def test_pick_2d_slice_clamps():
    arr3d = np.zeros((8, 10, 12))
    assert pick_2d_slice(arr3d, 20).shape == (10, 12)
    assert pick_2d_slice(np.zeros((10, 12)), 20).shape == (10, 12)


def test_mitgcm_lev_kw_3d():
    kw = mitgcm_lev_kw((12, 60, 80), 5)
    assert kw == {"lev": 5, "squeeze": False}


def test_mitgcm_lev_kw_2d():
    assert mitgcm_lev_kw((60, 80), 0) == {}


def test_read_level_slice_reads_slab(sample_dir):
    full, _, _ = io.read_field(sample_dir, "T", 0)
    slab = io.read_level_slice(sample_dir, "T", 0, 4)
    assert slab.shape == pick_2d_slice(full, 4).shape
    assert slab.shape == (60, 80)


def test_level_axis():
    assert level_axis((12, 60, 80)) == (0, 12)
    assert level_axis((60, 80)) == (None, 1)


def test_cli_info(sample_dir):
    result = subprocess.run(
        [sys.executable, "-m", "mdsview.cli", "info", "-d", str(sample_dir), "-v", "T"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "T" in result.stdout


def test_cli_plot_headless(sample_dir, tmp_path):
    out = tmp_path / "t.png"
    result = subprocess.run(
        [
            sys.executable, "-m", "mdsview.cli", "plot",
            "-d", str(sample_dir), "-v", "T", "-i", "0", "-l", "2",
            "--save-figure", str(out), "--no-show",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert out.is_file()


def test_cli_bad_iteration(sample_dir):
    result = subprocess.run(
        [
            sys.executable, "-m", "mdsview.cli", "plot",
            "-d", str(sample_dir), "-v", "T", "-i", "999999", "-l", "0", "--no-show",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "999999" in result.stderr


def test_cli_diff_cross_dir(sample_dir, tmp_path):
    import shutil

    other = tmp_path / "other_run"
    shutil.copytree(sample_dir, other)
    result = subprocess.run(
        [
            sys.executable, "-m", "mdsview.cli", "diff",
            "-d", str(sample_dir), "--dir-b", str(other),
            "-v", "T", "--later", "240", "--earlier", "240", "-l", "0", "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "@other_run" in result.stdout
    assert '"operation": "subtract_slice"' in result.stdout
