import tempfile

import numpy as np
import pytest

from mdsview import io, ops
from mdsview.samples.generate import generate_sample_run


@pytest.fixture
def sample_dir():
    with tempfile.TemporaryDirectory(prefix="mdsview_test_") as tmp:
        generate_sample_run(tmp, preset="tiny", progress=False)
        yield tmp


def test_list_prefixes(sample_dir):
    prefixes = io.list_prefixes(sample_dir)
    assert "T" in prefixes
    assert "S" in prefixes
    assert "XC" in prefixes
    assert "RC" in prefixes


def test_field_info_meta_only(sample_dir):
    info = io.field_info(sample_dir, "T")
    assert info.shape == (12, 60, 80)
    assert len(info.iterations) == 5
    assert info.nbytes > 0


def test_read_write_roundtrip(sample_dir):
    arr, its, _ = io.read_field(sample_dir, "T", 0)
    assert arr.shape == (12, 60, 80)
    assert its == [0]


def test_diff_slice_2d_field(sample_dir):
    diff, meta = ops.diff_slice(sample_dir, "Eta", 240, 0, level=0)
    assert diff.shape == (60, 80)
    assert meta["operation"] == "subtract_slice"


def test_pick_2d_slice_clamps_stale_level():
    from mdsview import plotting

    arr3d = np.zeros((8, 10, 12))
    assert plotting._pick_2d_slice(arr3d, 20).shape == (10, 12)
    arr2d = np.zeros((10, 12))
    assert plotting._pick_2d_slice(arr2d, 20).shape == (10, 12)


def test_diff_fields(sample_dir):
    diff, meta = ops.diff_fields(sample_dir, "T", 240, 0)
    assert diff.shape == (12, 60, 80)
    assert meta["operation"] == "subtract"
    assert diff.mean() == pytest.approx(0.252, rel=0.02)


def test_diff_slice_cross_dir(sample_dir, tmp_path):
    import shutil

    other = tmp_path / "other_run"
    shutil.copytree(sample_dir, other)
    diff, meta = ops.diff_slice(sample_dir, "T", 240, 240, level=0, data_dir_b=str(other))
    assert diff.shape == (60, 80)
    assert meta["data_dir_a"] != meta["data_dir_b"]
    np.testing.assert_allclose(diff, 0.0, atol=1e-6)


def test_combine_iterations(sample_dir):
    stacked, iters, _ = ops.combine_iterations(sample_dir, "T", [0, 120, 240])
    assert stacked.shape[0] == 3
    assert iters == [0, 120, 240]
