import time

import pytest

from mdsview import animation, catalog, io
from mdsview.samples.generate import generate_sample_run


@pytest.fixture
def many_dir(tmp_path):
    generate_sample_run(tmp_path, preset="many_files", variables=["T"], progress=False)
    return tmp_path


def test_catalog_single_scan(many_dir):
    cat = catalog.get_catalog(many_dir, refresh=True)
    assert "T" in cat.prefixes
    assert cat.iter_count("T") == 2000
    assert cat.iters_for("T")[0] == 0
    assert cat.iters_for("T")[-1] == 2000 * 12 - 12


def test_field_info_uses_catalog(many_dir):
    catalog.get_catalog(many_dir, refresh=True)
    info = io.field_info(many_dir, "T")
    assert len(info.iterations) == 2000


def test_resolve_playback_clim_subsamples(many_dir):
    iters = io.list_iterations(many_dir, "T")
    t0 = time.perf_counter()
    animation.resolve_playback_clim(
        many_dir, "T", iters, level=0,
        user_vmin=None, user_vmax=None, lock_scale=True, max_samples=32,
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0


def test_resolve_playback_clim_anchor_frame(tmp_path):
    from mdsview import plotting

    generate_sample_run(tmp_path, preset="tiny", progress=False)
    iters = io.list_iterations(tmp_path, "T")
    field2d = io.read_level_slice(tmp_path, "T", iters[0], 0)
    expected = plotting.resolved_clim(field2d, None, None)
    got = animation.resolve_playback_clim(
        tmp_path, "T", iters, level=0,
        user_vmin=None, user_vmax=None, lock_scale=True,
        anchor_iteration=iters[0],
    )
    assert got == expected
