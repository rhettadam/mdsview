import numpy as np
import pytest

from mdsview import grid
from mdsview.samples.generate import generate_sample_run


@pytest.fixture
def sample_dir(tmp_path):
    generate_sample_run(tmp_path, preset="tiny", progress=False)
    return tmp_path


def test_catalog_grid(sample_dir):
    cat = grid.catalog_grid(sample_dir)
    assert cat.has_centers
    assert cat.has_faces
    assert cat.xc_shape == (60, 80)
    assert cat.xg_shape == (61, 81)


def test_resolve_coord_mode_fallback():
    cat = grid.GridCatalog(prefixes={"XC", "YC"})
    assert grid.resolve_coord_mode("faces", cat) == "centers"
    assert grid.resolve_coord_mode("centers", cat) == "centers"
    assert grid.resolve_coord_mode("index", cat) == "index"


def test_plot_coords_index():
    cat = grid.GridCatalog(prefixes=set())
    coords = grid.plot_coords_for_field("/nonexistent", (60, 80), "index")
    assert coords.mode == "index"
    assert coords.x is None


def test_plot_coords_centers(sample_dir):
    coords = grid.plot_coords_for_field(sample_dir, (60, 80), "centers")
    assert coords.mode == "centers"
    assert coords.x is not None
    assert coords.x.shape == (60, 80)


def test_plot_coords_faces_corner_grid(sample_dir):
    coords = grid.plot_coords_for_field(sample_dir, (60, 80), "faces")
    assert coords.mode == "faces"
    assert coords.x is not None
    assert coords.x.shape == (61, 81)
    assert coords.shading == "flat"


def test_overlay_cell_outlines_runs(sample_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        grid.overlay_cell_outlines(ax, sample_dir, (60, 80))
    finally:
        plt.close(fig)


def test_plot_grid_preview_runs(sample_dir):
    fig, ax = grid.plot_grid_preview(sample_dir, mode="centers", show=False)
    plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
    plt.close(fig)
