"""Shared CLI help text and argument helpers."""

from __future__ import annotations

import argparse

from .colormaps import COLORMAP_CHOICES, DEFAULT_DIFF_CMAP, DEFAULT_PLOT_CMAP

EXAMPLES = """
Examples (run from your MITgcm output folder, or add -d /path/to/run):

  # List all variables (meta only — safe on large runs)
  mdsview info

  # Headless plot on a Linux server (no DISPLAY needed)
  mdsview plot -v T -i 0 -l 4 --save-figure t.png --no-show

  # Diff two times at one level (reads two slabs, not full volumes)
  mdsview diff -v T --later 480 --earlier 0 -l 4 --save-figure diff.png --no-show

  # Cross-run diff (later run -d, earlier --dir-b)
  mdsview diff -v T --later 2520 --earlier 2520 -l 20 -d /scratch/warm --dir-b /scratch/ref

  # Export MDS fields to NetCDF
  mdsview export -v T,S -o run.nc --iterations 0,360,720
  mdsview export -v T -o surface.nc --iterations all --levels 0

  # Subset to a smaller MDS folder (for transfer or sharing)
  mdsview extract -v T -o subset/ --iterations 0:1200:120 --levels 0:10

  # Time series at a point or domain mean
  mdsview timeseries -v T -l 4 --iterations all --at 40,30 -o point.csv
  mdsview timeseries -v Eta --iterations all --save-figure ssh.png --no-show
  mdsview timeseries -v T -l 0 --iterations 0:480:120 --json --no-plot

  # Open the GUI (requires display + pip install mdsview[gui])
  mdsview gui
"""


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Help formatter that preserves example newlines."""


def add_data_dir(parser: argparse.ArgumentParser, *, default: str = ".") -> None:
    parser.add_argument(
        "-d",
        "--dir",
        metavar="FOLDER",
        default=default,
        help="Folder containing .data and .meta files (default: current folder)",
    )


def add_variable(parser: argparse.ArgumentParser, *, required: bool = False, default=None) -> None:
    parser.add_argument(
        "-v",
        "--variable",
        "--prefix",
        "-p",
        dest="prefix",
        required=required,
        default=default,
        metavar="NAME",
        help="Variable name without extension, e.g. T, S, Eta, UVEL",
    )


def add_plot_options(group: argparse._ArgumentGroup, *, default_cmap: str = DEFAULT_PLOT_CMAP) -> None:
    group.add_argument(
        "-l",
        "--level",
        type=int,
        default=None,
        metavar="K",
        help="Vertical level index for 3-D fields (0 = top)",
    )
    group.add_argument(
        "--cmap",
        default=default_cmap,
        choices=COLORMAP_CHOICES,
        help=f"Matplotlib or cmocean colormap (default: {default_cmap})",
    )
    group.add_argument("--vmin", type=float, default=None, help="Colour scale minimum")
    group.add_argument("--vmax", type=float, default=None, help="Colour scale maximum")
    group.add_argument(
        "--save-figure",
        "--output",
        dest="output",
        metavar="FILE.png",
        help="Save plot to PNG instead of opening a window",
    )
    group.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window",
    )


def add_json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON on stdout",
    )


def add_iterations_arg(parser: argparse.ArgumentParser, *, default: str = "all") -> None:
    parser.add_argument(
        "--iterations",
        "--iters",
        dest="iterations",
        default=default,
        metavar="SPEC",
        help="Snapshots to export: all, last, comma list, or start:stop[:step]",
    )


def add_levels_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--levels",
        default=None,
        metavar="SPEC",
        help="Vertical levels to export: comma list or start:stop[:step] (default: all)",
    )


def add_region_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--region",
        default=None,
        metavar="I0,I1,J0,J1",
        help="Horizontal index bounds to export",
    )


def add_rec_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rec",
        type=int,
        default=None,
        metavar="K",
        help="Record index for multi-record files",
    )


def add_variables_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-v",
        "--variables",
        dest="variables",
        required=True,
        metavar="LIST",
        help="Comma-separated variable names, e.g. T,S,Eta",
    )
