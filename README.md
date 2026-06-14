# mdsview

Quick visualization and analysis of MITgcm MDS (`.data`/`.meta`) binary output.

## Feasibility

This is very feasible. MITgcm binary output is a well-documented paired format:

- **`.meta`** — ASCII header with dimensions, precision, iteration, record count
- **`.data`** — raw binary array (typically big-endian float32)

The official [`MITgcmutils`](https://github.com/MITgcm/MITgcm/tree/master/utils/python/MITgcmutils) package already handles tiled MPI output, meta parsing, and read/write. This tool wraps that for common workflows.

For large LLC or high-frequency output, consider [`xmitgcm`](https://xmitgcm.readthedocs.io/) (lazy xarray/dask loading). mdsview targets quick inspection of run directories.

## Install

```bash
pip install -e .
```

On Windows, if `mdsview` is not found, run via module instead:

```bash
python -m mdsview.cli info -d /path/to/run
```

## Sample data (no MITgcm run required)

Generate a synthetic run directory with realistic 3-D structure (thermocline,
halocline, etc.) and spatially varying DiD patterns for T vs S:

```bash
# Recommended: ~150 MB, 40 levels, visible DoD plots
python -m mdsview.cli generate-sample -o sample_data --preset demo

# Fast CI-sized set
python -m mdsview.cli generate-sample -o sample_tiny --preset tiny

# Larger sets if you have disk space
python -m mdsview.cli generate-sample -o sample_medium --preset medium
python -m mdsview.cli generate-sample -o sample_large --preset large
python -m mdsview.cli generate-sample -o sample_stress --preset stress

# MPI-style tiled 3D output
python -m mdsview.cli generate-sample -o sample_tiled --preset tiled

# Catalog stress test (~20,000 .data/.meta files; small grid, 2000 snapshots × 5 fields)
python -m mdsview.cli generate-sample -o sample_data_many --preset many_files

# Faster variant: one field, still thousands of files (~4,000 files at 2000 iters)
python -m mdsview.cli generate-sample -o sample_data_many --preset many_files --variables T --n-iters 2000

# Diff / cross-run testing (~40 MB pair: ref/ + warm/)
python -m mdsview.cli generate-sample -o sample_data_diff --preset diff_demo --diff-pair
```

Then try (see `sample_data/SAMPLE_README.txt` for recommended level & times):

```bash
python -m mdsview.cli info -d sample_data
python -m mdsview.cli plot -v T -i 2520 -l 20 -d sample_data --save-figure t.png --no-show
python -m mdsview.cli dod -a T -b S --time1 0 --time2 2520 -l 20 --plot --no-show --save-figure dod.png -d sample_data
python -m mdsview.cli gui -d sample_data
```

T and S use different spatial warming rates, so **DiD(T,S) varies in x and y** (not a flat field).

Presets:

| Preset | Grid (nx×ny×nz) | Iterations | ~Total size |
|--------|-----------------|------------|-------------|
| tiny | 80×60×12 | 5 | ~5 MB |
| **demo** | **240×160×40** | **8** | **~150 MB** |
| diff_demo | 128×96×20 | 6 | ~40 MB (use `--diff-pair` for ref/ + warm/) |
| small | 180×120×32 | 10 | ~100 MB |
| medium | 360×240×50 | 24 | ~1.5 GB |
| large | 900×600×50 | 12 | ~6 GB |
| stress | 1800×1200×50 | 6 | ~12 GB |
| tiled | 180×120×16 (3×2 tiles) | 8 | ~50 MB |
| many_files | 64×48×8 | 2000 | ~1 GB (20k files) |

## CLI cheat sheet

All commands accept `-d FOLDER` for the run directory (default: current folder).
Use `-v NAME` for the variable (T, S, Eta, …). Run `python -m mdsview.cli COMMAND --help` for full details and examples.

```bash
# Browse variables (reads .meta only)
python -m mdsview.cli info
python -m mdsview.cli info -v T
python -m mdsview.cli info -v T --show-meta

# Plot a snapshot
python -m mdsview.cli plot -v T -i 480 -l 4 --save-figure t.png --no-show

# Same variable, two times:  field(later) − field(earlier)
python -m mdsview.cli diff -v T --later 480 --earlier 0 --plot

# Two variables, two times (difference-of-differences)
python -m mdsview.cli dod -a T -b S --time1 0 --time2 480

# Graphical interface
python -m mdsview.cli gui
```

## What works today

| Feature | Status |
|---------|--------|
| Read tiled/global MDS output | Yes (via MITgcmutils) |
| Plot 2D slices with XC/YC | Yes |
| Diff two iterations | Yes |
| Combine (stack) iterations | Yes |
| Write diff/combined MDS | Yes |
| Interactive GUI | Yes (tkinter + matplotlib) |
| 3D volume rendering | Not yet |
| NetCDF / MNC tiled glue | Not yet (use MITgcm `gluemnc`) |
| LLC face unfolding | Partial (coords if XC/YC exist) |

## Server / batch use (Linux, HPC)

Install without the GUI (smaller footprint):

```bash
pip install -e .
# optional GUI later: pip install -e ".[gui]"
```

CLI defaults to the **Agg** matplotlib backend (no display). Use `--save-figure` and `--no-show` for plots:

```bash
export MPLBACKEND=Agg   # optional; set automatically for non-gui commands
mdsview info -d /path/to/run
mdsview plot -v T -i 0 -l 20 -d /path/to/run --save-figure t.png --no-show
mdsview diff -v T --later 480 --earlier 0 -l 20 --save-figure diff.png --no-show
mdsview dod -a T -b S --time1 0 --time2 2520   # stats only, level-by-level I/O
```

Memory notes:

- `info` and catalog scans read **`.meta` filenames only** (safe with thousands of snapshots).
- `plot` and `diff` at a single `--level` read **one horizontal slab** per snapshot, not the full 3-D volume.
- `dod` computes level-by-level; full 3-D output uses a **memmap temp file**, not four full arrays in RAM.
- `combine` loads every listed iteration — intended for small tests only.

Errors print to stderr with exit code 1; set `MDSVIEW_DEBUG=1` to see tracebacks.

## Limitations

- Assumes standard MDS output, not `pkg/mnc` NetCDF tiles (use `gluemnc` first).
- Multi-record diagnostics: `dod --rec`; extend other subcommands as needed.
- `combine` and full-volume `diff --save-field` (no `--level`) load entire snapshots into memory.
