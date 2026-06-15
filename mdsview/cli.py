"""Command-line interface for mdsview."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import __version__, io, ops, plotting
from .cli_help import (
    EXAMPLES,
    HelpFormatter,
    add_data_dir,
    add_iterations_arg,
    add_json_flag,
    add_levels_arg,
    add_plot_options,
    add_rec_arg,
    add_region_arg,
    add_variable,
    add_variables_arg,
)
from .colormaps import DEFAULT_DIFF_CMAP
from .errors import MdsViewError
from .validate import abs_data_dir, require_iteration, require_prefix, resolve_iteration


def _configure_matplotlib(argv: list[str]) -> None:
    """Use Agg for headless CLI; TkAgg only for the GUI subcommand."""
    if "gui" in argv:
        os.environ.setdefault("MPLBACKEND", "TkAgg")
        import matplotlib

        matplotlib.use("TkAgg")
    elif not os.environ.get("MPLBACKEND"):
        os.environ["MPLBACKEND"] = "Agg"


def _should_show(args: argparse.Namespace) -> bool:
    if getattr(args, "no_show", False) or getattr(args, "output", None):
        return False
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        return False
    return True


def _timeseries_should_show(args: argparse.Namespace) -> bool:
    if getattr(args, "no_show", False) or getattr(args, "save_figure", None):
        return False
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        return False
    return True


def _diff_dir_tags(later_dir: str, earlier_dir: str) -> tuple[str, str]:
    """Return @folder suffixes for plot labels when the two runs differ."""
    if os.path.abspath(later_dir) == os.path.abspath(earlier_dir):
        return "", ""

    def tag(path: str) -> str:
        name = os.path.basename(os.path.abspath(path)) or os.path.abspath(path)
        return f"@{name}"

    return tag(later_dir), tag(earlier_dir)


def cmd_info(args: argparse.Namespace) -> int:
    data_dir = abs_data_dir(args.dir)
    if args.prefix == "ALL":
        prefixes = io.list_prefixes(data_dir)
        if not prefixes:
            raise MdsViewError(f"No MDS fields found in {data_dir}")
        for prefix in prefixes:
            info = io.field_info(data_dir, prefix)
            print(f"{info.prefix:8s}  shape={info.shape}  ~{info.nbytes_human}/snap  iters={len(info.iterations)}")
        return 0

    require_prefix(data_dir, args.prefix)
    info = io.field_info(data_dir, args.prefix)
    summary = io.read_field_meta(data_dir, args.prefix, info.iterations[-1] if info.iterations else None)

    if args.json:
        payload = {
            "directory": data_dir,
            "variable": info.prefix,
            "shape": info.shape,
            "dtype": info.dtype,
            "bytes_per_snapshot": info.nbytes,
            "bytes_human": info.nbytes_human,
            "iterations": info.iterations,
            "meta_file": summary.meta_path,
            "grid_dims": summary.gdims,
            "nrecords": summary.nrecords,
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Directory : {data_dir}")
    print(f"Variable  : {info.prefix}")
    print(f"Shape     : {info.shape}  (Z, Y, X for 3-D fields)")
    print(f"Precision : {info.dtype}")
    print(f"Size      : ~{info.nbytes_human} per snapshot")
    print(f"Meta file : {summary.meta_path}")
    print(f"Records   : {summary.nrecords}")
    print(f"Iterations ({len(info.iterations)}):")
    if info.iterations:
        preview = info.iterations[:12]
        suffix = " ..." if len(info.iterations) > 12 else ""
        print(f"  {preview}{suffix}")
    else:
        print("  (none — static grid file)")

    if args.show_meta:
        print("\n--- raw .meta ---")
        with open(summary.meta_path, encoding="utf-8", errors="replace") as f:
            print(f.read().rstrip())
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    data_dir = abs_data_dir(args.dir)
    require_prefix(data_dir, args.prefix)
    iteration = resolve_iteration(data_dir, args.prefix, args.iter)

    plotting.plot_field(
        data_dir,
        args.prefix,
        iteration,
        level=args.level,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        use_coords=not args.no_coords,
        save=args.output,
        show=_should_show(args),
    )
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    data_dir = abs_data_dir(args.dir)
    data_dir_b = abs_data_dir(args.dir_b) if args.dir_b else None
    require_prefix(data_dir, args.prefix)
    if data_dir_b:
        require_prefix(data_dir_b, args.prefix)

    later = args.later if args.later is not None else args.iter_a
    earlier = args.earlier if args.earlier is not None else args.iter_b
    if later is None or earlier is None:
        raise MdsViewError("Provide both --later and --earlier (or two positional iteration numbers).", exit_code=2)

    require_iteration(data_dir, args.prefix, later)
    earlier_dir = data_dir_b or data_dir
    require_iteration(earlier_dir, args.prefix, earlier)

    shape = io.field_info(data_dir, args.prefix).shape
    if data_dir_b:
        shape_b = io.field_info(data_dir_b, args.prefix).shape
        if shape != shape_b:
            raise MdsViewError(
                f"Shape mismatch for {args.prefix}: {shape} in later run vs {shape_b} in earlier run"
            )
    _, nz = ops.level_axis(shape)
    want_full_volume = args.save_field and args.level is None and nz > 1

    if want_full_volume:
        diff, meta = ops.diff_fields(
            data_dir, args.prefix, later, earlier, data_dir_b=data_dir_b
        )
    else:
        diff, meta = ops.diff_slice(
            data_dir, args.prefix, later, earlier, level=args.level, data_dir_b=data_dir_b
        )

    later_tag, earlier_tag = _diff_dir_tags(data_dir, earlier_dir)
    formula = f"{args.prefix}({later}{later_tag}) - {args.prefix}({earlier}{earlier_tag})"

    summary = ops.stats(np.asarray(diff))
    safe_meta = {k: v for k, v in meta.items() if k != "source_meta"}
    if args.json:
        print(
            json.dumps(
                {
                    "formula": formula,
                    "meta": safe_meta,
                    "stats": summary,
                },
                indent=2,
            )
        )
    else:
        print(formula)
        for key, value in summary.items():
            print(f"  {key}: {value:.6g}")

    if args.save_field:
        out_base = args.save_field
        if out_base.endswith(".data"):
            out_base = out_base[:-5]
        io.write_field(out_base, diff, iteration=later)
        print(f"Wrote {out_base}.data / {out_base}.meta")

    if args.plot or args.output:
        from .slices import pick_2d_slice

        diff2d = diff if diff.ndim == 2 else pick_2d_slice(diff, args.level)
        plotting.plot_diff(
            data_dir,
            args.prefix,
            later,
            earlier,
            level=args.level,
            data_dir_b=data_dir_b,
            later_tag=later_tag,
            earlier_tag=earlier_tag,
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            symmetric=True,
            diff2d=diff2d,
            save=args.output,
            show=_should_show(args),
        )
    return 0


def cmd_combine(args: argparse.Namespace) -> int:
    data_dir = abs_data_dir(args.dir)
    require_prefix(data_dir, args.prefix)
    iters = [int(x.strip()) for x in args.iterations.split(",") if x.strip()]
    if not iters:
        raise MdsViewError("Provide at least one iteration in --iterations")
    for itr in iters:
        require_iteration(data_dir, args.prefix, itr)

    stacked, used_iters, meta = ops.combine_iterations(data_dir, args.prefix, iters)
    print(f"Stacked {len(used_iters)} snapshots -> shape {stacked.shape}")
    if args.json:
        print(json.dumps({"iterations": used_iters, "shape": list(stacked.shape), "meta": meta}, indent=2))

    if args.save_field:
        out_base = args.save_field
        if out_base.endswith(".data"):
            out_base = out_base[:-5]
        io.write_field(out_base, stacked, iteration=used_iters[0] if used_iters else None)
        print(f"Wrote {out_base}.data / {out_base}.meta")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    from .extract import extract_field

    data_dir = abs_data_dir(args.dir)
    require_prefix(data_dir, args.prefix)
    result = extract_field(
        data_dir,
        args.prefix,
        args.output,
        iterations=args.iterations,
        levels=args.levels,
        region=args.region,
        rec=args.rec,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"Wrote {result['snapshots_written']} snapshots of {args.prefix} "
            f"to {result['output_dir']}"
        )
    return 0


def cmd_timeseries(args: argparse.Namespace) -> int:
    from .timeseries import (
        compute_timeseries,
        plot_timeseries,
        write_timeseries_csv,
        write_timeseries_json,
    )

    data_dir = abs_data_dir(args.dir)
    require_prefix(data_dir, args.prefix)
    payload = compute_timeseries(
        data_dir,
        args.prefix,
        iterations=args.iterations,
        level=args.level,
        at=args.at,
        box=args.box,
        rec=args.rec,
    )

    if args.output:
        if args.output.lower().endswith(".json"):
            write_timeseries_json(args.output, payload)
        else:
            write_timeseries_csv(args.output, payload)

    if not args.no_plot:
        plot_timeseries(
            payload,
            save=args.save_figure,
            show=_timeseries_should_show(args),
        )

    if args.json:
        print(json.dumps(payload, indent=2))
    elif args.output:
        print(f"Wrote {os.path.abspath(args.output)} ({len(payload['series'])} points)")
    elif args.no_plot:
        print(json.dumps(payload, indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .export_nc import export_to_netcdf
    from .selection import parse_prefix_list

    data_dir = abs_data_dir(args.dir)
    prefixes = parse_prefix_list(args.variables)
    for prefix in prefixes:
        require_prefix(data_dir, prefix)

    try:
        result = export_to_netcdf(
            data_dir,
            prefixes,
            args.output,
            iterations=args.iterations,
            levels=args.levels,
            region=args.region,
            rec=args.rec,
            compress=not args.no_compress,
        )
    except ImportError as exc:
        raise MdsViewError(str(exc)) from exc

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        names = ", ".join(v["variable"] for v in result["variables"])
        print(
            f"Wrote {result['output']}  ({names}, "
            f"{len(result['iterations'])} snapshots)"
        )
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    try:
        from .gui import launch_gui
    except ImportError as exc:
        raise MdsViewError(
            "GUI requires customtkinter. Install with: pip install mdsview[gui]"
        ) from exc

    launch_gui(initial_dir=abs_data_dir(args.dir))
    return 0


def _parse_variables(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    return tuple(v.strip() for v in value.split(",") if v.strip())


def cmd_generate_sample(args: argparse.Namespace) -> int:
    from .samples.generate import PRESETS, generate_diff_pair, generate_sample_run

    if args.preset != "custom" and args.preset not in PRESETS:
        raise MdsViewError(f"Unknown preset {args.preset!r}. Choose: {', '.join(PRESETS)}")

    out_dir = os.path.abspath(args.output)

    if args.diff_pair:
        if args.preset not in ("diff_demo", "custom"):
            raise MdsViewError("--diff-pair requires --preset diff_demo")
        generate_diff_pair(out_dir, seed=args.seed, progress=not args.quiet)
        return 0

    info = generate_sample_run(
        out_dir,
        preset=args.preset,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        n_iters=args.n_iters,
        iter_start=args.iter_start,
        iter_step=args.iter_step,
        variables=_parse_variables(args.variables),
        tiled=args.tiled,
        ntx=args.ntx,
        nty=args.nty,
        seed=args.seed,
        progress=not args.quiet,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "output_dir": info.output_dir,
                    "nx": info.nx,
                    "ny": info.ny,
                    "nz": info.nz,
                    "iterations": info.iterations,
                    "variables": info.variables,
                    "tiled": info.tiled,
                    "nbytes_total": info.nbytes_total,
                    "nbytes_human": info.nbytes_human,
                },
                indent=2,
            )
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mdsview",
        description="Inspect, plot, and analyse MITgcm binary (.data/.meta) output.",
        formatter_class=HelpFormatter,
        epilog=EXAMPLES,
    )
    parser.add_argument("--version", action="version", version=f"mdsview {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p_info = sub.add_parser(
        "info",
        help="List variables or show metadata for one variable",
        formatter_class=HelpFormatter,
        description="Read .meta files only — never loads the binary .data.",
    )
    add_data_dir(p_info)
    add_variable(p_info, default="ALL")
    add_json_flag(p_info)
    p_info.add_argument("--show-meta", action="store_true", help="Also print the raw .meta file contents")
    p_info.set_defaults(func=cmd_info)

    p_plot = sub.add_parser(
        "plot",
        help="Draw a 2-D slice of one variable",
        formatter_class=HelpFormatter,
        description="For 3-D fields, use --level to pick a horizontal slice. Reads one slab from disk.",
    )
    add_data_dir(p_plot)
    add_variable(p_plot, required=True)
    p_plot.add_argument(
        "-i", "--iteration", "--iter", dest="iter", default=None, metavar="N",
        help="Snapshot iteration number, or 'last' (omit for grid files like XC)",
    )
    plot_grp = p_plot.add_argument_group("Display")
    add_plot_options(plot_grp)
    p_plot.add_argument("--no-coords", action="store_true", help="Plot array indices instead of XC/YC coordinates")
    p_plot.set_defaults(func=cmd_plot)

    p_diff = sub.add_parser(
        "diff",
        help="Subtract two snapshots of the same variable",
        formatter_class=HelpFormatter,
        description=(
            "Computes field(LATER) - field(EARLIER). "
            "Use -d for the later run and --dir-b for a different earlier run. "
            "Default: one 2-D slice (memory-efficient)."
        ),
    )
    add_data_dir(p_diff)
    p_diff.add_argument(
        "--dir-b",
        "--earlier-dir",
        dest="dir_b",
        default=None,
        metavar="FOLDER",
        help="Run directory for the earlier snapshot (default: same as -d)",
    )
    add_variable(p_diff, required=True)
    req = p_diff.add_argument_group("Iterations (required)")
    req.add_argument("--later", type=int, metavar="N", help="Later iteration (minuend)")
    req.add_argument("--earlier", type=int, metavar="N", help="Earlier iteration (subtrahend)")
    p_diff.add_argument("iter_a", nargs="?", type=int, help=argparse.SUPPRESS)
    p_diff.add_argument("iter_b", nargs="?", type=int, help=argparse.SUPPRESS)
    out = p_diff.add_argument_group("Output")
    out.add_argument("--save-field", metavar="PREFIX", help="Write result as new .data/.meta files")
    out.add_argument("--plot", action="store_true", help="Show a difference plot")
    add_plot_options(out, default_cmap=DEFAULT_DIFF_CMAP)
    add_json_flag(p_diff)
    p_diff.set_defaults(func=cmd_diff)

    p_combine = sub.add_parser(
        "combine",
        help="Stack snapshots into one multi-time array",
        formatter_class=HelpFormatter,
        description="Loads every listed iteration into memory — use only for small tests.",
    )
    add_data_dir(p_combine)
    add_variable(p_combine, required=True)
    p_combine.add_argument(
        "--iterations", "--iters", dest="iterations", required=True, metavar="LIST",
        help="Comma-separated iteration numbers, e.g. 0,120,240",
    )
    p_combine.add_argument("--save-field", metavar="PREFIX", help="Write stacked array to .data/.meta")
    add_json_flag(p_combine)
    p_combine.set_defaults(func=cmd_combine)

    p_export = sub.add_parser(
        "export",
        help="Export MDS fields to NetCDF",
        formatter_class=HelpFormatter,
        description=(
            "Write selected variables and snapshots to a .nc file with labelled "
            "time/depth/y/x dimensions and XC/YC/RC coordinates when available."
        ),
    )
    add_data_dir(p_export)
    add_variables_arg(p_export)
    p_export.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="FILE.nc",
        help="Output NetCDF path",
    )
    add_iterations_arg(p_export)
    add_levels_arg(p_export)
    add_region_arg(p_export)
    add_rec_arg(p_export)
    p_export.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable zlib compression on data variables",
    )
    add_json_flag(p_export)
    p_export.set_defaults(func=cmd_export)

    p_extract = sub.add_parser(
        "extract",
        help="Write a subset of snapshots to new MDS files",
        formatter_class=HelpFormatter,
        description=(
            "Copy selected iterations, levels, and/or horizontal region to a new folder "
            "as .data/.meta pairs. Reads one snapshot at a time."
        ),
    )
    add_data_dir(p_extract)
    add_variable(p_extract, required=True)
    p_extract.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="DIR",
        help="Output directory for subset MDS files",
    )
    add_iterations_arg(p_extract)
    add_levels_arg(p_extract)
    add_region_arg(p_extract)
    add_rec_arg(p_extract)
    add_json_flag(p_extract)
    p_extract.set_defaults(func=cmd_extract)

    p_ts = sub.add_parser(
        "timeseries",
        help="Extract a time series at a point, box, or domain mean",
        formatter_class=HelpFormatter,
        description=(
            "Read one 2-D slab per iteration (memory-efficient). "
            "Default: domain mean. Use --at or --box for point/box averages."
        ),
    )
    add_data_dir(p_ts)
    add_variable(p_ts, required=True)
    add_iterations_arg(p_ts)
    p_ts.add_argument(
        "-l",
        "--level",
        type=int,
        default=None,
        metavar="K",
        help="Vertical level for 3-D fields (default: mid-depth)",
    )
    p_ts.add_argument("--at", default=None, metavar="I,J", help="Grid point indices")
    p_ts.add_argument(
        "--box",
        default=None,
        metavar="I0,I1,J0,J1",
        help="Index bounds for a spatial mean (exclusive upper bounds)",
    )
    p_ts.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="FILE",
        help="Write CSV (.csv) or JSON (.json)",
    )
    p_ts.add_argument(
        "--save-figure",
        metavar="FILE.png",
        default=None,
        help="Save the line plot to PNG",
    )
    p_ts.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the line plot (CSV/JSON only)",
    )
    p_ts.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window",
    )
    add_rec_arg(p_ts)
    add_json_flag(p_ts)
    p_ts.set_defaults(func=cmd_timeseries)

    p_gen = sub.add_parser("generate-sample", help="Create synthetic test data", formatter_class=HelpFormatter)
    p_gen.add_argument("-o", "--output", default="sample_data", metavar="FOLDER")
    p_gen.add_argument(
        "--preset", default="demo",
        choices=["tiny", "demo", "diff_demo", "small", "medium", "large", "stress", "tiled", "many_files", "custom"],
    )
    p_gen.add_argument(
        "--diff-pair",
        action="store_true",
        help="Write ref/ and warm/ subdirs (use with --preset diff_demo)",
    )
    p_gen.add_argument("--nx", type=int, default=None)
    p_gen.add_argument("--ny", type=int, default=None)
    p_gen.add_argument("--nz", type=int, default=None)
    p_gen.add_argument("--n-iters", type=int, default=None, dest="n_iters")
    p_gen.add_argument("--iter-start", type=int, default=0)
    p_gen.add_argument("--iter-step", type=int, default=None)
    p_gen.add_argument("--variables", default=None, metavar="LIST")
    p_gen.add_argument("--tiled", action="store_true")
    p_gen.add_argument("--ntx", type=int, default=2)
    p_gen.add_argument("--nty", type=int, default=2)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--quiet", action="store_true")
    add_json_flag(p_gen)
    p_gen.set_defaults(func=cmd_generate_sample)

    p_gui = sub.add_parser("gui", help="Open the graphical interface", formatter_class=HelpFormatter)
    add_data_dir(p_gui)
    p_gui.set_defaults(func=cmd_gui)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _configure_matplotlib(["mdsview", *argv])
    parser = build_parser()

    try:
        args = parser.parse_args(argv)
        if args.command == "diff":
            if args.later is None and args.iter_a is not None:
                args.later = args.iter_a
            if args.earlier is None and args.iter_b is not None:
                args.earlier = args.iter_b
        return args.func(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except MdsViewError as exc:
        print(f"mdsview: {exc}", file=sys.stderr)
        return exc.exit_code
    except (OSError, ValueError, FileNotFoundError) as exc:
        if os.environ.get("MDSVIEW_DEBUG"):
            raise
        print(f"mdsview: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
