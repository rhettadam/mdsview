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
    add_json_flag,
    add_plot_options,
    add_variable,
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


def _close_memmap(arr) -> None:
    if hasattr(arr, "base") and hasattr(arr.base, "close"):
        arr.base.close()


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
    require_prefix(data_dir, args.prefix)
    later = args.later if args.later is not None else args.iter_a
    earlier = args.earlier if args.earlier is not None else args.iter_b
    if later is None or earlier is None:
        raise MdsViewError("Provide both --later and --earlier (or two positional iteration numbers).", exit_code=2)

    require_iteration(data_dir, args.prefix, later)
    require_iteration(data_dir, args.prefix, earlier)

    shape = io.field_info(data_dir, args.prefix).shape
    _, nz = ops.level_axis(shape)
    want_full_volume = args.save_field and args.level is None and nz > 1

    if want_full_volume:
        diff, meta = ops.diff_fields(data_dir, args.prefix, later, earlier)
    else:
        diff, meta = ops.diff_slice(data_dir, args.prefix, later, earlier, level=args.level)

    summary = ops.stats(np.asarray(diff))
    if args.json:
        print(
            json.dumps(
                {
                    "formula": f"{args.prefix}({later}) - {args.prefix}({earlier})",
                    "meta": meta,
                    "stats": summary,
                },
                indent=2,
            )
        )
    else:
        print(f"{args.prefix}({later}) - {args.prefix}({earlier})")
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


def cmd_dod(args: argparse.Namespace) -> int:
    data_dir = abs_data_dir(args.dir)
    require_prefix(data_dir, args.var_a)
    require_prefix(data_dir, args.var_b)
    require_iteration(data_dir, args.var_a, args.time1)
    require_iteration(data_dir, args.var_b, args.time1)
    require_iteration(data_dir, args.var_a, args.time2)
    require_iteration(data_dir, args.var_b, args.time2)

    var_a = args.var_a
    var_b = args.var_b
    time1 = args.time1
    time2 = args.time2

    formula = f"({var_b}@{time1} - {var_a}@{time1}) - ({var_b}@{time2} - {var_a}@{time2})"
    print("Difference-of-differences:")
    print(f"  {formula}")
    print("  (how the gap between the two variables changed from time1 to time2)")

    summary = ops.streaming_stats(data_dir, var_a, var_b, time1, time2, rec=args.rec)
    if args.json:
        print(json.dumps({"formula": formula, "stats": summary}, indent=2))
    else:
        print("Stats:")
        for key, value in summary.items():
            print(f"  {key}: {value:.6g}")

    if not (args.save_field or args.plot or args.output):
        return 0

    levels = None
    if (args.plot or args.output) and not args.save_field:
        meta = io.read_field_meta(data_dir, var_a, time1)
        _, nz = ops.level_axis(meta.shape)
        default_level = nz // 2 if nz > 1 else 0
        levels = [args.level if args.level is not None else default_level]

    result, meta = ops.difference_of_differences(
        data_dir,
        var_a,
        var_b,
        time1,
        time2,
        rec=args.rec,
        levels=levels,
        progress=not args.quiet,
    )

    try:
        if args.save_field:
            out_base = args.save_field
            if out_base.endswith(".data"):
                out_base = out_base[:-5]
            io.write_field(out_base, result, iteration=time1)
            print(f"Wrote {out_base}.data / {out_base}.meta")

        if args.plot or args.output:
            shape = io.field_info(data_dir, var_a).shape
            plot_level = args.level if args.level is not None else plotting.effective_level(shape, None)
            title = plotting.format_dod_title(
                var_a, var_b, time1, time2,
                level=plot_level,
                shape=shape,
            )
            plotting.plot_array(
                result,
                data_dir,
                title=title,
                level=args.level,
                cmap=args.cmap,
                vmin=args.vmin,
                vmax=args.vmax,
                symmetric=True,
                save=args.output,
                show=_should_show(args),
            )
    finally:
        _close_memmap(result)
        mmap_path = meta.get("mmap_path")
        if mmap_path and os.path.exists(mmap_path) and not args.save_field:
            os.unlink(mmap_path)

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
        description="Computes field(LATER) - field(EARLIER). Default: one 2-D slice (memory-efficient).",
    )
    add_data_dir(p_diff)
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

    p_dod = sub.add_parser(
        "dod",
        help="Difference-of-differences between two variables",
        formatter_class=HelpFormatter,
        description="Formula: (B@time1 - A@time1) - (B@time2 - A@time2). Level-by-level I/O for 3-D fields.",
    )
    add_data_dir(p_dod)
    req_d = p_dod.add_argument_group("Variables and times (required)")
    req_d.add_argument("-a", "--var-a", "--variable-a", dest="var_a", required=True, metavar="NAME")
    req_d.add_argument("-b", "--var-b", "--variable-b", dest="var_b", required=True, metavar="NAME")
    req_d.add_argument("--time1", "--t1", dest="time1", type=int, required=True, metavar="N")
    req_d.add_argument("--time2", "--t2", dest="time2", type=int, required=True, metavar="N")
    req_d.add_argument("--rec", type=int, default=None, metavar="K", help="Record index for multi-record files")
    out_d = p_dod.add_argument_group("Output")
    out_d.add_argument("--save-field", metavar="PREFIX", help="Write DiD result to .data/.meta")
    out_d.add_argument("--plot", action="store_true", help="Plot one level of the result")
    add_plot_options(out_d, default_cmap=DEFAULT_DIFF_CMAP)
    out_d.add_argument("--quiet", action="store_true", help="Hide per-level progress bar")
    add_json_flag(p_dod)
    p_dod.set_defaults(func=cmd_dod)

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
