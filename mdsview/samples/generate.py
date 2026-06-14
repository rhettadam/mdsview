"""Generate synthetic MITgcm MDS sample runs for testing mdsview."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from .. import io

PRESETS: dict[str, dict] = {
    "tiny": dict(nx=80, ny=60, nz=12, n_iters=5, iter_step=120, tiled=False),
    "demo": dict(nx=240, ny=160, nz=40, n_iters=8, iter_step=360, tiled=False),
    # Clear spatial trends for diff / cross-run testing (see generate_diff_pair).
    "diff_demo": dict(
        nx=128, ny=96, nz=20, n_iters=6, iter_step=600, tiled=False,
        variables=("T", "S", "TRAC", "Eta"),
    ),
    "small": dict(nx=180, ny=120, nz=32, n_iters=10, iter_step=120, tiled=False),
    "medium": dict(nx=360, ny=240, nz=50, n_iters=24, iter_step=120, tiled=False),
    "large": dict(nx=900, ny=600, nz=50, n_iters=12, iter_step=240, tiled=False),
    "stress": dict(nx=1800, ny=1200, nz=50, n_iters=6, iter_step=480, tiled=False),
    "tiled": dict(nx=180, ny=120, nz=24, n_iters=8, iter_step=120, tiled=True, ntx=3, nty=2),
    # Small snapshots, many iterations — stress-tests catalog / iteration lists (~20k files).
    "many_files": dict(nx=64, ny=48, nz=8, n_iters=2000, iter_step=12, tiled=False),
}

VARIABLES_3D = ("T", "S", "UVEL", "VVEL")
VARIABLES_2D = ("Eta",)


@dataclass
class SampleRunInfo:
    output_dir: str
    nx: int
    ny: int
    nz: int
    iterations: list[int]
    variables: list[str]
    nbytes_total: int
    tiled: bool
    recommended_level: int
    recommended_dod: dict[str, int | str]

    @property
    def nbytes_human(self) -> str:
        size = float(self.nbytes_total)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024 or unit == "TB":
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} TB"


@dataclass
class _Mesh:
    nx: int
    ny: int
    nz: int
    xc: np.ndarray
    yc: np.ndarray
    zz: np.ndarray
    z_centers: np.ndarray
    xnorm: np.ndarray
    ynorm: np.ndarray
    znorm: np.ndarray


def _build_mesh(nx: int, ny: int, nz: int) -> _Mesh:
    """Build horizontal grid and normalized coords for synthetic patterns."""
    dx, dy = 4000.0, 4000.0
    x = np.arange(nx, dtype=np.float64) * dx
    y = np.arange(ny, dtype=np.float64) * dy
    z_centers = (np.arange(nz, dtype=np.float64) + 0.5) * (5000.0 / nz)
    xc, yc = np.meshgrid(x, y)
    zz = z_centers[:, np.newaxis, np.newaxis]
    xnorm = xc / max(x[-1], 1.0)
    ynorm = yc / max(y[-1], 1.0)
    znorm = zz / max(z_centers[-1], 1.0)
    return _Mesh(nx, ny, nz, xc, yc, zz, z_centers, xnorm, ynorm, znorm)


def _temperature(mesh: _Mesh, iteration: int, *, variant: str = "baseline") -> np.ndarray:
    """
    Warm surface, cold deep water; slow warming rate varies across the basin.

    diff_demo uses stronger rates so ΔT between times shows clear zonal structure.
    variant='warm_east' doubles warming east of mid-basin (cross-run diff tests).
    """
    thermocline = np.exp(-mesh.znorm * 4.0)
    surface_blob = 3.0 * np.exp(-((mesh.xnorm - 0.35) ** 2 + (mesh.ynorm - 0.55) ** 2) / 0.04)
    base = 4.0 + 18.0 * thermocline + surface_blob
    base += 1.5 * np.sin(2 * np.pi * mesh.xnorm) * np.cos(2 * np.pi * mesh.ynorm) * thermocline
    if variant == "warm_east":
        rate = np.where(
            mesh.xnorm > 0.5,
            0.004 + 0.0012 * mesh.xnorm,
            0.002 + 0.0006 * mesh.xnorm,
        )
    elif variant == "diff_demo":
        rate = 0.002 + 0.0012 * mesh.xnorm
    else:
        rate = 0.0008 + 0.0005 * mesh.xnorm
    return (base + rate * iteration).astype(np.float32)


def _salinity(mesh: _Mesh, iteration: int, *, variant: str = "baseline") -> np.ndarray:
    halocline = np.exp(-mesh.znorm * 3.0)
    fresh_lens = -1.2 * np.exp(-((mesh.xnorm - 0.65) ** 2 + (mesh.ynorm - 0.40) ** 2) / 0.05)
    base = 34.5 + 1.5 * (1.0 - halocline) + fresh_lens
    base += 0.4 * np.cos(2 * np.pi * mesh.ynorm) * halocline
    if variant == "diff_demo":
        rate = 0.0018 + 0.0010 * mesh.ynorm
    else:
        rate = 0.0012 + 0.0007 * mesh.ynorm
    return (base + rate * iteration).astype(np.float32)


def _trac(mesh: _Mesh, iteration: int, *, variant: str = "baseline") -> np.ndarray:
    """Passive tracer blob that drifts east and grows — ideal for snapshot diffs."""
    del variant
    t = iteration / 3000.0
    x0 = 0.15 + 0.55 * min(t, 1.0)
    y0 = 0.45 + 0.1 * np.sin(2 * np.pi * t)
    blob = np.exp(-((mesh.xnorm - x0) ** 2 + (mesh.ynorm - y0) ** 2) / 0.025)
    blob = blob[np.newaxis, :, :] * np.exp(-mesh.znorm * 3.5)
    return (1.2 * min(t, 1.0) * blob).astype(np.float32)


def _u_velocity(mesh: _Mesh, iteration: int) -> np.ndarray:
    jet = np.exp(-((mesh.ynorm - 0.5) ** 2) / 0.015)
    depth = np.exp(-mesh.znorm * 2.5)
    phase = iteration * 2e-4
    return (0.25 * jet * depth * np.sin(2 * np.pi * mesh.xnorm + phase)).astype(np.float32)


def _v_velocity(mesh: _Mesh, iteration: int) -> np.ndarray:
    gyre = np.sin(np.pi * mesh.xnorm) * np.cos(np.pi * mesh.ynorm)
    depth = np.exp(-mesh.znorm * 2.0)
    phase = iteration * 1.5e-4
    return (0.18 * gyre * depth * np.cos(phase)).astype(np.float32)


def _eta(mesh: _Mesh, iteration: int, *, variant: str = "baseline") -> np.ndarray:
    phase = iteration * (5e-4 if variant == "diff_demo" else 3e-4)
    wave = np.sin(2 * np.pi * mesh.xnorm + phase)
    eddy = 0.35 * np.cos(2 * np.pi * mesh.ynorm) * np.sin(4 * np.pi * mesh.xnorm + 0.5 * phase)
    amp = 0.12 if variant == "diff_demo" else 0.08
    return (amp * wave + 0.5 * amp * eddy).astype(np.float32)


def expected_dod_mean(prefix_a: str, prefix_b: str, iter_1: int, iter_2: int) -> float | None:
    """
    Domain-mean DiD for generated T/S (rate fields are linear in x or y).

    rate_T mean ~ 0.0008 + 0.0005 * 0.5 = 0.00105
    rate_S mean ~ 0.0012 + 0.0007 * 0.5 = 0.00155
    mean DiD ~ (0.00155 - 0.00105) * (iter_1 - iter_2)
    """
    if prefix_a != "T" or prefix_b != "S":
        return None
    gap_rate_mean = 0.0005
    return gap_rate_mean * (iter_1 - iter_2)


def expected_dod_has_structure(prefix_a: str, prefix_b: str) -> bool:
    return prefix_a == "T" and prefix_b == "S"


def _write_global(prefix_path: str, arr: np.ndarray, iteration: int | None) -> int:
    io.write_field(prefix_path, arr, iteration=iteration, dataprec="float32")
    return arr.nbytes


def _write_tiled(
    prefix_path: str,
    arr: np.ndarray,
    iteration: int,
    ntx: int,
    nty: int,
) -> int:
    from MITgcmutils import mds

    nz, ny, nx = arr.shape
    tile_nx = int(math.ceil(nx / ntx))
    tile_ny = int(math.ceil(ny / nty))
    nbytes = 0
    base = prefix_path + f".{iteration:010d}"

    for j in range(nty):
        for i in range(ntx):
            x0, x1 = i * tile_nx, min((i + 1) * tile_nx, nx)
            y0, y1 = j * tile_ny, min((j + 1) * tile_ny, ny)
            tile = arr[:, y0:y1, x0:x1]
            tile_path = f"{base}.{i + 1:03d}.{j + 1:03d}"
            mds.wrmds(
                tile_path,
                tile,
                dataprec="float32",
                ndims=3,
                dimlist=(tile.shape[0], tile.shape[1], tile.shape[2]),
            )
            nbytes += tile.nbytes
    return nbytes


def _write_readme(
    path: str,
    *,
    output_dir: str,
    nx: int,
    ny: int,
    nz: int,
    iterations: list[int],
    variables: tuple[str, ...],
    tiled: bool,
    level: int,
    t1: int,
    t2: int,
    diff_pair: bool = False,
    companion_dir: str | None = None,
) -> None:
    t_later, t_earlier = iterations[-1], iterations[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("Synthetic MITgcm sample run (mdsview)\n")
        f.write("=" * 48 + "\n\n")
        f.write(f"Grid       : {nx} x {ny} horizontal, {nz} vertical levels\n")
        f.write(f"Iterations : {iterations}\n")
        f.write(f"Variables  : {', '.join(variables)}\n")
        f.write(f"Tiled      : {tiled}\n\n")
        if diff_pair or "TRAC" in variables:
            f.write("Diff testing design\n")
            f.write("  T    - warming rate increases eastward (ΔT shows W→E gradient)\n")
            f.write("  S    - trend increases northward (ΔS shows S→N gradient)\n")
            f.write("  TRAC - blob drifts east and grows (Δ shows ring / patch)\n")
            f.write("  Eta  - phase-evolving 2-D field (no level index needed)\n")
            if companion_dir:
                f.write(f"  Pair : compare this folder to {companion_dir} at the same iteration\n")
                f.write("         (warm/ has extra heating in the eastern half of T)\n")
            f.write("\n")
        else:
            f.write("Field design\n")
            f.write("  T  - thermocline + warm surface blob; warming faster in the east\n")
            f.write("  S  - halocline + fresh lens; warming faster in the north\n")
            f.write("  DiD(T,S) has visible x/y structure (different spatial warming rates)\n\n")
        f.write("Recommended quick looks\n")
        f.write(f"  Level index for 3-D plots : {level} (mid-depth)\n")
        f.write(f"  Same-run diff             : later={t_later}, earlier={t_earlier}\n")
        f.write(f"  DiD times                 : t1={t1}, t2={t2}\n\n")
        f.write("Commands\n")
        f.write(f"  python -m mdsview.cli info -d {output_dir}\n")
        f.write(
            f"  python -m mdsview.cli diff -v T --later {t_later} --earlier {t_earlier} "
            f"-l {level} -d {output_dir} --plot --no-show --save-figure diff_t.png\n"
        )
        f.write(
            f"  python -m mdsview.cli diff -v TRAC --later {t_later} --earlier {t_earlier} "
            f"-l {level} -d {output_dir} --plot --no-show --save-figure diff_trac.png\n"
        )
        f.write(
            f"  python -m mdsview.cli diff -v Eta --later {t_later} --earlier {t_earlier} "
            f"-d {output_dir} --plot --no-show --save-figure diff_eta.png\n"
        )
        if companion_dir:
            f.write("\nCross-directory diff (GUI Diff tab):\n")
            f.write(f"  Later run   : {output_dir}\n")
            f.write(f"  Earlier run : {companion_dir}\n")
            f.write(f"  Variable T, later/earlier iter {t_later}, level {level}\n")
            f.write("  Expect east-side warming excess in warm/ minus ref/\n\n")
        f.write(
            f"  python -m mdsview.cli dod -a T -b S --time1 {t1} --time2 {t2} "
            f"-l {level} --plot --no-show --save-figure dod_ts.png -d {output_dir}\n"
        )
        f.write(f"  python -m mdsview.cli gui -d {output_dir}\n\n")
        f.write("GUI: Field tab → T, level ~ mid, scrub iterations.\n")
        f.write("     Diff tab → T, later/earlier times above, level ~ mid, Plot.\n")
        if companion_dir:
            f.write("     Diff tab → set Later/Earlier run folders to the ref/ and warm/ pair.\n")


def generate_sample_run(
    output_dir: str,
    *,
    preset: str = "demo",
    nx: int | None = None,
    ny: int | None = None,
    nz: int | None = None,
    n_iters: int | None = None,
    iter_start: int = 0,
    iter_step: int | None = None,
    variables: tuple[str, ...] | None = None,
    tiled: bool | None = None,
    ntx: int = 2,
    nty: int = 2,
    seed: int = 42,
    progress: bool = True,
    variant: str = "baseline",
) -> SampleRunInfo:
    """
    Create a synthetic run with realistic vertical structure and visible DiD patterns.

    T and S use spatially varying linear-in-time trends so DiD(T,S) varies in x/y
    (not a flat constant). Vertical levels show thermocline / halocline structure.
    """
    if preset not in PRESETS and preset != "custom":
        raise ValueError(f"Unknown preset {preset!r}; choose from {', '.join(PRESETS)}")

    cfg = PRESETS.get(preset, {})
    nx = nx if nx is not None else cfg.get("nx", 240)
    ny = ny if ny is not None else cfg.get("ny", 160)
    nz = nz if nz is not None else cfg.get("nz", 40)
    n_iters = n_iters if n_iters is not None else cfg.get("n_iters", 8)
    iter_step = iter_step if iter_step is not None else cfg.get("iter_step", 360)
    tiled = cfg.get("tiled", False) if tiled is None else tiled
    ntx = cfg.get("ntx", ntx)
    nty = cfg.get("nty", nty)

    if variables is None:
        variables = cfg.get("variables") or (VARIABLES_3D + VARIABLES_2D)

    if preset == "diff_demo" and variant == "baseline":
        variant = "diff_demo"

    np.random.seed(seed)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    mesh = _build_mesh(nx, ny, nz)
    iterations = [iter_start + k * iter_step for k in range(n_iters)]
    recommended_level = nz // 2
    t1, t2 = iterations[0], iterations[-1]

    builders = {
        "T": lambda itr: _temperature(mesh, itr, variant=variant),
        "S": lambda itr: _salinity(mesh, itr, variant=variant),
        "UVEL": lambda itr: _u_velocity(mesh, itr),
        "VVEL": lambda itr: _v_velocity(mesh, itr),
        "Eta": lambda itr: _eta(mesh, itr, variant=variant),
        "TRAC": lambda itr: _trac(mesh, itr, variant=variant),
    }

    nbytes_total = 0

    if progress:
        n_files = len(variables) * len(iterations) * 2 + 6
        print(f"Writing grid to {output_dir} ({nx}x{ny}, {nz} levels)")
        print(f"  ~{n_files:,} .data/.meta files ({len(variables)} fields x {len(iterations)} iters)")

    nbytes_total += _write_global(os.path.join(output_dir, "XC"), mesh.xc.astype(np.float32), None)
    nbytes_total += _write_global(os.path.join(output_dir, "YC"), mesh.yc.astype(np.float32), None)
    dx = 4000.0
    dy = 4000.0
    x_edges = (np.arange(nx + 1, dtype=np.float64) - 0.5) * dx
    y_edges = (np.arange(ny + 1, dtype=np.float64) - 0.5) * dy
    xg, yg = np.meshgrid(x_edges, y_edges)
    nbytes_total += _write_global(os.path.join(output_dir, "XG"), xg.astype(np.float32), None)
    nbytes_total += _write_global(os.path.join(output_dir, "YG"), yg.astype(np.float32), None)
    rc = mesh.z_centers.astype(np.float32)
    nbytes_total += _write_global(
        os.path.join(output_dir, "RC"),
        rc[:, np.newaxis, np.newaxis] * np.ones((1, ny, nx), dtype=np.float32),
        None,
    )

    for var in variables:
        if var not in builders:
            raise ValueError(f"Unknown variable {var!r}")
        for k, iteration in enumerate(iterations):
            if progress:
                pct = 100.0 * (k + 1) / len(iterations)
                print(f"\r  {var}: iter {iteration} ({k + 1}/{len(iterations)}, {pct:.0f}%)", end="", flush=True)
            arr = builders[var](iteration)
            path = os.path.join(output_dir, var)
            if tiled and arr.ndim == 3:
                nbytes_total += _write_tiled(path, arr, iteration, ntx, nty)
            else:
                nbytes_total += _write_global(path, arr, iteration)
        if progress:
            print()

    readme = os.path.join(output_dir, "SAMPLE_README.txt")
    _write_readme(
        readme,
        output_dir=output_dir,
        nx=nx,
        ny=ny,
        nz=nz,
        iterations=iterations,
        variables=variables,
        tiled=tiled,
        level=recommended_level,
        t1=t1,
        t2=t2,
    )

    info = SampleRunInfo(
        output_dir=output_dir,
        nx=nx,
        ny=ny,
        nz=nz,
        iterations=iterations,
        variables=list(variables),
        nbytes_total=nbytes_total,
        tiled=tiled,
        recommended_level=recommended_level,
        recommended_dod={"var_a": "T", "var_b": "S", "time1": t1, "time2": t2, "level": recommended_level},
    )

    if progress:
        print(f"Done: {len(variables)} variables x {len(iterations)} iters, ~{info.nbytes_human} total")
        print(f"Try DiD: dod -a T -b S --time1 {t1} --time2 {t2} -l {recommended_level} --plot")
        print(f"See {readme}")

    return info


def generate_diff_pair(
    output_dir: str,
    *,
    seed: int = 42,
    progress: bool = True,
) -> tuple[SampleRunInfo, SampleRunInfo]:
    """
    Write ref/ and warm/ subdirectories for same-iteration cross-run diff tests.

    warm/ has amplified T warming in the eastern half of the basin.
    """
    output_dir = os.path.abspath(output_dir)
    ref_dir = os.path.join(output_dir, "ref")
    warm_dir = os.path.join(output_dir, "warm")

    if progress:
        print(f"Generating diff pair under {output_dir}\n")

    info_ref = generate_sample_run(
        ref_dir,
        preset="diff_demo",
        variant="diff_demo",
        seed=seed,
        progress=progress,
    )
    info_warm = generate_sample_run(
        warm_dir,
        preset="diff_demo",
        variant="warm_east",
        seed=seed,
        progress=progress,
    )

    parent_readme = os.path.join(output_dir, "SAMPLE_README.txt")
    level = info_ref.recommended_level
    t_later = info_ref.iterations[-1]
    with open(parent_readme, "w", encoding="utf-8") as f:
        f.write("mdsview diff testing pair\n")
        f.write("=" * 48 + "\n\n")
        f.write("Folders\n")
        f.write(f"  ref/   baseline warming trends\n")
        f.write(f"  warm/  extra T heating east of basin centre (x > mid)\n\n")
        f.write("Same-run diffs (open ref/ or warm/ alone)\n")
        f.write(f"  Δ T @ level {level}: iter {t_later} minus {info_ref.iterations[0]}\n")
        f.write("  → west-to-east gradient (T) or south-to-north (S)\n")
        f.write(f"  Δ TRAC: moving blob between snapshots\n\n")
        f.write("Cross-run diff (GUI Diff tab)\n")
        f.write("  Later run   : warm/\n")
        f.write("  Earlier run : ref/\n")
        f.write(f"  Variable T, both at iter {t_later}, level {level}\n")
        f.write("  → positive anomaly on eastern half\n\n")
        f.write("Generate again:\n")
        f.write(f"  python -m mdsview.cli generate-sample -o {output_dir} --preset diff_demo --diff-pair\n\n")
        f.write(f"  python -m mdsview.cli gui -d {ref_dir}\n")

    if progress:
        print(f"Pair readme: {parent_readme}")

    return info_ref, info_warm
