"""Colormap names and resolution (matplotlib + cmocean)."""

from __future__ import annotations

from typing import Any

# Matplotlib colormaps useful for tracers and differences
MATPLOTLIB_CMAPS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "turbo",
    "RdBu_r",
    "RdBu",
    "coolwarm",
    "seismic",
    "PiYG_r",
    "BrBG_r",
    "hot",
    "terrain",
    "ocean",
]

# Main cmocean colormaps (see https://cmocean.readthedocs.io)
CMOCEAN_CMAPS = [
    "thermal",
    "haline",
    "solar",
    "ice",
    "gray",
    "oxy",
    "deep",
    "dense",
    "algae",
    "matter",
    "turbid",
    "speed",
    "amp",
    "tempo",
    "phase",
    "balance",
    "delta",
    "curl",
    "diff",
    "tarn",
]

COLORMAP_CHOICES = MATPLOTLIB_CMAPS + CMOCEAN_CMAPS

DEFAULT_PLOT_CMAP = "thermal"
DEFAULT_DIFF_CMAP = "balance"


def _ensure_cmocean() -> Any:
    import cmocean

    return cmocean


def resolve_cmap(name: str) -> Any:
    """Return a matplotlib Colormap from a name (matplotlib or cmocean)."""
    import matplotlib.pyplot as plt

    try:
        return plt.get_cmap(name)
    except (ValueError, TypeError):
        pass

    cmo = _ensure_cmocean()
    if hasattr(cmo.cm, name):
        return getattr(cmo.cm, name)

    reversed_name = name[:-2] if name.endswith("_r") else None
    if reversed_name and hasattr(cmo.cm, f"{reversed_name}_r"):
        return getattr(cmo.cm, f"{reversed_name}_r")

    raise ValueError(f"Unknown colormap: {name!r}")


def is_valid_cmap(name: str) -> bool:
    try:
        resolve_cmap(name)
        return True
    except (ValueError, ImportError):
        return False
