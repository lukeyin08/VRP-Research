"""Publication-quality plot defaults: labeled axes, no chartjunk, grayscale-safe."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler

from src.config import FIGURES_DIR

matplotlib.use("Agg")

# grayscale-distinguishable: black solid, mid-gray, light-gray + linestyle variation
COLORS = ["#000000", "#7f7f7f", "#bdbdbd", "#404040"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (9, 4.5),
            "figure.constrained_layout.use": True,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": cycler(color=COLORS),
            "legend.frameon": False,
        }
    )


def savefig(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path
