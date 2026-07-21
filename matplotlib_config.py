"""Shared matplotlib defaults for LIBS plots."""

from __future__ import annotations

import os
from pathlib import Path

# Keep the font cache in the project so it stays valid after moves
# (e.g. off iCloud) and does not depend on ~/.matplotlib.
_MPLCONFIG = Path(__file__).resolve().parent / ".mplconfig"
_MPLCONFIG.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG))

import matplotlib as mpl


def apply_matplotlib_config() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (11, 5),
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "lines.linewidth": 0.9,
            "legend.fontsize": 9,
        }
    )
