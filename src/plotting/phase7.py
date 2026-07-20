"""Phase 7 figure: full-period equity curves with the holdout demarcated."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.config import HOLDOUT_START
from src.plotting.phase5 import STYLES
from src.plotting.style import apply_style, savefig


def plot_full_equity(pnl: pd.DataFrame) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(9.8, 4.4))
    for name in pnl.columns:
        ax.plot(pnl.index, pnl[name].cumsum(), label=name, **STYLES.get(name, {}))
    x_hold = float(mdates.date2num(pd.Timestamp(HOLDOUT_START)))
    ax.axvline(x_hold, color="#000000", lw=0.9, ls="--")
    ymax = ax.get_ylim()[1]
    ax.annotate(
        "holdout begins\n(evaluated once)",
        xy=(x_hold, ymax * 0.92),
        fontsize=8,
        ha="left",
    )
    ax.set_ylabel("Cumulative net P&L (per unit capital)")
    ax.set_xlabel("Date")
    ax.set_title("Net equity curves at 0.5 vol-pt half-spread, 2000-2026")
    ax.legend(loc="upper left", fontsize=8)
    return savefig(fig, "phase7_equity_full.png")
