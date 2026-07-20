"""Phase 5 figures: the breakeven-cost chart (the headline) and equity curves."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.plotting.style import apply_style, savefig

STYLES: dict[str, dict[str, Any]] = {
    "always_short": {"color": "#9e9e9e", "ls": "-", "lw": 1.6},
    "always_short_5d": {"color": "#9e9e9e", "ls": "--", "lw": 1.1},
    "naive_binary": {"color": "#666666", "ls": "--", "lw": 1.2},
    "model_binary": {"color": "#000000", "ls": "-", "lw": 1.6},
    "model_binary_5d": {"color": "#000000", "ls": "--", "lw": 1.2},
    "model_linear": {"color": "#000000", "ls": ":", "lw": 1.4},
    "model_linear_5d": {"color": "#404040", "ls": ":", "lw": 1.1},
    "robust_binary": {"color": "#bdbdbd", "ls": "-.", "lw": 1.1},
}


def plot_breakeven(breakeven: pd.DataFrame) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for name in breakeven.columns:
        st = STYLES.get(name, {})
        ax.plot(breakeven.index, breakeven[name], label=name, **st)
    ax.axhline(0.0, lw=0.8, color="#000000")
    ax.axvspan(0.5, 1.0, color="#dddddd", alpha=0.5, label="realistic SPX spreads")
    ax.set_xlabel("Assumed half-spread (vol points)")
    ax.set_ylabel("Net annualized Sharpe")
    ax.set_title("Strategy Sharpe vs transaction costs - the chart that decides the project")
    ax.legend(loc="upper right", fontsize=8)
    return savefig(fig, "phase5_breakeven.png")


def plot_equity(pnls: dict[str, pd.Series]) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for name, pnl in pnls.items():
        st = STYLES.get(name, {})
        ax.plot(pnl.index, pnl.cumsum(), label=name, **st)
    ax.set_ylabel("Cumulative net P&L (per unit capital)")
    ax.set_xlabel("Date")
    ax.set_title("Net equity curves at 0.5 vol-pt half-spread, development period")
    ax.legend(loc="upper left", fontsize=8)
    return savefig(fig, "phase5_equity.png")
