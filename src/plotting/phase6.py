"""Phase 6 figures: drawdown profile and crisis event studies."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.metrics import drawdown_series
from src.plotting.phase5 import STYLES
from src.plotting.style import apply_style, savefig


def plot_drawdowns(pnls: dict[str, pd.Series]) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    for name, pnl in pnls.items():
        dd = drawdown_series(pnl)
        ax.plot(dd.index, dd, label=name, **STYLES.get(name, {}))
    ax.set_ylabel("Drawdown (per unit capital)")
    ax.set_xlabel("Date")
    ax.set_title("Net drawdowns at 0.5 vol-pt half-spread, development period")
    ax.legend(loc="lower left", fontsize=8)
    return savefig(fig, "phase6_drawdowns.png")


def plot_event_studies(
    pnl: pd.DataFrame,
    strategies: list[str],
    events: dict[str, tuple[str, str]],
    fname: str = "phase6_events_dev.png",
) -> Path:
    apply_style()
    n = len(events)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 3.6))
    axes_list = [axes] if n == 1 else list(axes)
    for ax, (label, (start, end)) in zip(axes_list, events.items(), strict=True):
        for name in strategies:
            seg = pnl[name].loc[start:end].fillna(0.0)
            ax.plot(seg.index, seg.cumsum(), label=name, **STYLES.get(name, {}))
        ax.axhline(0, lw=0.6, color="#bbbbbb")
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Cumulative net P&L")
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    axes_list[0].legend(loc="lower left", fontsize=7)
    return savefig(fig, fname)
