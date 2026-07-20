"""Phase 3 figure: why summing the variance path matters vs naive 22x scaling."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.plotting.style import apply_style, savefig


def plot_garch_aggregation_check(fc_h1: pd.Series, fc_h22: pd.Series) -> Path:
    """Correct 22-day aggregation (sum of the forecast variance path) vs the
    naive shortcut (1-day forecast x 22, i.e. the annualized 1-day forecast).

    With fitted persistence alpha+beta ~ 0.99 the two differ by only a few
    percent in calm regimes - the divergence concentrates exactly after
    variance shocks, when the model expects mean reversion over the next
    month. Plotting the ratio makes that visible; overlaying levels would not.
    """
    apply_style()
    sub = pd.concat([fc_h22, fc_h1], axis=1, keys=["correct", "naive"]).dropna()
    ratio_pct = (sub["correct"] / sub["naive"] - 1.0) * 100.0

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.plot(ratio_pct.index, ratio_pct, lw=0.7, color="#000000")
    ax.axhline(0, lw=0.8, color="#7f7f7f")
    ax.set_ylabel("Correct 22d aggregate vs naive $\\times$22 (%)")
    ax.set_xlabel("Date")
    ax.set_title(
        "GARCH(1,1): error of the naive x22 shortcut "
        f"(mean {ratio_pct.mean():+.1f}%, min {ratio_pct.min():+.1f}%, "
        f"max {ratio_pct.max():+.1f}%)"
    )
    return savefig(fig, "phase3_agg_check.png")
