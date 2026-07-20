"""Phase 2 figures: forecast vs realized time series, and the MZ scatter."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.style import apply_style, savefig


def _vol_pts(v: pd.Series) -> pd.Series:
    return pd.Series(np.sqrt(v.to_numpy(dtype=float)) * 100.0, index=v.index)


def plot_forecast_ts(y: pd.Series, f_har: pd.Series, f_vix: pd.Series, har_name: str) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.plot(y.index, _vol_pts(y), lw=0.8, color="#bdbdbd", label="Realized (next 22d)")
    ax.plot(f_vix.index, _vol_pts(f_vix), lw=0.7, color="#7f7f7f", ls=":", label="VIX")
    ax.plot(f_har.index, _vol_pts(f_har), lw=0.8, color="#000000", label=f"{har_name} forecast")
    ax.set_ylabel("Annualized volatility (%)")
    ax.set_xlabel("Date")
    ax.set_title("Walk-forward 22-day volatility forecasts, development period")
    ax.legend(loc="upper left", ncols=3)
    return savefig(fig, "phase2_forecasts_ts.png")


def plot_mz_scatter(y: pd.Series, f: pd.Series, name: str) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.scatter(f, y, s=4, alpha=0.35, color="#404040", edgecolors="none")
    lim_lo = float(min(f.min(), y.min())) * 0.8
    lim_hi = float(max(f.max(), y.max())) * 1.2
    grid = np.linspace(lim_lo, lim_hi, 2)
    ax.plot(grid, grid, lw=1.0, color="#000000", label="45$^\\circ$ (unbiased)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"{name} forecast (annualized variance)")
    ax.set_ylabel("Realized variance, t+1..t+22")
    ax.set_title("Mincer-Zarnowitz (log axes)")
    ax.legend(loc="upper left")
    return savefig(fig, "phase2_mz_scatter.png")
