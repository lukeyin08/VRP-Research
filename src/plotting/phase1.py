"""Phase 1 figures: implied vs subsequently-realized vol, and the ex-post VRP."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.style import apply_style, savefig


def plot_vix_vs_forward_rv(df: pd.DataFrame) -> Path:
    apply_style()
    d = df.dropna(subset=["iv30", "target_rv_22"])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(d.index, d["vix_close"], lw=0.7, label="VIX (implied, 30d)", color="#000000")
    ax.plot(
        d.index,
        np.sqrt(d["target_rv_22"]) * 100,
        lw=0.7,
        label="Realized vol, next 22 trading days",
        color="#969696",
    )
    ax.set_ylabel("Annualized volatility (%)")
    ax.set_xlabel("Date")
    ax.set_title("Implied (VIX) vs subsequently realized S&P 500 volatility")
    ax.legend(loc="upper left")
    return savefig(fig, "phase1_vix_vs_forward_rv.png")


def plot_vrp_expost(df: pd.DataFrame) -> Path:
    apply_style()
    d = df.dropna(subset=["vrp_expost"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [2.4, 1]})
    ax1.plot(d.index, d["vrp_expost"], lw=0.6, color="#000000")
    ax1.axhline(0, lw=0.8, color="#7f7f7f")
    ax1.axhline(
        d["vrp_expost"].mean(),
        lw=1.0,
        ls="--",
        color="#404040",
        label=f"mean = {d['vrp_expost'].mean():.4f}",
    )
    ax1.set_ylabel("VRP (annualized variance units)")
    ax1.set_xlabel("Date")
    ax1.set_title("Ex-post variance risk premium: (VIX/100)$^2$ $-$ realized var (t+1..t+22)")
    ax1.legend(loc="lower left")

    clip = d["vrp_expost"].clip(d["vrp_expost"].quantile(0.005), d["vrp_expost"].quantile(0.995))
    ax2.hist(clip, bins=80, color="#7f7f7f")
    ax2.axvline(0, lw=0.8, color="#000000")
    ax2.set_xlabel("VRP (variance units, clipped at 0.5/99.5%)")
    ax2.set_ylabel("Days")
    ax2.set_title(f"{(d['vrp_expost'] > 0).mean():.0%} of days positive")
    return savefig(fig, "phase1_vrp_expost.png")
