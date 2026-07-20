"""Diebold-Mariano test with Newey-West (Bartlett/HAC) long-run variance.

The 22-day-ahead target on daily data creates overlapping windows and hence
MA(21)-type serial correlation in loss differentials by construction, plus
volatility clustering on top. Every DM statistic here uses a HAC long-run
variance; the default truncation lag is 2 x horizon = 44 (conservative).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DMResult:
    mean_diff: float  # mean(loss_a - loss_b); negative => a better
    stat: float
    pvalue: float
    n: int
    lag: int


def newey_west_lrv(d: np.ndarray, lag: int) -> float:
    """Long-run variance of the mean with Bartlett weights."""
    d = d - d.mean()
    n = len(d)
    gamma0 = float(d @ d) / n
    lrv = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        gamma_k = float(d[k:] @ d[:-k]) / n
        lrv += 2.0 * w * gamma_k
    # Guard: HAC estimates can go non-positive in small samples
    return max(lrv, 1e-300)


def dm_test(loss_a: pd.Series, loss_b: pd.Series, lag: int = 44) -> DMResult:
    """H0: equal predictive accuracy. Negative stat favors model a."""
    d = (loss_a - loss_b).dropna()
    n = len(d)
    if n < 10 * lag // 4:
        raise ValueError(f"too few observations ({n}) for lag {lag}")
    arr = d.to_numpy(dtype=float)
    lrv = newey_west_lrv(arr, lag)
    stat = float(arr.mean() / np.sqrt(lrv / n))
    pvalue = float(2.0 * (1.0 - stats.norm.cdf(abs(stat))))
    return DMResult(mean_diff=float(arr.mean()), stat=stat, pvalue=pvalue, n=n, lag=lag)
