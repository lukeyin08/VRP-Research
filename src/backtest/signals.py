"""VRP signals -> base weights. Everything at t uses information through close t.

VRP[t] = IV30[t] - E_t[RV over t+1..t+22]. The physical-variance expectation is
either a model forecast (walk-forward, from Part 1) or trailing 22-day RV (the
naive version). Weight rules are deliberately simple and fixed a priori:

- always_short: w = 1 every day (the benchmark the forecast must beat)
- binary:       w = 1 if VRP > 0 else 0
- linear:       w = clip(VRP / expanding median of past positive VRP, 0, 1.5)
                (self-normalizing; no tuned constant)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def vrp_signal(iv30: pd.Series, physical_forecast: pd.Series) -> pd.Series:
    return (iv30 - physical_forecast).rename("vrp")


def always_short(index: pd.Index) -> pd.Series:
    return pd.Series(1.0, index=index, name="always_short")


def binary_weights(vrp: pd.Series) -> pd.Series:
    w = (vrp > 0).astype(float)
    w[vrp.isna()] = np.nan
    return w.rename("binary")


def linear_weights(vrp: pd.Series, cap: float = 1.5, min_periods: int = 252) -> pd.Series:
    pos = vrp.where(vrp > 0)
    norm = pos.expanding(min_periods=min_periods).median().shift(1)  # strictly past info
    w = (vrp / norm).clip(lower=0.0, upper=cap)
    w[norm.isna() | vrp.isna()] = np.nan
    return w.rename("linear")
