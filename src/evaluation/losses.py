"""Forecast loss functions.

QLIKE is the primary loss: QLIKE(y, f) = y/f - ln(y/f) - 1, minimized at f = y.
It is the standard in the volatility-forecasting literature because it is
robust to noise in the volatility proxy: rankings under QLIKE against a noisy
but conditionally unbiased proxy (like daily-data RV) are consistent for the
rankings against the true variance (Patton 2011), which mean-squared-error on
variance levels is not in general. MSE is reported alongside for completeness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Forecast floor in annualized-variance units: (1 vol pt)^2. Negative or tiny
# forecasts (possible for OLS in levels) are clipped here; clip counts are
# reported, never hidden.
FORECAST_FLOOR = 1e-4

# Evaluation-target floor for QLIKE only: (0.1 vol pt)^2. SPX closed exactly
# flat on 5 days in 1990-2026, making the 1-day realized-variance proxy 0 and
# QLIKE's -ln(y/f) infinite. A zero measured variance is quote-resolution
# artifact, not a true zero. Floored observations are counted and reported.
# MSE and MZ regressions always use the raw target.
TARGET_EVAL_FLOOR = 1e-6


def clip_forecasts(f: pd.Series, floor: float = FORECAST_FLOOR) -> tuple[pd.Series, int]:
    n_clipped = int((f.dropna() < floor).sum())
    return f.clip(lower=floor), n_clipped


def qlike(y: pd.Series, f: pd.Series) -> pd.Series:
    r = y / f
    return pd.Series(r - np.log(r) - 1.0, index=y.index)


def mse(y: pd.Series, f: pd.Series) -> pd.Series:
    return (y - f) ** 2


def oos_r2(y: pd.Series, f: pd.Series, f_bench: pd.Series) -> float:
    """Campbell-Thompson out-of-sample R^2 vs a benchmark forecast."""
    aligned = pd.concat([y, f, f_bench], axis=1, keys=["y", "f", "b"]).dropna()
    sse = float(((aligned["y"] - aligned["f"]) ** 2).sum())
    sse_b = float(((aligned["y"] - aligned["b"]) ** 2).sum())
    return 1.0 - sse / sse_b
