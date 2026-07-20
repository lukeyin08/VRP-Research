"""Non-optional baseline forecasts. Values at t use information through close t only.

Each returns a Series over the full feature index: forecast (made at t) of the
annualized realized variance over t+1 .. t+h.
"""

from __future__ import annotations

import pandas as pd

from src.config import TRADING_DAYS_PER_YEAR


def random_walk(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Next period's RV = this period's RV (trailing window of the same length)."""
    return df[f"rv_cc_trail_{horizon}"].rename("random_walk")


def expanding_mean(df: pd.DataFrame, min_periods: int = 252) -> pd.Series:
    """Expanding mean of annualized daily variance == expanding-window mean RV."""
    dv = df["dv_cc"] * TRADING_DAYS_PER_YEAR
    return dv.expanding(min_periods=min_periods).mean().rename("expanding_mean")


def ewma_riskmetrics(df: pd.DataFrame, lam: float = 0.94) -> pd.Series:
    """RiskMetrics: sigma2_t = lam * sigma2_{t-1} + (1-lam) * r2_t, flat term
    structure across horizons (the RiskMetrics convention)."""
    dv = df["dv_cc"]
    sig2 = dv.ewm(alpha=1.0 - lam, adjust=False).mean() * TRADING_DAYS_PER_YEAR
    return sig2.rename("ewma_riskmetrics")


def vix_as_forecast(df: pd.DataFrame) -> pd.Series:
    """The option market's own 30-day expectation, (VIX/100)^2. A genuinely hard
    benchmark - though biased high by construction since it embeds the VRP."""
    return df["iv30"].rename("vix")
