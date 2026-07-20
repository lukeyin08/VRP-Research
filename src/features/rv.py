"""Realized-variance estimators from daily OHLC data.

Units convention: every function returns DAILY variance of log returns
(decimal^2). Aggregation to annualized h-day realized variance is done by
`realized_var_forward` / `realized_var_trailing`: ann RV = (252/h) * sum.

Estimator notes (disclosed prominently in the README):
- cc (close-to-close squared log return) is the payoff-relevant series: the
  floating leg of a variance swap is defined on close-to-close daily returns.
  It is unbiased but noisy, and includes the overnight gap.
- parkinson / garman_klass / rogers_satchell use the intraday range. They are
  more efficient but EXCLUDE the overnight gap, so they are biased low as
  estimates of whole-day variance. We use them as conditioning features and
  robustness checks, never as the strategy payoff.
- yang_zhang combines overnight, open-to-close, and Rogers-Satchell terms over
  a rolling window; it is drift-independent and includes the overnight gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS_PER_YEAR

_LN2 = float(np.log(2.0))


def _log(s: pd.Series) -> pd.Series:
    """Elementwise natural log, preserving the Series type and index."""
    return pd.Series(np.log(s.to_numpy(dtype=float)), index=s.index)


def log_returns(close: pd.Series) -> pd.Series:
    return _log(close / close.shift(1))


def cc_var(close: pd.Series) -> pd.Series:
    """Squared close-to-close log return (daily variance, includes overnight)."""
    r = log_returns(close)
    return (r**2).rename("cc")


def parkinson_var(high: pd.Series, low: pd.Series) -> pd.Series:
    hl = _log(high / low)
    return (hl**2 / (4.0 * _LN2)).rename("parkinson")


def garman_klass_var(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    hl = _log(high / low)
    co = _log(close / open_)
    return (0.5 * hl**2 - (2.0 * _LN2 - 1.0) * co**2).rename("garman_klass")


def rogers_satchell_var(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    hc, ho = _log(high / close), _log(high / open_)
    lc, lo = _log(low / close), _log(low / open_)
    return (hc * ho + lc * lo).rename("rogers_satchell")


def yang_zhang_var(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 22,
) -> pd.Series:
    """Rolling Yang-Zhang per-day variance over `window` days (trailing).

    sigma^2_YZ = sigma^2_overnight + k * sigma^2_open_to_close + (1-k) * mean(RS),
    k = 0.34 / (1.34 + (n+1)/(n-1)).
    """
    n = window
    overnight = _log(open_ / close.shift(1))
    open_close = _log(close / open_)
    rs = rogers_satchell_var(open_, high, low, close)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var_o = overnight.rolling(n).var(ddof=1)
    var_c = open_close.rolling(n).var(ddof=1)
    rs_bar = rs.rolling(n).mean()
    return (var_o + k * var_c + (1.0 - k) * rs_bar).rename("yang_zhang")


def realized_var_forward(
    daily_var: pd.Series, horizon: int, trading_days: int = TRADING_DAYS_PER_YEAR
) -> pd.Series:
    """Annualized realized variance over t+1 .. t+horizon, indexed at t.

    This is the forecast TARGET. The window starts strictly after t: nothing
    dated <= t enters the value at t. The final `horizon` observations are NaN.
    Implementation: rolling sum ending at t+h covers t+1..t+h; shift(-h) moves
    it back to t.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    fwd_sum = daily_var.rolling(horizon).sum().shift(-horizon)
    return (fwd_sum * (trading_days / horizon)).rename(f"target_rv_{horizon}")


def realized_var_trailing(
    daily_var: pd.Series, horizon: int, trading_days: int = TRADING_DAYS_PER_YEAR
) -> pd.Series:
    """Annualized realized variance over t-horizon+1 .. t (known at close of t)."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return (daily_var.rolling(horizon).sum() * (trading_days / horizon)).rename(
        f"rv_trail_{horizon}"
    )


def bipower_var_trailing(
    returns: pd.Series, horizon: int, trading_days: int = TRADING_DAYS_PER_YEAR
) -> pd.Series:
    """Annualized bipower variation over the trailing `horizon` days.

    BV over r_1..r_n = (pi/2) * (n/(n-1)) * sum_{i=2..n} |r_i||r_{i-1}|, which is
    robust to jumps; max(RV - BV, 0) proxies the jump contribution
    (Barndorff-Nielsen & Shephard). NOTE: applied here at DAILY frequency across
    days in the window - a coarse approximation of the intraday construction,
    disclosed as such wherever it is used.
    """
    if horizon < 2:
        raise ValueError("bipower needs horizon >= 2")
    prod = returns.abs() * returns.abs().shift(1)
    # the n-day window (t-n+1 .. t) contains n-1 adjacent products, ending at t
    window_sum = prod.rolling(horizon - 1).sum()
    return ((np.pi / 2.0) * window_sum * (trading_days / (horizon - 1.0))).rename(
        f"bv_trail_{horizon}"
    )


def jump_component(rv_trail: pd.Series, bv_trail: pd.Series) -> pd.Series:
    """Jump proxy: max(RV - BV, 0), annualized variance units."""
    return (rv_trail - bv_trail).clip(lower=0.0).rename("jump")
