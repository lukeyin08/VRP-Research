"""Synthetic variance-swap backtest engine.

Instrument: a daily-rolled, constant-maturity SHORT position in 30-day (22
trading day) S&P 500 variance. Each day 1/22 of the book matures and is
re-struck at the current VIX^2 level. Daily P&L per unit short notional:

    unit_pnl[s] = carry[s] + m2m[s]
    carry[s] = (IV[s-1] - 252 * r[s]^2) / 22      # strike amortization vs realization
    m2m[s]   = -(21/22) * (IV[s] - IV[s-1])       # mark on the unexpired book

where IV = (VIX/100)^2. With a flat IV path this telescopes over 22 days to
exactly (strike - realized variance), the swap payoff (unit-tested). The
remaining-maturity mark uses the 30-day VIX for all residual maturities (flat
term structure) - a stated simplification.

Timing: weights are decided at the close of s (signal uses info <= s) and earn
from s+1; trades execute at the close of s and pay costs at s. A poisoning test
asserts day-s P&L cannot see day-s signals.

Sizing: weights are vol-targeted using the trailing realized vol of the UNIT
short-variance P&L (not the strategy's own, avoiding self-reference), capped at
`w_cap`. A month-to-date stop flattens the book for the rest of the month if
net P&L breaches `stop_mtd`. All parameters explicit in BacktestParams.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.backtest.costs import spread_cost_per_notional
from src.config import TRADING_DAYS_PER_YEAR


@dataclass(frozen=True)
class BacktestParams:
    half_spread_vp: float = 0.5  # half-spread in vol points
    roll_days: int = 22
    vol_target: float = 0.10  # annualized, on unit capital
    vol_lookback: int = 66
    w_cap: float = 1.5  # hard notional cap (leverage limit)
    stop_mtd: float = -0.05  # flatten for the month below -5% of capital
    rebalance_every: int = 1  # trade position changes only every k days (roll always runs)

    def as_dict(self) -> dict[str, object]:
        return dict(asdict(self))


def unit_short_var_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """Daily P&L components per 1.0 short notional (before costs)."""
    iv = df["iv30"]
    iv_prev = iv.shift(1)
    realized_day = TRADING_DAYS_PER_YEAR * df["dv_cc"]
    carry = (iv_prev - realized_day) / 22.0
    m2m = -(21.0 / 22.0) * (iv - iv_prev)
    out = pd.DataFrame({"carry": carry, "m2m": m2m})
    out["unit_pnl"] = out["carry"] + out["m2m"]
    return out


def run_strategy(
    df: pd.DataFrame,
    base_w: pd.Series,
    params: BacktestParams,
) -> pd.DataFrame:
    """Sequential daily simulation on the index of `base_w` (NaN => flat).

    Returns a frame with weights, gross/net P&L per unit capital, and costs.
    """
    idx = base_w.index
    unit = unit_short_var_pnl(df).reindex(idx)
    spread = spread_cost_per_notional(df["vix_close"], params.half_spread_vp).reindex(idx)

    # vol estimate of the unit pnl, info through close s (rolling includes s)
    vol_est = unit["unit_pnl"].rolling(params.vol_lookback, min_periods=30).std() * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )
    scale = (params.vol_target / vol_est).clip(upper=params.w_cap)

    n = len(idx)
    w = np.zeros(n)  # w[i] decided at close i, exposure over day i+1
    pnl_gross = np.zeros(n)
    cost = np.zeros(n)
    stopped_month: pd.Period | None = None
    mtd = 0.0
    cur_month = None

    base = base_w.to_numpy(dtype=float)
    unit_np = unit["unit_pnl"].to_numpy(dtype=float)
    spread_np = spread.to_numpy(dtype=float)
    scale_np = scale.to_numpy(dtype=float)
    months = pd.PeriodIndex(idx, freq="M")

    for i in range(n):
        if months[i] != cur_month:
            cur_month = months[i]
            mtd = 0.0
        w_prev = w[i - 1] if i > 0 else 0.0
        pnl_gross[i] = w_prev * unit_np[i] if np.isfinite(unit_np[i]) else 0.0
        mtd += pnl_gross[i]

        # stop check BEFORE trading: today's realized P&L is known at the close
        if mtd < params.stop_mtd and stopped_month != months[i]:
            stopped_month = months[i]

        # decide today's close weight from info <= today; position changes only
        # on the rebalance schedule (the stop overrides it immediately)
        if stopped_month == months[i]:
            desired = 0.0
        elif i % params.rebalance_every != 0:
            desired = w_prev
        elif np.isfinite(base[i]) and np.isfinite(scale_np[i]):
            desired = float(np.clip(base[i] * scale_np[i], 0.0, params.w_cap))
        else:
            desired = 0.0
        # turnover: aged-slice roll + position change (scalar form of costs.daily_turnover)
        turn = w_prev / params.roll_days + abs(desired - w_prev)
        cost[i] = turn * spread_np[i] if np.isfinite(spread_np[i]) else 0.0
        w[i] = desired
        mtd -= cost[i]

    out = pd.DataFrame(
        {
            "w": w,
            "pnl_gross": pnl_gross,
            "cost": cost,
            "pnl_net": pnl_gross - cost,
            "unit_pnl": unit_np,
        },
        index=idx,
    )
    return out
