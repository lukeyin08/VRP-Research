"""Transaction-cost model, parameterized in VOL POINTS (not % of premium).

Crossing the spread on a variance swap quoted in vol terms: selling at
(sigma - h) instead of mid sigma gives up sigma^2 - (sigma - h)^2 variance
units per unit notional. We charge the symmetric exact form
(sigma + h)^2 - sigma^2 = 2*sigma*h + h^2 on every unit of notional traded -
slightly conservative (the buy-side crossing) - so results are not flattered.

Turnover of the daily-rolled constant-maturity book: each day 1/22 of the book
matures and is re-struck (w_prev / 22) plus any position change |w - w_prev|.
"""

from __future__ import annotations

import pandas as pd


def spread_cost_per_notional(vix_pct: pd.Series, half_spread_vp: float) -> pd.Series:
    """Variance units given up per 1.0 notional traded at half-spread h vol pts."""
    sigma = vix_pct / 100.0
    h = half_spread_vp / 100.0
    return (sigma + h) ** 2 - sigma**2


def daily_turnover(w: pd.Series, w_prev: pd.Series, roll_days: int = 22) -> pd.Series:
    """Notional traded today: book roll of the aged slice + position changes."""
    return w_prev / roll_days + (w - w_prev).abs()
