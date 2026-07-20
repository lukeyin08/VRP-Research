"""Backtest engine: cost arithmetic, payoff telescoping, and timing tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import daily_turnover, spread_cost_per_notional
from src.backtest.engine import BacktestParams, run_strategy, unit_short_var_pnl
from src.backtest.metrics import deflated_sharpe, max_drawdown, sharpe, var_cvar
from src.backtest.signals import binary_weights, linear_weights


def _market(n: int = 300, seed: int = 23, const_vix: float | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    dv = rng.uniform(2e-5, 3e-4, n)
    vix = np.full(n, const_vix) if const_vix else 15 + 8 * rng.random(n)
    df = pd.DataFrame({"dv_cc": dv, "vix_close": vix}, index=idx)
    df["iv30"] = (df["vix_close"] / 100.0) ** 2
    return df


def test_spread_cost_hand_computed() -> None:
    # sigma = 20%, half-spread 1 vol pt: (0.21)^2 - (0.20)^2 = 0.0041
    vix = pd.Series([20.0])
    got = spread_cost_per_notional(vix, half_spread_vp=1.0).iloc[0]
    assert got == pytest.approx(0.0041, rel=1e-12)
    # zero spread costs nothing
    assert spread_cost_per_notional(vix, 0.0).iloc[0] == 0.0


def test_turnover_hand_computed() -> None:
    w = pd.Series([1.0, 1.0, 0.5])
    w_prev = pd.Series([0.0, 1.0, 1.0])
    got = daily_turnover(w, w_prev, roll_days=22)
    # day 1: open 1.0 (+ prev 0 roll); day 2: hold -> roll 1/22; day 3: 1/22 + 0.5 cut
    assert got.iloc[0] == pytest.approx(1.0)
    assert got.iloc[1] == pytest.approx(1.0 / 22.0)
    assert got.iloc[2] == pytest.approx(1.0 / 22.0 + 0.5)


def test_unit_pnl_telescopes_to_swap_payoff_under_flat_iv() -> None:
    """With IV constant at K, 22 days of rolled P&L == K - realized variance."""
    df = _market(n=60, const_vix=20.0)
    unit = unit_short_var_pnl(df)
    k = (20.0 / 100.0) ** 2
    window = unit["unit_pnl"].iloc[10 : 10 + 22]
    realized = 252.0 * df["dv_cc"].iloc[10 : 10 + 22].mean()
    assert window.sum() == pytest.approx((k - realized) / 1.0, rel=1e-10)


def test_pnl_timing_weight_earns_from_next_day() -> None:
    """The weight decided at close s must earn day s+1's P&L, not day s's."""
    df = _market(n=100)
    base = pd.Series(0.0, index=df.index)
    s = 50
    base.iloc[s] = 1.0  # in the market for exactly one day: close s -> close s+1
    params = BacktestParams(half_spread_vp=0.0, vol_target=1.0, w_cap=1.0, stop_mtd=-99.0)
    res = run_strategy(df, base, params)
    unit = unit_short_var_pnl(df)
    assert res["pnl_gross"].iloc[s] == 0.0
    expected = res["w"].iloc[s] * unit["unit_pnl"].iloc[s + 1]
    assert res["pnl_gross"].iloc[s + 1] == pytest.approx(expected, rel=1e-12)
    assert res["pnl_gross"].iloc[s + 2] == 0.0


def test_signal_poisoning_cannot_reach_todays_pnl() -> None:
    """Poisoning the base weight at s must not change P&L up to and incl. s."""
    df = _market(n=120)
    base = pd.Series(1.0, index=df.index)
    params = BacktestParams(half_spread_vp=0.5, vol_target=0.10, stop_mtd=-99.0)
    baseline = run_strategy(df, base, params)

    s = 80
    poisoned = base.copy()
    poisoned.iloc[s:] = 0.0  # signal flips at s
    res = run_strategy(df, poisoned, params)
    # gross P&L identical through day s (weights only act from s+1)
    pd.testing.assert_series_equal(
        baseline["pnl_gross"].iloc[: s + 1], res["pnl_gross"].iloc[: s + 1]
    )
    # costs at s differ (the trade happens at s) - that is correct - but not before
    pd.testing.assert_series_equal(baseline["cost"].iloc[:s], res["cost"].iloc[:s])


def test_stop_loss_flattens_rest_of_month() -> None:
    df = _market(n=80, const_vix=20.0)
    # engineer a catastrophic realized day (after the vol-target warmup period)
    d = df.index[50]
    df.loc[d, "dv_cc"] = 0.02  # ~14% daily move
    base = pd.Series(1.0, index=df.index)
    params = BacktestParams(half_spread_vp=0.0, vol_target=0.10, stop_mtd=-0.01)
    res = run_strategy(df, base, params)
    month = pd.Period(d, freq="M")
    after = res[(pd.PeriodIndex(res.index, freq="M") == month) & (res.index >= d)]
    assert (after["w"] == 0.0).all()
    next_month = res[pd.PeriodIndex(res.index, freq="M") == month + 1]
    if len(next_month) > 5:
        assert (next_month["w"] > 0).any()  # re-enters after the month turns


def test_weights_respect_cap_and_nan() -> None:
    df = _market(n=150)
    vrp = pd.Series(np.nan, index=df.index)
    assert binary_weights(vrp).isna().all()
    v = pd.Series(0.01, index=df.index)
    assert (linear_weights(v, cap=1.5).dropna() <= 1.5).all()
    params = BacktestParams()
    res = run_strategy(df, binary_weights(v * 0 + 1e-3), params)
    assert (res["w"] <= params.w_cap + 1e-12).all()


def test_metrics_sanity() -> None:
    idx = pd.bdate_range("2015-01-01", periods=1000)
    rng = np.random.default_rng(7)
    pnl = pd.Series(rng.normal(0.001, 0.006, 1000), index=idx)
    assert 0 < sharpe(pnl) < 5
    assert max_drawdown(pnl) < 0
    v95, _c95 = var_cvar(pnl, 0.95)
    v99, c99 = var_cvar(pnl, 0.99)
    assert c99 <= v99 <= v95 < 0
    ds = deflated_sharpe(pnl, n_trials=50)
    assert 0.0 <= ds.dsr <= 1.0
    assert ds.sr_benchmark_ann > 0
