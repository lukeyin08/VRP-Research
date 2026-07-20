"""Lookahead-bias guards: the target at date t must depend only on data after t.

These tests exist because the single most dangerous bug in this project is a
misaligned target: VIX at close of t must be paired with realized variance over
t+1..t+22, never a window that reaches back into t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import rv
from src.features.build import assemble_features


def _synthetic_market(n: int = 260, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-05", periods=n)
    c = 2000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.roll(c, 1) * np.exp(rng.normal(0, 0.002, n))
    o[0] = 2000.0
    hi = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    spx = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c}, index=idx)
    v = 15 + 5 * np.abs(rng.normal(0, 1, n))
    vix = pd.DataFrame({"open": v, "high": v * 1.05, "low": v * 0.95, "close": v}, index=idx)
    return spx, vix


def test_forward_target_uses_exact_future_slice() -> None:
    idx = pd.bdate_range("2019-01-01", periods=60)
    dv = pd.Series(np.arange(1.0, 61.0) * 1e-5, index=idx)  # distinct values
    h = 22
    target = rv.realized_var_forward(dv, h)
    for t in (0, 7, 30, len(dv) - h - 1):
        manual = dv.iloc[t + 1 : t + 1 + h].sum() * 252 / h
        assert target.iloc[t] == pytest.approx(manual, rel=1e-12)
    assert target.iloc[len(dv) - h :].isna().all()


def test_target_immune_to_past_mutation() -> None:
    """Poison everything at or before t; the target at t must not move."""
    idx = pd.bdate_range("2019-01-01", periods=80)
    rng = np.random.default_rng(11)
    dv = pd.Series(rng.uniform(1e-5, 5e-4, len(idx)), index=idx)
    h, t = 22, 30

    baseline = rv.realized_var_forward(dv, h).iloc[t]

    poisoned = dv.copy()
    poisoned.iloc[: t + 1] = 99.0  # absurd values at all dates <= t
    assert rv.realized_var_forward(poisoned, h).iloc[t] == pytest.approx(baseline, rel=1e-12)

    poisoned2 = dv.copy()
    poisoned2.iloc[t + 1] *= 10  # first date inside the window
    assert rv.realized_var_forward(poisoned2, h).iloc[t] != pytest.approx(baseline, rel=1e-12)


def test_trailing_rv_uses_only_past_and_present() -> None:
    idx = pd.bdate_range("2019-01-01", periods=80)
    rng = np.random.default_rng(13)
    dv = pd.Series(rng.uniform(1e-5, 5e-4, len(idx)), index=idx)
    h, t = 22, 40

    baseline = rv.realized_var_trailing(dv, h).iloc[t]
    poisoned = dv.copy()
    poisoned.iloc[t + 1 :] = 99.0  # poison the future; trailing value must not move
    assert rv.realized_var_trailing(poisoned, h).iloc[t] == pytest.approx(baseline, rel=1e-12)


def test_features_frame_pairs_vix_t_with_future_rv() -> None:
    """End-to-end wiring check on the assembled frame."""
    spx, vix = _synthetic_market()
    df, _, _ = assemble_features(spx, vix, term={}, strict=False)

    h = 22
    t = 100
    date_t = df.index[t]
    # target at t equals annualized mean of cc daily variance over rows t+1..t+22
    manual = df["dv_cc"].iloc[t + 1 : t + 1 + h].sum() * 252 / h
    assert df.loc[date_t, "target_rv_22"] == pytest.approx(manual, rel=1e-10)
    # iv30 at t is (VIX_t/100)^2 - same-day close, no shift
    assert df.loc[date_t, "iv30"] == pytest.approx((vix.loc[date_t, "close"] / 100) ** 2, rel=1e-12)
    # ex-post VRP is their difference
    assert df.loc[date_t, "vrp_expost"] == pytest.approx(
        df.loc[date_t, "iv30"] - df.loc[date_t, "target_rv_22"], rel=1e-12
    )
    # trailing feature at t must not include any daily variance after t
    manual_trail = df["dv_cc"].iloc[t - h + 1 : t + 1].sum() * 252 / h
    assert df.loc[date_t, "rv_cc_trail_22"] == pytest.approx(manual_trail, rel=1e-10)
