"""Loss functions, DM test, and MZ regression sanity, with hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation.dm import dm_test, newey_west_lrv
from src.evaluation.losses import clip_forecasts, mse, oos_r2, qlike
from src.evaluation.mz import mz_regression


def _s(vals: list[float]) -> pd.Series:
    return pd.Series(vals, index=pd.bdate_range("2020-01-01", periods=len(vals)))


def test_qlike_zero_at_perfect_forecast() -> None:
    y = _s([0.01, 0.04, 0.09])
    assert qlike(y, y).abs().max() < 1e-15


def test_qlike_hand_computed() -> None:
    # y/f = 2 -> QLIKE = 2 - ln 2 - 1; y/f = 0.5 -> 0.5 - ln 0.5 - 1
    y = _s([0.02, 0.01])
    f = _s([0.01, 0.02])
    expected = [2 - math.log(2) - 1, 0.5 - math.log(0.5) - 1]
    np.testing.assert_allclose(qlike(y, f).to_numpy(), expected, rtol=1e-12)
    # under-forecasting (f = y/2) hurts more than over-forecasting (f = 2y)
    assert expected[0] > expected[1]


def test_mse_and_clip() -> None:
    y = _s([0.02, 0.03])
    f = _s([0.01, 0.05])
    np.testing.assert_allclose(mse(y, f).to_numpy(), [1e-4, 4e-4], rtol=1e-12)
    clipped, n = clip_forecasts(_s([-0.01, 0.05, 1e-6]), floor=1e-4)
    assert n == 2
    assert clipped.min() == pytest.approx(1e-4)


def test_oos_r2_bounds() -> None:
    rng = np.random.default_rng(1990)
    y = _s(list(rng.uniform(0.01, 0.05, 200)))
    bench = y * 0 + float(y.mean())
    assert oos_r2(y, y, bench) == pytest.approx(1.0)
    assert oos_r2(y, bench, bench) == pytest.approx(0.0)


def test_newey_west_iid_close_to_variance() -> None:
    rng = np.random.default_rng(7)
    d = rng.normal(0, 1.0, 20000)
    lrv = newey_west_lrv(d, lag=10)
    assert lrv == pytest.approx(1.0, rel=0.1)


def test_dm_detects_dominated_model() -> None:
    rng = np.random.default_rng(11)
    base = pd.Series(rng.uniform(0.9, 1.1, 2000), index=pd.bdate_range("2010-01-01", periods=2000))
    worse = base + 0.5
    res = dm_test(base, worse, lag=10)
    assert res.mean_diff == pytest.approx(-0.5, rel=1e-9)
    assert res.stat < -10
    assert res.pvalue < 1e-10


def test_dm_no_difference_high_pvalue() -> None:
    rng = np.random.default_rng(13)
    idx = pd.bdate_range("2010-01-01", periods=3000)
    a = pd.Series(rng.uniform(0.9, 1.1, 3000), index=idx)
    b = pd.Series(rng.uniform(0.9, 1.1, 3000), index=idx)
    res = dm_test(a, b, lag=10)
    assert res.pvalue > 0.05


def test_mz_near_perfect_forecast() -> None:
    rng = np.random.default_rng(17)
    f = pd.Series(rng.uniform(0.01, 0.09, 1500), index=pd.bdate_range("2010-01-01", periods=1500))
    y = f + rng.normal(0, 1e-5, 1500)
    res = mz_regression(y, f, hac_lag=10)
    assert res.alpha == pytest.approx(0.0, abs=1e-4)
    assert res.beta == pytest.approx(1.0, abs=1e-2)
    assert res.p_joint > 0.05
    assert res.r2 > 0.99


def test_mz_flags_overreacting_forecast() -> None:
    rng = np.random.default_rng(19)
    f = pd.Series(rng.uniform(0.01, 0.09, 1500), index=pd.bdate_range("2010-01-01", periods=1500))
    y = 0.5 * f + 0.01 + pd.Series(rng.normal(0, 1e-3, 1500), index=f.index)
    res = mz_regression(y, f, hac_lag=10)
    assert res.beta == pytest.approx(0.5, abs=0.05)
    assert res.p_joint < 1e-6
