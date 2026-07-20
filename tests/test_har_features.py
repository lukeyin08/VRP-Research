"""HAR feature construction against hand-computed examples."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import rv
from src.models.har import har_design


def _feature_frame(dv: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"dv_cc": dv})
    for h in (1, 5, 22):
        df[f"rv_cc_trail_{h}"] = rv.realized_var_trailing(dv, h)
    df["target_rv_22"] = rv.realized_var_forward(dv, 22)
    df["iv30"] = 0.04
    return df


def test_har_features_hand_computed() -> None:
    n = 80  # must exceed t + 22, else the target is (correctly) NaN
    idx = pd.bdate_range("2010-01-01", periods=n)
    dv = pd.Series(np.arange(1.0, n + 1.0) * 1e-5, index=idx)  # 1e-5, 2e-5, ...
    df = _feature_frame(dv)
    x, y = har_design(df, horizon=22)

    t = 40  # 0-based position
    # daily component: today's annualized variance
    assert x.iloc[t]["rv_cc_trail_1"] == pytest.approx(dv.iloc[t] * 252, rel=1e-12)
    # weekly: mean of the last 5 daily variances (t-4 .. t), annualized
    assert x.iloc[t]["rv_cc_trail_5"] == pytest.approx(
        dv.iloc[t - 4 : t + 1].sum() * 252 / 5, rel=1e-12
    )
    # monthly: mean of the last 22 (t-21 .. t), annualized
    assert x.iloc[t]["rv_cc_trail_22"] == pytest.approx(
        dv.iloc[t - 21 : t + 1].sum() * 252 / 22, rel=1e-12
    )
    # target at t is the future window t+1 .. t+22 - and nothing earlier
    assert y.iloc[t] == pytest.approx(dv.iloc[t + 1 : t + 23].sum() * 252 / 22, rel=1e-12)


def test_har_features_constant_series() -> None:
    idx = pd.bdate_range("2010-01-01", periods=80)
    dv = pd.Series(3e-5, index=idx)
    x, y = har_design(_feature_frame(dv), horizon=22)
    t = 50
    expected = 3e-5 * 252
    for col in ("rv_cc_trail_1", "rv_cc_trail_5", "rv_cc_trail_22"):
        assert x.iloc[t][col] == pytest.approx(expected, rel=1e-12)
    assert y.iloc[t] == pytest.approx(expected, rel=1e-12)


def test_har_design_log_floor_handles_zero_variance_days() -> None:
    """SPX has a handful of exactly-flat closes; log-space design must stay finite."""
    idx = pd.bdate_range("2010-01-01", periods=80)
    rng = np.random.default_rng(9)
    dv = pd.Series(rng.uniform(1e-5, 4e-4, len(idx)), index=idx)
    dv.iloc[30] = 0.0  # exactly-flat close
    x, y = har_design(_feature_frame(dv), horizon=22, log_space=True)
    assert np.isfinite(x.dropna().to_numpy()).all()
    assert np.isfinite(y.dropna().to_numpy()).all()


def test_har_design_log_space_and_iv() -> None:
    idx = pd.bdate_range("2010-01-01", periods=80)
    rng = np.random.default_rng(3)
    dv = pd.Series(rng.uniform(1e-5, 4e-4, len(idx)), index=idx)
    df = _feature_frame(dv)

    x_lvl, y_lvl = har_design(df, horizon=22, use_iv=True)
    assert list(x_lvl.columns) == ["rv_cc_trail_1", "rv_cc_trail_5", "rv_cc_trail_22", "iv30"]

    x_log, y_log = har_design(df, horizon=22, use_iv=True, log_space=True)
    t = 50  # target defined only while t + 22 < len(df)
    np.testing.assert_allclose(x_log.iloc[t], np.log(x_lvl.iloc[t].to_numpy()), rtol=1e-12)
    assert y_log.iloc[t] == pytest.approx(np.log(y_lvl.iloc[t]), rel=1e-12)
