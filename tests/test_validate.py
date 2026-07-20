"""The validation module must actually catch corrupted data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.validate import (
    ValidationError,
    assert_clean,
    check_close,
    check_index,
    check_ohlc,
    check_variance,
)


def _clean_ohlc(n: int = 50) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    c = np.linspace(3000, 3100, n)
    return pd.DataFrame(
        {"open": c * 0.999, "high": c * 1.01, "low": c * 0.99, "close": c}, index=idx
    )


def _codes(issues: list) -> set[str]:
    return {i.code for i in issues if i.severity == "error"}


def test_clean_frame_passes() -> None:
    issues = check_ohlc(_clean_ohlc(), "x", lo_bound=100, hi_bound=50000)
    assert_clean(issues)  # must not raise


def test_duplicate_dates_flagged() -> None:
    df = _clean_ohlc()
    df = pd.concat([df, df.iloc[[10]]]).sort_index()
    assert "dup_dates" in _codes(check_index(df, "x"))


def test_weekend_dates_flagged() -> None:
    df = _clean_ohlc()
    df.index = pd.DatetimeIndex([*df.index[:-1], pd.Timestamp("2020-03-14")])  # a Saturday
    assert "weekend" in _codes(check_index(df.sort_index(), "x"))


def test_high_low_violation_flagged() -> None:
    df = _clean_ohlc()
    df.loc[df.index[5], "high"] = df.iloc[5]["low"] * 0.5  # high < low < close
    codes = _codes(check_ohlc(df, "x", lo_bound=100, hi_bound=50000))
    assert "high_violated" in codes
    with pytest.raises(ValidationError):
        assert_clean(check_ohlc(df, "x", lo_bound=100, hi_bound=50000))


def test_nonpositive_price_flagged() -> None:
    df = _clean_ohlc()
    df.loc[df.index[3], "close"] = -1.0
    assert "nonpositive" in _codes(check_ohlc(df, "x", lo_bound=100, hi_bound=50000))


def test_calendar_gap_flagged() -> None:
    df = _clean_ohlc(60)
    df = pd.concat([df.iloc[:20], df.iloc[45:]])  # 25-business-day hole
    codes = _codes(check_index(df, "x"))
    assert "calendar_gap" in codes


def test_check_close_downgrades_unused_ohlc_fields() -> None:
    """VIX-family series: broken o/h/l fields (which we never consume) must be
    a counted warning, while close-field problems stay hard errors."""
    idx = pd.bdate_range("1992-02-10", periods=3)
    df = pd.DataFrame(
        {  # middle row mimics CBOE's real VIX 1992-02-11: open > high
            "open": [19.0, 19.24, 18.0],
            "high": [19.5, 18.57, 18.5],
            "low": [18.5, 17.61, 17.8],
            "close": [19.1, 17.70, 18.2],
        },
        index=idx,
    )
    issues = check_close(df, "vix", lo_bound=4, hi_bound=200)
    assert_clean(issues)  # no errors
    assert any(i.code == "ohlc_fields_inconsistent" and i.severity == "warn" for i in issues)

    df.loc[idx[1], "close"] = -5.0
    with pytest.raises(ValidationError):
        assert_clean(check_close(df, "vix", lo_bound=4, hi_bound=200))


def test_negative_variance_flagged() -> None:
    s = pd.Series([1e-4, 2e-4, -1e-6], index=pd.bdate_range("2020-01-01", periods=3))
    assert "negative_variance" in _codes(check_variance(s, "dv"))
    assert _codes(check_variance(s.abs(), "dv")) == set()
