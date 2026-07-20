"""RV estimators against a hand-computed single-day example and exact window math."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.features import rv

# One day: prev close 99, open 100, high 104, low 98, close 102.
PREV_C, OP, H, L, C = 99.0, 100.0, 104.0, 98.0, 102.0


def _frame() -> pd.DataFrame:
    idx = pd.to_datetime(["2020-01-02", "2020-01-03"])
    return pd.DataFrame(
        {"open": [PREV_C, OP], "high": [PREV_C, H], "low": [PREV_C, L], "close": [PREV_C, C]},
        index=idx,
    )


def test_cc_hand_computed() -> None:
    df = _frame()
    got = rv.cc_var(df["close"]).iloc[1]
    expected = math.log(C / PREV_C) ** 2
    assert got == pytest.approx(expected, rel=1e-12)
    assert got == pytest.approx(0.0008911994088162194, rel=1e-9)


def test_parkinson_hand_computed() -> None:
    df = _frame()
    got = rv.parkinson_var(df["high"], df["low"]).iloc[1]
    expected = math.log(H / L) ** 2 / (4 * math.log(2))
    assert got == pytest.approx(expected, rel=1e-12)
    assert got == pytest.approx(0.001273590587787223, rel=1e-9)


def test_garman_klass_hand_computed() -> None:
    df = _frame()
    got = rv.garman_klass_var(df["open"], df["high"], df["low"], df["close"]).iloc[1]
    expected = 0.5 * math.log(H / L) ** 2 - (2 * math.log(2) - 1) * math.log(C / OP) ** 2
    assert got == pytest.approx(expected, rel=1e-12)
    assert got == pytest.approx(0.0016140884158007947, rel=1e-9)


def test_rogers_satchell_hand_computed() -> None:
    df = _frame()
    got = rv.rogers_satchell_var(df["open"], df["high"], df["low"], df["close"]).iloc[1]
    expected = math.log(H / C) * math.log(H / OP) + math.log(L / C) * math.log(L / OP)
    assert got == pytest.approx(expected, rel=1e-12)
    assert got == pytest.approx(0.0015698072417271618, rel=1e-9)


def test_range_estimators_nonnegative_random() -> None:
    rng = np.random.default_rng(1990)
    n = 500
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.roll(c, 1) * np.exp(rng.normal(0, 0.003, n))
    o[0] = 100.0
    hi = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.005, n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.005, n)))
    idx = pd.bdate_range("2000-01-03", periods=n)
    df = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c}, index=idx)
    for est in (
        rv.parkinson_var(df["high"], df["low"]),
        rv.garman_klass_var(df["open"], df["high"], df["low"], df["close"]),
        rv.rogers_satchell_var(df["open"], df["high"], df["low"], df["close"]),
    ):
        assert (est.dropna() >= -1e-15).all()


def test_yang_zhang_degenerate_case() -> None:
    # Constant overnight and open-to-close returns: both variance terms are 0,
    # so YZ = (1-k) * mean(RS) exactly.
    n = 30
    idx = pd.bdate_range("2020-01-01", periods=n)
    o = np.empty(n)
    c = np.empty(n)
    o[0] = 100.0
    for i in range(n):
        c[i] = o[i] * 1.008  # constant open-to-close ratio
        if i + 1 < n:
            o[i + 1] = c[i] * 1.002  # constant overnight ratio
    hi = np.maximum(o, c) * 1.004
    lo = np.minimum(o, c) * 0.996
    df = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c}, index=idx)
    window = 10
    yz = rv.yang_zhang_var(df["open"], df["high"], df["low"], df["close"], window=window)
    rs = rv.rogers_satchell_var(df["open"], df["high"], df["low"], df["close"])
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    expected = (1 - k) * rs.rolling(window).mean()
    # overnight ratio is constant only from day 2 on; compare where both defined
    got, exp = yz.iloc[window + 2 :], expected.iloc[window + 2 :]
    assert np.allclose(got, exp, rtol=1e-10)


def test_bipower_hand_computed() -> None:
    idx = pd.bdate_range("2020-01-01", periods=3)
    r = pd.Series([0.01, -0.02, 0.015], index=idx)
    bv = rv.bipower_var_trailing(r, horizon=3)
    # window of 3 days has 2 adjacent products: |r2||r1| + |r3||r2|
    expected = (math.pi / 2) * (0.02 * 0.01 + 0.015 * 0.02) * (252 / 2)
    assert bv.iloc[2] == pytest.approx(expected, rel=1e-12)
    assert bv.iloc[:2].isna().all()


def test_jump_component_nonnegative_and_hand_computed() -> None:
    idx = pd.bdate_range("2020-01-01", periods=3)
    rv_t = pd.Series([0.05, 0.02, 0.04], index=idx)
    bv_t = pd.Series([0.03, 0.05, 0.04], index=idx)
    j = rv.jump_component(rv_t, bv_t)
    assert j.tolist() == pytest.approx([0.02, 0.0, 0.0], rel=1e-12)


def test_forward_and_trailing_window_arithmetic() -> None:
    idx = pd.bdate_range("2021-03-01", periods=4)
    dv = pd.Series([1e-4, 2e-4, 3e-4, 4e-4], index=idx)

    fwd = rv.realized_var_forward(dv, horizon=2)
    assert fwd.iloc[0] == pytest.approx(126 * (2e-4 + 3e-4), rel=1e-12)  # t+1, t+2
    assert fwd.iloc[1] == pytest.approx(126 * (3e-4 + 4e-4), rel=1e-12)
    assert fwd.iloc[2:].isna().all()  # not enough future data: NaN, never partial

    trail = rv.realized_var_trailing(dv, horizon=2)
    assert np.isnan(trail.iloc[0])
    assert trail.iloc[1] == pytest.approx(126 * (1e-4 + 2e-4), rel=1e-12)  # t-1, t
    assert trail.iloc[3] == pytest.approx(126 * (3e-4 + 4e-4), rel=1e-12)
