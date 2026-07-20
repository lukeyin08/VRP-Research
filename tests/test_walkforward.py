"""The walk-forward engine must be provably unable to see the future.

Strategy: poison the series after (or inside) the forbidden region and assert
the forecasts are bit-identical (stronger than 'performance does not improve').
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.walkforward import (
    HoldoutViolation,
    WalkForwardSpec,
    dev_eval_index,
    holdout_guard,
    walkforward_ols,
)
from src.features import rv
from src.models.har import har_design


def _feature_frame(n: int = 400, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2001-01-01", periods=n)
    dv = pd.Series(rng.uniform(1e-5, 5e-4, n), index=idx)
    df = pd.DataFrame({"dv_cc": dv})
    for h in (1, 5, 22):
        df[f"rv_cc_trail_{h}"] = rv.realized_var_trailing(dv, h)
    df["target_rv_22"] = rv.realized_var_forward(dv, 22)
    df["iv30"] = 0.04
    return df


def test_forecasts_immune_to_future_poisoning() -> None:
    df = _feature_frame()
    x, y = har_design(df, horizon=22)
    eval_index = pd.DatetimeIndex(df.index[300:378])  # targets defined (n-22)
    spec = WalkForwardSpec(horizon=22, refit_every=5, min_train=100)

    base = walkforward_ols(x, y, eval_index, spec)

    t0_pos = 350
    x_p, y_p = x.copy(), y.copy()
    x_p.iloc[t0_pos + 1 :] = 777.0  # absurd features strictly after t0
    y_p.iloc[t0_pos + 1 :] = 777.0
    poisoned = walkforward_ols(x_p, y_p, eval_index, spec)

    upto_t0 = eval_index[df.index.get_indexer(eval_index) <= t0_pos]
    np.testing.assert_array_equal(base[upto_t0].to_numpy(), poisoned[upto_t0].to_numpy())


def test_training_excludes_unobservable_targets() -> None:
    """A row's target is unobservable at t if its window extends past t. Rows in
    positions (t-22, t] must NOT be trainable for the forecast at t."""
    df = _feature_frame()
    x, y = har_design(df, horizon=22)
    t0_pos = 350
    eval_index = pd.DatetimeIndex([df.index[t0_pos]])
    spec = WalkForwardSpec(horizon=22, refit_every=1, min_train=100)

    base = walkforward_ols(x, y, eval_index, spec)

    # poison targets whose windows are NOT fully observed at t0: no effect allowed
    y_p = y.copy()
    y_p.iloc[t0_pos - 21 : t0_pos + 1] = 1e6
    same = walkforward_ols(x, y_p, eval_index, spec)
    assert same.iloc[0] == base.iloc[0]

    # poison the last OBSERVABLE training row: forecast must change
    y_q = y.copy()
    y_q.iloc[t0_pos - 22] = 1e6
    changed = walkforward_ols(x, y_q, eval_index, spec)
    assert changed.iloc[0] != base.iloc[0]


def test_log_space_positive_and_finite() -> None:
    df = _feature_frame()
    x, y = har_design(df, horizon=22, log_space=True)
    eval_index = pd.DatetimeIndex(df.index[300:378])
    f = walkforward_ols(x, y, eval_index, WalkForwardSpec(22, 5, 100), log_space=True)
    got = f.dropna()
    assert len(got) > 50
    assert (got > 0).all()
    assert np.isfinite(got).all()


def test_holdout_guard_blocks_2019_plus() -> None:
    good = pd.bdate_range("2000-01-03", "2018-11-30")
    holdout_guard(pd.DatetimeIndex(good))  # must not raise
    bad = pd.bdate_range("2018-01-01", "2019-06-01")
    with pytest.raises(HoldoutViolation):
        holdout_guard(pd.DatetimeIndex(bad))


def test_dev_eval_index_never_overlaps_holdout() -> None:
    full = pd.DatetimeIndex(pd.bdate_range("1998-01-01", "2020-12-31"))
    idx = dev_eval_index(full, horizon=22)
    assert idx.min() >= pd.Timestamp("2000-01-03")
    # the last eval date's target window (22 trading days) ends by DEV_END
    last_pos = int(full.get_indexer(pd.DatetimeIndex([idx.max()]))[0])
    assert full[last_pos + 22] <= pd.Timestamp("2018-12-31")
    holdout_guard(idx)
