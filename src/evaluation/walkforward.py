"""Walk-forward engine with an explicit information barrier.

The barrier for h-step targets: a training row s (whose target covers
s+1 .. s+h) is usable for a forecast made at t only if its target window is
fully observed by t, i.e. position(s) <= position(t) - h. Using rows up to t-1
would leak the forecast's own target window into training.

Refits happen every `refit_every` eval dates (deterministic schedule, no future
dependence). Between refits, coefficients are frozen; features update daily.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from src.config import DEV_END, EVAL_START, HOLDOUT_START


class HoldoutViolation(RuntimeError):
    pass


def holdout_eval_index(full_index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """Final-holdout eval dates: from HOLDOUT_START, targets fully observable.

    This index is scored EXACTLY ONCE, in Phase 7. Re-running Phase 7
    reproduces that single evaluation; it must never inform model or strategy
    selection.
    """
    last_ok = len(full_index) - 1 - horizon
    idx = full_index[full_index >= pd.Timestamp(HOLDOUT_START)]
    idx = idx[full_index.get_indexer(idx) <= last_ok]
    return pd.DatetimeIndex(idx)


def dev_eval_index(full_index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """Development-period eval dates: within [EVAL_START, DEV_END] AND with the
    entire target window t+1..t+h inside the development period, so no dev
    forecast is ever scored against holdout-period realizations."""
    pos_end = int(full_index.searchsorted(pd.Timestamp(DEV_END), side="right")) - 1
    last_ok = pos_end - horizon
    idx = full_index[(full_index >= pd.Timestamp(EVAL_START))]
    idx = idx[full_index.get_indexer(idx) <= last_ok]
    return pd.DatetimeIndex(idx)


def holdout_guard(eval_index: pd.DatetimeIndex) -> None:
    """Phases 2-4 must call this. Raises if any eval date is in the holdout."""
    if len(eval_index) and eval_index.max() > pd.Timestamp(DEV_END):
        raise HoldoutViolation(
            f"eval index reaches {eval_index.max().date()} but the final holdout "
            f"starts {HOLDOUT_START}; it may only be evaluated once, in Phase 7"
        )


@dataclass
class WalkForwardSpec:
    horizon: int
    refit_every: int = 22  # eval dates between refits (~monthly)
    min_train: int = 252  # refuse to forecast on fewer usable training rows


class SupportsFitPredict(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> object: ...

    def predict(self, x: np.ndarray) -> np.ndarray: ...


def walkforward_model(
    x: pd.DataFrame,
    y: pd.Series,
    eval_index: pd.DatetimeIndex,
    spec: WalkForwardSpec,
    model_factory: Callable[[], SupportsFitPredict],
    log_space: bool = False,
) -> pd.Series:
    """Same information barrier as walkforward_ols for any fit/predict model.

    A fresh model instance is fitted at each refit on rows whose target windows
    are fully observed (position <= t - horizon); between refits the fitted
    model is frozen. Log-space retransformation applies Duan smearing estimated
    on training residuals.
    """
    full_index = x.index
    if not full_index.equals(y.index):
        raise ValueError("x and y must share an index")
    positions = full_index.get_indexer(eval_index)
    if (positions < 0).any():
        raise ValueError("eval dates missing from feature index")

    x_np = x.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=float)
    model: SupportsFitPredict | None = None
    smear = 1.0
    out = np.full(len(eval_index), np.nan)

    for i, pos in enumerate(positions):
        if i % spec.refit_every == 0:
            train_end = pos - spec.horizon + 1
            if train_end > 0:
                xt, yt = x_np[:train_end], y_np[:train_end]
                ok = ~(np.isnan(xt).any(axis=1) | np.isnan(yt))
                if ok.sum() >= spec.min_train:
                    m = model_factory()
                    m.fit(xt[ok], yt[ok])
                    model = m
                    if log_space:
                        resid = yt[ok] - np.asarray(m.predict(xt[ok]), dtype=float)
                        smear = float(np.mean(np.exp(resid)))
        if model is None:
            continue
        row = x_np[pos]
        if np.isnan(row).any():
            continue
        pred = float(np.asarray(model.predict(row.reshape(1, -1)), dtype=float)[0])
        out[i] = np.exp(pred) * smear if log_space else pred

    return pd.Series(out, index=eval_index)


def walkforward_ols(
    x: pd.DataFrame,
    y: pd.Series,
    eval_index: pd.DatetimeIndex,
    spec: WalkForwardSpec,
    log_space: bool = False,
) -> pd.Series:
    """Expanding-window OLS forecasts of y at each eval date.

    In log space the regression runs on ln(y) ~ ln-features (the caller passes
    already-logged x and y) and the retransformation applies Duan's smearing
    factor mean(exp(residual)) estimated on the training window, since a naive
    exp() retransform is biased low - which QLIKE punishes asymmetrically.
    """
    full_index = x.index
    if not full_index.equals(y.index):
        raise ValueError("x and y must share an index")
    positions = full_index.get_indexer(eval_index)
    if (positions < 0).any():
        raise ValueError("eval dates missing from feature index")

    x_np = x.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=float)
    n_feat = x_np.shape[1]

    beta: np.ndarray | None = None
    smear = 1.0
    out = np.full(len(eval_index), np.nan)

    for i, pos in enumerate(positions):
        if i % spec.refit_every == 0:
            train_end = pos - spec.horizon + 1  # exclusive slice bound: rows 0 .. pos-h
            if train_end > 0:
                xt, yt = x_np[:train_end], y_np[:train_end]
                ok = ~(np.isnan(xt).any(axis=1) | np.isnan(yt))
                if ok.sum() >= spec.min_train:
                    design = np.column_stack([np.ones(ok.sum()), xt[ok]])
                    coef, *_ = np.linalg.lstsq(design, yt[ok], rcond=None)
                    beta = coef
                    if log_space:
                        resid = yt[ok] - design @ coef
                        smear = float(np.mean(np.exp(resid)))
        if beta is None:
            continue
        row = x_np[pos]
        if np.isnan(row).any():
            continue
        pred = float(beta[0] + row @ beta[1 : n_feat + 1])
        out[i] = np.exp(pred) * smear if log_space else pred

    return pd.Series(out, index=eval_index)
