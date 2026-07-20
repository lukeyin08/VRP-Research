"""Performance and tail metrics. Sharpe is never reported alone: a short-vol
strategy's Sharpe overstates quality precisely because its risk hides in the
skew/kurtosis/CVaR columns reported next to it."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from scipy import stats

from src.config import TRADING_DAYS_PER_YEAR

EULER_GAMMA = 0.5772156649015329


def annualized_return(pnl: pd.Series) -> float:
    return float(pnl.mean() * TRADING_DAYS_PER_YEAR)


def annualized_vol(pnl: pd.Series) -> float:
    return float(pnl.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe(pnl: pd.Series) -> float:
    vol = annualized_vol(pnl)
    return annualized_return(pnl) / vol if vol > 0 else float("nan")


def drawdown_series(pnl: pd.Series) -> pd.Series:
    equity = pnl.cumsum()
    return equity - equity.cummax()


def max_drawdown(pnl: pd.Series) -> float:
    return float(drawdown_series(pnl).min())


def drawdown_episodes(pnl: pd.Series, top: int = 5) -> pd.DataFrame:
    dd = drawdown_series(pnl)
    flags = (dd < 0).to_numpy()
    dates = pd.DatetimeIndex(dd.index)
    episodes = []
    start: pd.Timestamp | None = None
    for t, flag in zip(dates, flags, strict=True):
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            seg = dd.loc[start:t]
            episodes.append(
                {
                    "start": start.date(),
                    "trough": pd.Timestamp(seg.idxmin()).date(),
                    "end": t.date(),
                    "depth": float(seg.min()),
                    "days": len(seg),
                }
            )
            start = None
    if start is not None:
        seg = dd.loc[start:]
        episodes.append(
            {
                "start": start.date(),
                "trough": pd.Timestamp(seg.idxmin()).date(),
                "end": None,
                "depth": float(seg.min()),
                "days": len(seg),
            }
        )
    out = pd.DataFrame(episodes)
    return out.sort_values("depth").head(top).reset_index(drop=True) if len(out) else out


def var_cvar(pnl: pd.Series, level: float) -> tuple[float, float]:
    """Historical daily VaR/CVaR at `level` (e.g. 0.95), reported as losses (<0)."""
    q = float(pnl.quantile(1.0 - level))
    tail = pnl[pnl <= q]
    return q, float(tail.mean()) if len(tail) else float("nan")


def perf_summary(pnl: pd.Series, label: str) -> dict[str, float | str]:
    v95, c95 = var_cvar(pnl, 0.95)
    v99, c99 = var_cvar(pnl, 0.99)
    return {
        "strategy": label,
        "ann_return": annualized_return(pnl),
        "ann_vol": annualized_vol(pnl),
        "sharpe": sharpe(pnl),
        "max_dd": max_drawdown(pnl),
        "skew": float(stats.skew(pnl.dropna())),
        "kurtosis": float(stats.kurtosis(pnl.dropna(), fisher=False)),
        "VaR95": v95,
        "CVaR95": c95,
        "VaR99": v99,
        "CVaR99": c99,
        "hit_rate": float((pnl > 0).mean()),
    }


@dataclass
class DeflatedSharpe:
    sharpe_ann: float
    dsr: float  # probability the true Sharpe exceeds the multiple-testing benchmark
    sr_benchmark_ann: float
    n_trials: int


def deflated_sharpe(pnl: pd.Series, n_trials: int) -> DeflatedSharpe:
    """Bailey & Lopez de Prado deflated Sharpe ratio.

    The benchmark SR0 is the expected maximum Sharpe among `n_trials`
    independent zero-skill trials with the variance of the SR estimator
    (approximated from this strategy's own T/skew/kurtosis); DSR = PSR(SR0) is
    the probability the observed Sharpe exceeds it given non-normality.
    """
    x = pnl.dropna()
    t_obs = len(x)
    sr_d = float(x.mean() / x.std(ddof=1))  # per-day Sharpe
    g3 = float(stats.skew(x))
    g4 = float(stats.kurtosis(x, fisher=False))
    var_sr = (1.0 - g3 * sr_d + (g4 - 1.0) / 4.0 * sr_d**2) / (t_obs - 1)
    var_sr = max(var_sr, 1e-12)
    n = max(n_trials, 2)
    sr0 = math.sqrt(var_sr) * (
        (1.0 - EULER_GAMMA) * stats.norm.ppf(1.0 - 1.0 / n)
        + EULER_GAMMA * stats.norm.ppf(1.0 - 1.0 / (n * math.e))
    )
    denom = math.sqrt(1.0 - g3 * sr_d + (g4 - 1.0) / 4.0 * sr_d**2)
    dsr = float(stats.norm.cdf((sr_d - sr0) * math.sqrt(t_obs - 1) / denom))
    ann = math.sqrt(TRADING_DAYS_PER_YEAR)
    return DeflatedSharpe(
        sharpe_ann=sr_d * ann, dsr=dsr, sr_benchmark_ann=sr0 * ann, n_trials=n_trials
    )
