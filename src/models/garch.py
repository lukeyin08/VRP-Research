"""GARCH-family forecasts with correct multi-day aggregation.

Aggregation: the h-day annualized variance forecast is (252/h) * sum of the h
per-day conditional variance forecasts from the model recursion - NOT the 1-day
forecast times h, which is wrong whenever current variance differs from the
model's long-run level (mean reversion). GARCH/GJR use analytic multi-step
recursions; EGARCH has no analytic multi-step form, so it uses seeded
simulation.

Information barrier: parameters are estimated on returns through the refit
date; between refits the parameters are frozen while the conditional-variance
state updates with each day's return (data passed to the model never extends
past the block being forecast). GARCH fits on returns have no overlapping-
target problem, so training may use data through t itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from arch.univariate import EGARCH, GARCH, ConstantMean, Normal, StudentsT
from arch.univariate.base import ARCHModelResult
from arch.univariate.volatility import VolatilityProcess

from src.config import SEED, TRADING_DAYS_PER_YEAR


@dataclass(frozen=True)
class GarchSpec:
    vol: str
    o: int
    dist: str
    method: Literal["analytic", "simulation"]


GARCH_SPECS: dict[str, GarchSpec] = {
    "garch_n": GarchSpec("garch", 0, "normal", "analytic"),
    "gjr_n": GarchSpec("garch", 1, "normal", "analytic"),
    "gjr_t": GarchSpec("garch", 1, "t", "analytic"),
    "egarch_n": GarchSpec("egarch", 1, "normal", "simulation"),
}


def spec_params(name: str) -> dict[str, object]:
    return asdict(GARCH_SPECS[name])


def _make_model(returns_pct: pd.Series, spec: GarchSpec, seed: int) -> ConstantMean:
    dist = StudentsT(seed=seed) if spec.dist == "t" else Normal(seed=seed)
    vol: VolatilityProcess
    if spec.vol == "egarch":
        vol = EGARCH(p=1, o=spec.o, q=1)
    else:
        vol = GARCH(p=1, o=spec.o, q=1)
    return ConstantMean(returns_pct, volatility=vol, distribution=dist)


def garch_forecasts(
    returns: pd.Series,
    eval_index: pd.DatetimeIndex,
    name: str,
    horizons: tuple[int, ...] = (1, 5, 22),
    refit_every: int = 22,
    min_train: int = 1250,
    simulations: int = 200,
    max_horizon: int = 22,
    block_offset: int = 0,
) -> dict[int, pd.Series]:
    """Walk-forward forecasts of annualized RV at each horizon, indexed at t.

    `block_offset` keys the per-block simulation seed to the absolute block
    index so chunked/resumed runs are bit-identical to a single pass.
    """
    spec = GARCH_SPECS[name]
    r = (returns.dropna() * 100.0).rename("ret_pct")  # arch prefers % units
    full_index = r.index
    buf = {h: np.full(len(eval_index), np.nan) for h in horizons}
    eval_pos = {t: i for i, t in enumerate(eval_index)}

    blocks = [eval_index[i : i + refit_every] for i in range(0, len(eval_index), refit_every)]
    for rel_bi, block in enumerate(blocks):
        bi = block_offset + rel_bi
        refit_date, block_end = block[0], block[-1]
        fit_data = r.loc[:refit_date]
        if len(fit_data) < min_train:
            continue
        am_fit = _make_model(fit_data, spec, seed=SEED + bi)
        res: ARCHModelResult = am_fit.fit(disp="off", show_warning=False)
        if not np.isfinite(res.params.to_numpy()).all():
            continue
        # frozen params, state updated through each block date; data never
        # extends beyond the block being forecast
        am_fc = _make_model(r.loc[:block_end], spec, seed=SEED + bi)
        fixed = am_fc.fix(res.params, first_obs=None)
        fc = fixed.forecast(
            horizon=max_horizon,
            start=refit_date,
            method=spec.method,
            simulations=simulations,
            reindex=False,
        )
        var = fc.variance  # %^2 daily units; row t = forecast made at close t
        for t in block:
            if t not in var.index or t not in full_index:
                continue
            row = var.loc[t].to_numpy(dtype=float)
            for h in horizons:
                buf[h][eval_pos[t]] = row[:h].sum() * (TRADING_DAYS_PER_YEAR / h) / 1e4
    return {h: pd.Series(buf[h], index=eval_index, name=name) for h in horizons}
