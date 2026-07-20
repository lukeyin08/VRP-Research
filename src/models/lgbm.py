"""LightGBM on the same information set as HAR-IV plus jump/range features.

Included specifically to test whether nonlinear ML beats HAR out of sample; in
this literature it usually does not by much, and either outcome is reported.
One FIXED hyperparameter configuration, chosen a priori at modest complexity
and logged - never tuned on evaluation data. Trained on log-RV (skewed target)
with Duan smearing on the retransform, same as HAR-log. Uses the native
lightgbm API (no scikit-learn dependency).

Feature note: only long-history, lag-safe features. Parkinson trails are valid
over the whole sample (high/low only); open-dependent estimators (GK/RS/YZ)
are excluded because pre-2008 Yahoo opens are synthetic.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import SEED
from src.evaluation.walkforward import WalkForwardSpec, walkforward_model
from src.models.har import LOG_FLOOR_ANN

LGBM_FEATURES = (
    "rv_cc_trail_1",
    "rv_cc_trail_5",
    "rv_cc_trail_22",
    "bv_trail_22",
    "jump_22",
    "iv30",
    "rv_pk_trail_5",
    "rv_pk_trail_22",
)

# Single fixed configuration (logged); never tuned on evaluation data.
LGBM_PARAMS: dict[str, object] = {
    "objective": "regression",
    "num_boost_round": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 50,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "seed": SEED,
    "verbosity": -1,
}


class NativeLGBM:
    """Minimal fit/predict wrapper over lightgbm's native training API."""

    def __init__(self) -> None:
        self.booster: lgb.Booster | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> NativeLGBM:
        params = {k: v for k, v in LGBM_PARAMS.items() if k != "num_boost_round"}
        self.booster = lgb.train(
            params,
            lgb.Dataset(x, label=y),
            num_boost_round=int(str(LGBM_PARAMS["num_boost_round"])),
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.booster is not None
        return np.asarray(self.booster.predict(x), dtype=float)


def lgbm_design(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    x = df[list(LGBM_FEATURES)].copy()
    xv = np.log(np.clip(x.to_numpy(dtype=float), LOG_FLOOR_ANN, None))
    x = pd.DataFrame(xv, index=x.index, columns=x.columns)
    y_raw = df[f"target_rv_{horizon}"]
    y = pd.Series(
        np.log(np.clip(y_raw.to_numpy(dtype=float), LOG_FLOOR_ANN, None)),
        index=y_raw.index,
        name=y_raw.name,
    )
    return x, y


def lgbm_forecast(
    df: pd.DataFrame, eval_index: pd.DatetimeIndex, horizon: int = 22, refit_every: int = 22
) -> pd.Series:
    x, y = lgbm_design(df, horizon)
    spec = WalkForwardSpec(horizon=horizon, refit_every=refit_every, min_train=756)
    f = walkforward_model(x, y, eval_index, spec, model_factory=NativeLGBM, log_space=True)
    return f.rename("lightgbm")


def lgbm_feature_importance(df: pd.DataFrame, horizon: int, train_end: pd.Timestamp) -> pd.Series:
    """Gain importance from a single fit on the dev training set (diagnostic only)."""
    x, y = lgbm_design(df, horizon)
    m = pd.concat([x, y], axis=1).loc[:train_end].dropna()
    model = NativeLGBM()
    model.fit(m[list(LGBM_FEATURES)].to_numpy(), m[str(y.name)].to_numpy())
    assert model.booster is not None
    imp = pd.Series(
        model.booster.feature_importance(importance_type="gain"),
        index=list(LGBM_FEATURES),
        name="gain_importance",
    )
    return imp.sort_values(ascending=False).round(1)
