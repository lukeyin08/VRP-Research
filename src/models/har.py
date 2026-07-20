"""HAR-RV (Corsi 2009) and variants.

Regress future RV on daily, weekly (5d), and monthly (22d) trailing averages of
past RV. `log_space=True` runs the regression on ln(RV) (RV is right-skewed;
log usually improves fit) with Duan smearing on the retransform. `use_iv=True`
adds (VIX/100)^2 as a regressor (HAR-RV-IV): tests whether past RV carries
information beyond what the option market has already priced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.evaluation.walkforward import WalkForwardSpec, walkforward_ols

HAR_FEATURES = ("rv_cc_trail_1", "rv_cc_trail_5", "rv_cc_trail_22")

# Floor (annualized variance, ~= 0.32 vol pts) applied INSIDE the log transform
# only. SPX closed exactly or nearly flat on a handful of days (5 exact zeros in
# 1990-2026); log of a ~1e-12 daily variance is a -27 outlier in log space and a
# pure artifact of index rounding, not information. Levels regressions see the
# raw data. Disclosed in the README.
LOG_FLOOR_ANN = 1e-5


def har_design(
    df: pd.DataFrame, horizon: int, use_iv: bool = False, log_space: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix (info through close t) and target (t+1 .. t+horizon)."""
    cols = list(HAR_FEATURES) + (["iv30"] if use_iv else [])
    x = df[cols].copy()
    y = df[f"target_rv_{horizon}"].copy()
    if log_space:
        xv = np.clip(x.to_numpy(dtype=float), LOG_FLOOR_ANN, None)
        yv = np.clip(y.to_numpy(dtype=float), LOG_FLOOR_ANN, None)
        x = pd.DataFrame(np.log(xv), index=x.index, columns=x.columns)
        y = pd.Series(np.log(yv), index=y.index, name=y.name)
    return x, y


@dataclass
class HARForecaster:
    log_space: bool = False
    use_iv: bool = False
    spec: WalkForwardSpec = field(default_factory=lambda: WalkForwardSpec(horizon=22))

    @property
    def name(self) -> str:
        base = "har_log" if self.log_space else "har"
        return base + ("_iv" if self.use_iv else "")

    def forecast(self, df: pd.DataFrame, eval_index: pd.DatetimeIndex) -> pd.Series:
        x, y = har_design(df, self.spec.horizon, use_iv=self.use_iv, log_space=self.log_space)
        f = walkforward_ols(x, y, eval_index, self.spec, log_space=self.log_space)
        return f.rename(self.name)
