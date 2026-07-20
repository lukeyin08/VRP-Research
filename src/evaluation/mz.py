"""Mincer-Zarnowitz forecast-efficiency regressions with HAC standard errors.

Regress realized on forecast: y_t = a + b f_t + e_t. An efficient, unbiased
forecast has (a, b) = (0, 1); b well below 1 means the forecast systematically
overreacts (too dispersed). The joint Wald test uses HAC covariance because of
the overlapping-horizon autocorrelation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm


@dataclass
class MZResult:
    alpha: float
    beta: float
    r2: float
    p_joint: float  # H0: alpha = 0 and beta = 1
    n: int


def mz_regression(y: pd.Series, f: pd.Series, hac_lag: int = 44) -> MZResult:
    aligned = pd.concat([y.rename("y"), f.rename("forecast")], axis=1).dropna()
    x = sm.add_constant(aligned["forecast"])
    res = sm.OLS(aligned["y"], x).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lag})
    wald = res.wald_test("const = 0, forecast = 1", scalar=True)
    return MZResult(
        alpha=float(res.params["const"]),
        beta=float(res.params["forecast"]),
        r2=float(res.rsquared),
        p_joint=float(wald.pvalue),
        n=int(res.nobs),
    )
