"""Data validation: hard assertions about anything we compute on.

Checks: unique sorted dates, weekday-only, calendar gaps, positive prices,
OHLC consistency, plausible index levels, no negative variances. Severity
"error" fails the run (and CI); "warn"/"info" are reported but tolerated.

Run directly:  python -m src.data.validate
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_EPS = 1e-9


class ValidationError(RuntimeError):
    pass


@dataclass
class Issue:
    severity: str  # "error" | "warn" | "info"
    name: str
    code: str
    message: str
    examples: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        ex = f" e.g. {self.examples[:3]}" if self.examples else ""
        return f"[{self.severity.upper():5s}] {self.name}: {self.code}: {self.message}{ex}"


def _dates(idx: pd.Index, n: int = 3) -> list[str]:
    return [str(pd.Timestamp(d).date()) for d in idx[:n]]


def check_index(df: pd.DataFrame, name: str) -> list[Issue]:
    issues: list[Issue] = []
    idx = pd.DatetimeIndex(df.index)
    if idx.has_duplicates:
        dups = idx[idx.duplicated()]
        issues.append(
            Issue("error", name, "dup_dates", f"{len(dups)} duplicate dates", _dates(dups))
        )
    if not idx.is_monotonic_increasing:
        issues.append(Issue("error", name, "unsorted", "index not sorted"))
    wknd = idx[idx.dayofweek >= 5]
    if len(wknd):
        issues.append(Issue("error", name, "weekend", f"{len(wknd)} weekend dates", _dates(wknd)))
    # business-day gaps (holidays produce small gaps; big ones are suspicious)
    bgaps = np.busday_count(
        idx[:-1].values.astype("datetime64[D]"), idx[1:].values.astype("datetime64[D]")
    )
    big = np.where(bgaps > 3)[0]
    if len(big):
        sev = "error" if (bgaps > 10).any() else "warn"
        issues.append(
            Issue(
                sev,
                name,
                "calendar_gap",
                f"{len(big)} gaps > 3 business days (max {int(bgaps.max())})",
                [f"{idx[i].date()}->{idx[i + 1].date()}" for i in big[:3]],
            )
        )
    return issues


def check_ohlc(df: pd.DataFrame, name: str, lo_bound: float, hi_bound: float) -> list[Issue]:
    issues = check_index(df, name)
    o, h, lo, c = (df[k] for k in ("open", "high", "low", "close"))
    nonpos = df[(df[["open", "high", "low", "close"]] <= 0).any(axis=1)]
    if len(nonpos):
        issues.append(
            Issue("error", name, "nonpositive", f"{len(nonpos)} rows <= 0", _dates(nonpos.index))
        )
    bad_h = df[(h + _EPS < o) | (h + _EPS < c) | (h + _EPS < lo)]
    bad_l = df[(lo - _EPS > o) | (lo - _EPS > c)]
    if len(bad_h):
        issues.append(
            Issue(
                "error",
                name,
                "high_violated",
                f"{len(bad_h)} rows high < max(o,c,l)",
                _dates(bad_h.index),
            )
        )
    if len(bad_l):
        issues.append(
            Issue(
                "error",
                name,
                "low_violated",
                f"{len(bad_l)} rows low > min(o,c)",
                _dates(bad_l.index),
            )
        )
    oob = df[(c < lo_bound) | (c > hi_bound)]
    if len(oob):
        issues.append(
            Issue(
                "warn",
                name,
                "range",
                f"{len(oob)} closes outside [{lo_bound}, {hi_bound}]",
                _dates(oob.index),
            )
        )
    ratio = c / c.shift(1)
    with np.errstate(invalid="ignore"):  # nonpositive prices are flagged separately above
        r = pd.Series(np.log(ratio), index=ratio.index).dropna()
    wild = r[r.abs() > 0.35]
    if len(wild):
        issues.append(
            Issue("warn", name, "wild_return", f"{len(wild)} |log ret| > 35%", _dates(wild.index))
        )
    # informational: synthetic-open detection (open == previous close) and zero-range days
    open_eq_prev = float((o.round(4) == c.shift(1).round(4)).mean())
    zero_range = float((h.round(6) == lo.round(6)).mean())
    issues.append(
        Issue("info", name, "open_eq_prevclose", f"share open==prev close: {open_eq_prev:.3f}")
    )
    issues.append(Issue("info", name, "zero_range_days", f"share high==low: {zero_range:.4f}"))
    return issues


def check_close(df: pd.DataFrame, name: str, lo_bound: float, hi_bound: float) -> list[Issue]:
    """Validation for series where only the CLOSE is consumed (the VIX family).

    CBOE's published index histories contain rows whose open/high/low fields
    violate OHLC relations (e.g. VIX 1992-02-11 has open 19.24 > high 18.57;
    VIX6M 2019-07-05 has high < low). We never use those fields, so full index
    and close checks stay hard errors while OHLC-field inconsistencies are
    downgraded to a counted warning. Documented in the README data notes.
    """
    issues = check_index(df, name)
    c = df["close"]
    nonpos = c[c <= 0]
    if len(nonpos):
        issues.append(
            Issue("error", name, "nonpositive", f"{len(nonpos)} closes <= 0", _dates(nonpos.index))
        )
    oob = c[(c < lo_bound) | (c > hi_bound)]
    if len(oob):
        issues.append(
            Issue(
                "warn",
                name,
                "range",
                f"{len(oob)} closes outside [{lo_bound}, {hi_bound}]",
                _dates(oob.index),
            )
        )
    if {"open", "high", "low"}.issubset(df.columns):
        o, h, lo = df["open"], df["high"], df["low"]
        bad = df[
            (h + _EPS < o) | (h + _EPS < c) | (h + _EPS < lo) | (lo - _EPS > o) | (lo - _EPS > c)
        ]
        if len(bad):
            issues.append(
                Issue(
                    "warn",
                    name,
                    "ohlc_fields_inconsistent",
                    f"{len(bad)} rows with inconsistent (unused) o/h/l fields",
                    _dates(bad.index),
                )
            )
    return issues


def check_variance(s: pd.Series, name: str) -> list[Issue]:
    issues: list[Issue] = []
    neg = s[s < -_EPS]
    if len(neg):
        issues.append(
            Issue(
                "error", name, "negative_variance", f"{len(neg)} negative values", _dates(neg.index)
            )
        )
    return issues


def assert_clean(issues: list[Issue]) -> None:
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        raise ValidationError("\n".join(str(i) for i in errors))


def report(issues: list[Issue]) -> str:
    order = {"error": 0, "warn": 1, "info": 2}
    return "\n".join(str(i) for i in sorted(issues, key=lambda i: order[i.severity]))


def main() -> int:
    from src.data.raw_loaders import load_cboe_index, load_optional_cboe, load_yahoo_ohlcv

    issues: list[Issue] = []
    issues += check_ohlc(load_yahoo_ohlcv("gspc"), "gspc", lo_bound=100, hi_bound=50000)
    issues += check_close(load_cboe_index("vix"), "vix", lo_bound=4, hi_bound=200)
    for name in ("vix9d", "vix3m", "vix6m"):
        df = load_optional_cboe(name)
        if df is not None:
            issues += check_close(df, name, lo_bound=4, hi_bound=200)
    print(report(issues))
    if any(i.severity == "error" for i in issues):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
