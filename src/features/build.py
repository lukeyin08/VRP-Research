"""Assemble the processed feature/target frame from raw files.

Output: data/processed/features.parquet, one row per joint trading day with
- spx OHLC, vix (and term-structure closes where they exist)
- daily variance estimators (cc, parkinson, garman_klass, rogers_satchell, yang_zhang)
- trailing annualized RV at 1/5/22 days (information available at close t)
- forward annualized RV targets at 1/5/22 days (strictly t+1 onward)
- implied variance: iv30 = (VIX/100)^2 and term-structure equivalents
- ex-post VRP: iv30 - target_rv_22

`assemble_features` is pure (frames in, frame out) so tests can drive it with
synthetic data; `build_features` wires it to the raw files and the parquet cache.
"""

from __future__ import annotations

import pandas as pd

from src.config import HORIZONS, PROCESSED_DIR
from src.data.align import AlignmentReport, build_joint
from src.data.raw_loaders import load_cboe_index, load_optional_cboe, load_stooq, load_yahoo_ohlcv
from src.data.validate import Issue, assert_clean, check_close, check_ohlc, check_variance
from src.features import rv


def assemble_features(
    spx: pd.DataFrame,
    vix: pd.DataFrame,
    term: dict[str, pd.DataFrame | None],
    stooq: pd.DataFrame | None = None,
    strict: bool = True,
) -> tuple[pd.DataFrame, AlignmentReport, list[Issue]]:
    issues: list[Issue] = []
    # SPX: full OHLC checks (we consume the whole bar for range estimators).
    issues += check_ohlc(spx, "gspc", lo_bound=100, hi_bound=50000)
    # VIX family: we consume closes only; see check_close docstring.
    issues += check_close(vix, "vix", lo_bound=4, hi_bound=200)
    for name, df_t in term.items():
        if df_t is not None:
            issues += check_close(df_t, name, lo_bound=4, hi_bound=200)
    if strict:
        assert_clean(issues)

    df, rep = build_joint(spx, vix, term, spx_check=stooq)

    o, h, lo, c = (df[f"spx_{k}"] for k in ("open", "high", "low", "close"))
    df["dv_cc"] = rv.cc_var(c)
    df["dv_parkinson"] = rv.parkinson_var(h, lo)
    df["dv_garman_klass"] = rv.garman_klass_var(o, h, lo, c)
    df["dv_rogers_satchell"] = rv.rogers_satchell_var(o, h, lo, c)
    df["dv_yang_zhang22"] = rv.yang_zhang_var(o, h, lo, c, window=22)

    # Yahoo ^GSPC opens are synthetic (== prior close) on most days before ~2008.
    # cc and Parkinson are unaffected; Garman-Klass / Rogers-Satchell / Yang-Zhang
    # depend on the open and are only trustworthy where this flag is mostly False.
    # Modeling phases must mask open-dependent features on the synthetic-open era.
    df["spx_open_synthetic"] = o.round(4) == c.shift(1).round(4)

    for name in ("cc", "parkinson", "garman_klass", "rogers_satchell"):
        issues += check_variance(df[f"dv_{name}"].dropna(), f"dv_{name}")
    if strict:
        assert_clean(issues)

    # trailing RV (features): defined on information through close of t
    for hz in HORIZONS:
        df[f"rv_cc_trail_{hz}"] = rv.realized_var_trailing(df["dv_cc"], hz)
        df[f"rv_gk_trail_{hz}"] = rv.realized_var_trailing(df["dv_garman_klass"], hz)
        df[f"rv_pk_trail_{hz}"] = rv.realized_var_trailing(df["dv_parkinson"], hz)

    # signed daily log return (needed by GARCH and bipower variation)
    df["ret_cc"] = rv.log_returns(c)
    for hz in (5, 22):
        df[f"bv_trail_{hz}"] = rv.bipower_var_trailing(df["ret_cc"], hz)
        df[f"jump_{hz}"] = rv.jump_component(df[f"rv_cc_trail_{hz}"], df[f"bv_trail_{hz}"])

    # forward RV (targets): strictly t+1 .. t+h, payoff-relevant cc estimator
    for hz in HORIZONS:
        df[f"target_rv_{hz}"] = rv.realized_var_forward(df["dv_cc"], hz)

    # implied variance in decimal units
    df["iv30"] = (df["vix_close"] / 100.0) ** 2
    for name, col in (("vix9d", "iv9d"), ("vix3m", "iv3m"), ("vix6m", "iv6m")):
        if f"{name}_close" in df.columns:
            df[col] = (df[f"{name}_close"] / 100.0) ** 2

    df["vrp_expost"] = df["iv30"] - df["target_rv_22"]
    return df, rep, issues


def build_features(strict: bool = True) -> tuple[pd.DataFrame, AlignmentReport, list[Issue]]:
    spx = load_yahoo_ohlcv("gspc")
    vix = load_cboe_index("vix")
    term = {name: load_optional_cboe(name) for name in ("vix9d", "vix3m", "vix6m")}
    stooq = load_stooq()

    df, rep, issues = assemble_features(spx, vix, term, stooq, strict=strict)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / "features.parquet")
    return df, rep, issues


def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run python -m src.run_phase1")
    return pd.read_parquet(path)
