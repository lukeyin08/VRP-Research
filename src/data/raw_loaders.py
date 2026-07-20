"""Parse raw files from data/raw into tidy frames.

These functions are pure file-parsers: no network access anywhere in src/.
Raw data is fetched once by scripts/download_raw.py and cached on disk.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import RAW_DIR


class RawDataError(RuntimeError):
    pass


def _postprocess(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if df.index.has_duplicates:
        raise RawDataError(f"{name}: duplicate dates survived dedup")
    if not df.index.is_monotonic_increasing:
        raise RawDataError(f"{name}: dates not sorted")
    return df


def load_cboe_index(name: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """CBOE daily OHLC history (VIX family). Columns: open, high, low, close."""
    path = raw_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run scripts/download_raw.py")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"date": "date"})
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df = df.set_index("date")[["open", "high", "low", "close"]].astype(float)
    return _postprocess(df, name)


def load_yahoo_ohlcv(name: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Yahoo daily OHLCV as written by download_raw.py."""
    path = raw_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run scripts/download_raw.py")
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    df = df[["open", "high", "low", "close", "adj_close", "volume"]].astype(float)
    return _postprocess(df, name)


def load_stooq(name: str = "spx_stooq", raw_dir: Path = RAW_DIR) -> pd.DataFrame | None:
    """Stooq daily OHLCV; optional (cross-check only). Returns None if absent."""
    path = raw_dir / f"{name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df.set_index("date")[cols].astype(float)
    return _postprocess(df, name)


def load_optional_cboe(name: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame | None:
    try:
        return load_cboe_index(name, raw_dir)
    except FileNotFoundError:
        return None
