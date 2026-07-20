"""Loader parsing against literal raw-format fixtures (no network, no real files)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.raw_loaders import load_cboe_index, load_stooq, load_yahoo_ohlcv

CBOE_SAMPLE = """DATE,OPEN,HIGH,LOW,CLOSE
01/02/1990,17.240000,17.240000,17.240000,17.240000
01/03/1990,18.190000,18.190000,18.190000,18.190000
01/04/1990,19.220000,19.220000,19.220000,19.220000
"""

YAHOO_SAMPLE = """date,open,high,low,close,adj_close,volume
1990-01-02,353.400000,359.690000,351.980000,359.690000,359.690000,162070000
1990-01-03,359.690000,360.590000,357.890000,358.760000,358.760000,192330000
"""

STOOQ_SAMPLE = """Date,Open,High,Low,Close,Volume
1990-01-02,353.4,359.69,351.98,359.69,162070000
1990-01-03,359.69,360.59,357.89,358.76,192330000
"""


def test_cboe_loader(tmp_path: Path) -> None:
    (tmp_path / "vix.csv").write_text(CBOE_SAMPLE)
    df = load_cboe_index("vix", raw_dir=tmp_path)
    assert list(df.columns) == ["open", "high", "low", "close"]
    assert len(df) == 3
    assert str(df.index[0].date()) == "1990-01-02"
    assert df["close"].iloc[0] == pytest.approx(17.24)


def test_yahoo_loader(tmp_path: Path) -> None:
    (tmp_path / "gspc.csv").write_text(YAHOO_SAMPLE)
    df = load_yahoo_ohlcv("gspc", raw_dir=tmp_path)
    assert df["close"].iloc[1] == pytest.approx(358.76)
    assert df.index.is_monotonic_increasing


def test_stooq_loader(tmp_path: Path) -> None:
    (tmp_path / "spx_stooq.csv").write_text(STOOQ_SAMPLE)
    df = load_stooq(raw_dir=tmp_path)
    assert df is not None
    assert df["close"].iloc[0] == pytest.approx(359.69)


def test_stooq_loader_absent_returns_none(tmp_path: Path) -> None:
    assert load_stooq(raw_dir=tmp_path) is None


def test_missing_required_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_cboe_index("vix", raw_dir=tmp_path)


def test_duplicate_rows_deduped(tmp_path: Path) -> None:
    (tmp_path / "vix.csv").write_text(
        CBOE_SAMPLE + "01/04/1990,19.220000,19.220000,19.220000,19.220000\n"
    )
    df = load_cboe_index("vix", raw_dir=tmp_path)
    assert len(df) == 3
    assert not df.index.has_duplicates
