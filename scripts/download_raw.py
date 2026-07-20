#!/usr/bin/env python3
"""Download all raw inputs for vrp-research into data/raw/.

Deliberately stdlib-only so it runs with any Python >= 3.10 before the project
environment exists. Every file's provenance (URL, sha256, rows, date range,
timestamp) is recorded in data/raw/manifest.json.

Sources
-------
- CBOE index histories (VIX, VIX9D, VIX3M, VIX6M): free daily OHLC CSVs from cdn.cboe.com.
- S&P 500 (^GSPC) and SPY daily OHLCV: Yahoo Finance v8 chart API (JSON -> CSV).
- S&P 500 (^SPX) from Stooq: independent second source used only to cross-check closes.

Usage: python3 scripts/download_raw.py [--force]
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) vrp-research/0.1 (academic use)"}

CBOE_BASE = "https://cdn.cboe.com/api/global/us_indices/daily_prices"
CBOE_FILES = {
    "vix.csv": f"{CBOE_BASE}/VIX_History.csv",
    "vix9d.csv": f"{CBOE_BASE}/VIX9D_History.csv",
    "vix3m.csv": f"{CBOE_BASE}/VIX3M_History.csv",
    "vix6m.csv": f"{CBOE_BASE}/VIX6M_History.csv",
}
# Required for the project to proceed; the others degrade gracefully.
REQUIRED = {"vix.csv", "gspc.csv"}


def fetch(url: str, tries: int = 3, timeout: int = 60) -> bytes:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed after {tries} tries: {url}: {last!r}")


def yahoo_daily_csv(symbol: str, start: str = "1985-01-01") -> bytes:
    """Fetch daily OHLCV from the Yahoo v8 chart API and render as CSV."""
    p1 = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    p2 = int(time.time())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}"
        f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit"
    )
    payload = json.loads(fetch(url).decode())
    result = payload["chart"]["result"][0]
    ts = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    rows = ["date,open,high,low,close,adj_close,volume"]
    for i, t in enumerate(ts):
        o, h, lo, c = (quote[k][i] for k in ("open", "high", "low", "close"))
        if None in (o, h, lo, c):
            continue  # halted / bad row: skip, never fill
        d = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
        a = adj[i] if adj and adj[i] is not None else c
        v = quote["volume"][i] or 0
        rows.append(f"{d},{o:.6f},{h:.6f},{lo:.6f},{c:.6f},{a:.6f},{int(v)}")
    return ("\n".join(rows) + "\n").encode()


def stooq_csv(symbol: str = "^spx") -> bytes:
    return fetch(f"https://stooq.com/q/d/l/?s={urllib.parse.quote(symbol)}&i=d")


def csv_meta(data: bytes) -> tuple[int, str, str]:
    lines = [ln for ln in data.decode(errors="replace").strip().splitlines() if ln.strip()]
    body = lines[1:]
    first = body[0].split(",")[0] if body else ""
    last = body[-1].split(",")[0] if body else ""
    return len(body), first, last


def main() -> int:
    force = "--force" in sys.argv
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW_DIR / "manifest.json"
    manifest: dict[str, dict[str, object]] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    jobs: list[tuple[str, str, object]] = [
        *[(name, url, None) for name, url in CBOE_FILES.items()],
        ("gspc.csv", "yahoo:v8-chart:^GSPC", lambda: yahoo_daily_csv("^GSPC")),
        ("spy.csv", "yahoo:v8-chart:SPY", lambda: yahoo_daily_csv("SPY", start="1993-01-01")),
        ("spx_stooq.csv", "https://stooq.com/q/d/l/?s=^spx&i=d", lambda: stooq_csv("^spx")),
    ]

    failures: list[str] = []
    for name, source, getter in jobs:
        out = RAW_DIR / name
        if out.exists() and not force:
            print(f"[skip] {name} exists (use --force to re-download)")
            continue
        try:
            data = getter() if callable(getter) else fetch(str(source))
            if len(data) < 200 or b"<html" in data[:400].lower():
                raise RuntimeError(f"response looks wrong ({len(data)} bytes)")
            out.write_bytes(data)
            nrows, first, last = csv_meta(data)
            manifest[name] = {
                "source": source,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "rows": nrows,
                "first_date": first,
                "last_date": last,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            print(f"[ok]   {name}: {nrows} rows, {first} .. {last}")
        except Exception as exc:
            failures.append(name)
            print(f"[FAIL] {name}: {exc}")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nmanifest -> {manifest_path}")

    missing_required = [n for n in REQUIRED if not (RAW_DIR / n).exists()]
    if missing_required:
        print(f"ERROR: required files missing: {missing_required}")
        return 1
    if failures:
        print(f"warning: optional files failed: {failures} (project degrades gracefully)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
