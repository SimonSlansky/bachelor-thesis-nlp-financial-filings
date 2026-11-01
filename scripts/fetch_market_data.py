"""Download market data needed by Phase-1 controls (cached, resumable).

* data/prices/SPY.csv           — adjusted closes of the market proxy.
* data/prices_raw/{T}.csv       — split-adjusted Close + 'Stock Splits'
                                  (yfinance auto_adjust=False, actions=True),
                                  used to rebuild as-traded prices for
                                  market capitalisation.

Usage:  python scripts/fetch_market_data.py [--refresh]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, MARKET_TICKER, PRICE_START, PRICE_END
from volatility import load_closes, PRICE_RAW_DIR


def fetch_raw(ticker: str) -> bool:
    import yfinance as yf
    h = yf.Ticker(ticker).history(start=PRICE_START, end=PRICE_END,
                                  auto_adjust=False, actions=True)
    if h is None or h.empty or "Close" not in h:
        return False
    h.index = pd.DatetimeIndex(h.index).tz_localize(None)
    cols = ["Close", "Stock Splits"]
    if "Stock Splits" not in h:
        h["Stock Splits"] = 0.0
    h[cols].to_csv(PRICE_RAW_DIR / f"{ticker}.csv")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    m = load_closes(MARKET_TICKER, refresh=a.refresh)
    print(f"{MARKET_TICKER}: {len(m)} closes, {m.index.min().date()} – "
          f"{m.index.max().date()}  (data cut-off)")

    PRICE_RAW_DIR.mkdir(exist_ok=True)
    tickers = sorted(pd.read_csv(DATA_DIR / "annual_panel.csv")["ticker"].unique())
    miss = []
    for i, t in enumerate(tickers, 1):
        if (PRICE_RAW_DIR / f"{t}.csv").exists() and not a.refresh:
            continue
        ok = False
        for attempt in range(3):
            try:
                ok = fetch_raw(t)
                break
            except Exception as e:  # network hiccup
                time.sleep(2 ** attempt)
        if not ok:
            miss.append(t)
        if i % 50 == 0:
            print(f"  [{i}/{len(tickers)}]")
    print(f"raw prices: {len(tickers) - len(miss)}/{len(tickers)} ok; missing: {miss[:20]}")


if __name__ == "__main__":
    main()
