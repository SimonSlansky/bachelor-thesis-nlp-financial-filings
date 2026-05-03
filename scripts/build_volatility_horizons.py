"""Compute additional post-filing volatility windows (30d, 90d, 180d).

Standalone, idempotent: reads ``data/annual_panel.csv`` to obtain the
(ticker, year_end, filing_date) tuples already used by the main pipeline,
pulls yfinance closes once per ticker, and writes
``data/volatility_horizons.csv`` with one row per filing and three new
annualised-volatility columns.

The 365-day window stays in ``annual_panel.csv`` under ``vol_next_year``;
this script only adds the *short* horizons used in the robustness check
of Chapter~\\ref{ch:robustness} (post-filing decay of textual signals).

Usage:
    .venv\\Scripts\\python.exe scripts/build_volatility_horizons.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, POST_FILING_LAG_DAYS, REQUEST_SLEEP

ANNUALISATION_FACTOR = float(np.sqrt(252))

# (window_days, output_column, min_trading_days)
HORIZONS: list[tuple[int, str, int]] = [
    (5,   "vol_5d",   3),
    (10,  "vol_10d",  6),
    (30,  "vol_30d",  15),
    (90,  "vol_90d",  45),
    (180, "vol_180d", 90),
]

OUT_PATH = DATA_DIR / "volatility_horizons.csv"


def _vol_in_window(closes: pd.DataFrame, start: pd.Timestamp,
                   window_days: int, min_td: int) -> float:
    """Annualised SD of daily log returns over *window_days* calendar days."""
    end = start + pd.Timedelta(days=window_days)
    w = closes.loc[start:end]
    if len(w) < min_td:
        return float("nan")
    daily = np.log(w["Close"] / w["Close"].shift(1)).dropna()
    if len(daily) < min_td - 1:
        return float("nan")
    return float(daily.std() * ANNUALISATION_FACTOR)


def main() -> None:
    panel_path = DATA_DIR / "annual_panel.csv"
    if not panel_path.exists():
        print(f"Missing {panel_path}; run build_annual_panel.py first.",
              file=sys.stderr)
        sys.exit(1)

    panel = pd.read_csv(panel_path, parse_dates=["filing_date", "year_end"])
    panel = panel.dropna(subset=["filing_date"])
    print(f"Loaded {len(panel):,} filings, "
          f"{panel['ticker'].nunique()} tickers")

    min_date = panel["year_end"].min()
    longest = max(w for w, _, _ in HORIZONS)
    download_end = panel["filing_date"].max() + pd.Timedelta(
        days=POST_FILING_LAG_DAYS + longest + 10
    )

    rows: list[dict] = []
    tickers = sorted(panel["ticker"].unique())
    for i, ticker in enumerate(tickers, 1):
        prices = yf.download(
            ticker,
            start=min_date,
            end=download_end,
            progress=False,
            auto_adjust=True,
        )
        if prices.empty:
            print(f"  [{i:>3d}/{len(tickers)}] {ticker:<6s}  no price data")
            continue

        if isinstance(prices.columns, pd.MultiIndex):
            prices = prices.droplevel("Ticker", axis=1)
        if "Close" not in prices.columns:
            print(f"  [{i:>3d}/{len(tickers)}] {ticker:<6s}  no Close column")
            continue
        closes = prices[["Close"]].dropna()

        firm = panel.loc[panel["ticker"] == ticker]
        usable = 0
        for _, r in firm.iterrows():
            start = r["filing_date"] + pd.Timedelta(days=POST_FILING_LAG_DAYS)
            row = {"ticker": ticker, "year_end": r["year_end"]}
            any_ok = False
            for window_days, col, min_td in HORIZONS:
                v = _vol_in_window(closes, start, window_days, min_td)
                row[col] = v
                if not np.isnan(v):
                    any_ok = True
            if any_ok:
                usable += 1
            rows.append(row)

        print(f"  [{i:>3d}/{len(tickers)}] {ticker:<6s}  "
              f"{usable}/{len(firm)} filings with at least one valid window")
        time.sleep(REQUEST_SLEEP)

    out = pd.DataFrame(rows)
    if out.empty:
        print("No volatility horizons computed.", file=sys.stderr)
        sys.exit(1)

    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH.name}: {len(out):,} rows")
    for _, col, _ in HORIZONS:
        cov = out[col].notna().mean()
        print(f"  {col:10s}  coverage = {cov:6.1%}  "
              f"(median = {out[col].median():.4f})")


if __name__ == "__main__":
    main()
