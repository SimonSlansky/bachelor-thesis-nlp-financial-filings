"""Compute short post-filing volatility windows (5d … 180d) and their lags.

Reads ``data/annual_panel.csv`` for the (ticker, fiscal_year, filing_date)
tuples, uses the cached prices from ``data/prices/`` (see volatility.py) and
writes ``data/volatility_horizons.csv`` with, for each horizon h,

* ``vol_{h}d``        — outcome window (starts 2 trading days after filing),
* ``lagged_vol_{h}d`` — same window after the fiscal-year t-1 filing,
                        truncated before the current filing; NaN if t-1 is
                        missing (no multi-year lags).

Usage:
    python scripts/build_volatility_horizons.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR
from volatility import load_closes, current_and_lagged, pre_filing_vol

# (window_days, column, minimum number of daily RETURNS)
# The 5-day window holds only 3–4 closes after the 2-trading-day skip, so it
# admits 3 returns; zero / degenerate SDs are set to NaN, never clipped.
HORIZONS: list[tuple[int, str, int]] = [
    (5,   "vol_5d",   3),
    (10,  "vol_10d",  5),
    (30,  "vol_30d",  14),
    (90,  "vol_90d",  44),
    (180, "vol_180d", 89),
]

OUT_PATH = DATA_DIR / "volatility_horizons.csv"


def main() -> None:
    panel_path = DATA_DIR / "annual_panel.csv"
    if not panel_path.exists():
        print(f"Missing {panel_path}; run build_annual_panel.py first.",
              file=sys.stderr)
        sys.exit(1)

    panel = pd.read_csv(panel_path, parse_dates=["filing_date", "year_end"])
    panel = panel.dropna(subset=["filing_date"])
    print(f"Loaded {len(panel):,} filings, {panel['ticker'].nunique()} tickers")

    # First-year filings dropped from the panel (asset-growth lag year).
    # They are used only as the t-1 anchor for lagged volatility, exactly
    # as returns.py does for the 365-day horizon.
    lag_path = DATA_DIR / "lag_filings.csv"
    lag_rows = (pd.read_csv(lag_path, parse_dates=["filing_date"])
                if lag_path.exists() else
                pd.DataFrame(columns=["ticker", "fiscal_year", "filing_date"]))
    if not lag_path.exists():
        print("  [warn] lag_filings.csv missing: first-year lags will be NaN")

    frames: list[pd.DataFrame] = []
    tickers = sorted(panel["ticker"].unique())
    for i, ticker in enumerate(tickers, 1):
        firm = panel.loc[panel["ticker"] == ticker,
                         ["ticker", "year_end", "fiscal_year", "filing_date"]]
        out = firm[["ticker", "year_end"]].copy()
        extra = lag_rows.loc[(lag_rows["ticker"] == ticker)
                             & ~lag_rows["fiscal_year"].isin(firm["fiscal_year"])]
        extra = extra.drop_duplicates("fiscal_year")
        extra.index = [f"_lag{j}" for j in range(len(extra))]
        firm_all = pd.concat([firm, extra])
        closes = load_closes(ticker)
        if closes is None:
            print(f"  [{i:>3d}/{len(tickers)}] {ticker:<6s}  no price data")
            continue
        for window_days, col, min_ret in HORIZONS:
            recs = current_and_lagged(closes, firm_all, window_days, min_ret)
            m = {r["_ix"]: r for r in recs}
            out[col] = [m[ix]["vol"] for ix in out.index]
            out[f"lagged_{col}"] = [m[ix]["lag_vol"] for ix in out.index]
        # Pre-filing volatility: 60 trading days ending the day before filing.
        out["pre_vol"] = [pre_filing_vol(closes, fd) for fd in firm["filing_date"]]
        frames.append(out)

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH.name}: {len(out):,} rows")
    for _, col, _ in HORIZONS:
        print(f"  {col:10s} coverage = {out[col].notna().mean():6.1%}  "
              f"lag coverage = {out['lagged_' + col].notna().mean():6.1%}  "
              f"(median = {out[col].median():.4f})")
    print(f"  pre_vol    coverage = {out['pre_vol'].notna().mean():6.1%}  "
          f"(median = {out['pre_vol'].median():.4f})")


if __name__ == "__main__":
    main()
