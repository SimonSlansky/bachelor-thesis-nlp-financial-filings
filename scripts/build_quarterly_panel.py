#!/usr/bin/env python3
"""Build the quarterly (10-Q, Q1–Q3) panel dataset.

Pipeline:
  1. Resolve tickers → CIK via SEC API
  2. Load company metadata (fiscal-year-end month)
  3. Extract XBRL facts from 10-Q filings (Q1–Q3 only)
  4. Save raw extraction → quarterly_financials.csv
  5. Impute: component computation, time-series fill
  6. Compute post-filing stock returns (63-day window)
  7. Derive financial ratios, winsorize
  8. Save final panel → quarterly_panel.csv
"""

import time
import pandas as pd

from config import (
    SAMPLE_SIZE, DATA_DIR, NUM_QUARTERS, MIN_VALID_QUARTERS,
    QUARTERLY_RETURN_WINDOW, MIN_TRADING_DAYS_QUARTERLY, REQUEST_SLEEP,
)
from sec_edgar import fetch_universe, load_company_metadata, extract_quarterly_facts
from returns import compute_returns
from panel import (
    compute_missing_components, impute_balance_sheet, add_financial_ratios,
    save_panel, QUARTERLY_COLS,
)


def main() -> None:
    # 1. Fetch SEC universe
    sec_df = fetch_universe(SAMPLE_SIZE)
    print(f"Universe: {len(sec_df)} companies (top {SAMPLE_SIZE or 'all'} by market cap)\n")

    # 2. Load metadata
    print("Loading company metadata …")
    companies = load_company_metadata(sec_df)
    print(f"  Loaded {len(companies)}/{len(sec_df)} companies\n")

    # 3. Extract XBRL
    print("Extracting quarterly XBRL data …")
    frames: list[pd.DataFrame] = []
    for ticker, meta in sorted(companies.items()):
        df = extract_quarterly_facts(meta, NUM_QUARTERS)
        if df.empty:
            print(f"  {ticker}: no data")
        elif len(df) < MIN_VALID_QUARTERS:
            print(f"  {ticker}: {len(df)} quarters (below minimum {MIN_VALID_QUARTERS})")
        else:
            frames.append(df)
            print(f"  {ticker}: {len(df)} quarters")
        time.sleep(REQUEST_SLEEP)

    if not frames:
        print("No data extracted — aborting.")
        return
    df_all = pd.concat(frames, ignore_index=True)
    print(f"\n  Total: {df_all['ticker'].nunique()} firms, {len(df_all)} firm-quarters\n")

    # 4. Save raw
    raw_path = DATA_DIR / "quarterly_financials.csv"
    df_all.to_csv(raw_path, index=False)
    print(f"  Raw data → {raw_path.name}\n")

    # 5. Impute
    print("Computing missing components …")
    df_all = compute_missing_components(df_all)
    print("Imputing balance-sheet gaps …")
    df_all = impute_balance_sheet(df_all, "quarter_end")

    # 6. Stock returns
    print("\nComputing stock returns (63-day window) …")
    df_all = compute_returns(
        df_all,
        date_col="quarter_end",
        return_col="return_next_q",
        window_days=QUARTERLY_RETURN_WINDOW,
        min_trading_days=MIN_TRADING_DAYS_QUARTERLY,
    )

    # 7. Financial ratios
    print("\nDeriving financial ratios …")
    df_all = add_financial_ratios(df_all, "quarter_end")

    # 8. Save
    print("\nSaving final panel …")
    save_panel(df_all, DATA_DIR / "quarterly_panel.csv", QUARTERLY_COLS)
    print("\nDone.")


if __name__ == "__main__":
    main()
