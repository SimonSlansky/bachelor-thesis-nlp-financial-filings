#!/usr/bin/env python3
"""Build the annual (10-K) panel dataset.

Pipeline:
  1. Resolve tickers → CIK via SEC API
  2. Load company metadata (fiscal-year-end month)
  3. Extract XBRL facts from 10-K filings
  4. Save raw extraction → annual_financials.csv
  5. Clean: filter transitions, resolve duplicates
  6. Impute: component computation, time-series fill
  7. Compute post-filing stock returns and volatility (365-day window)
  8. Derive financial ratios, winsorize, drop lag year
  9. Save final panel → annual_panel.csv
"""

import time
import pandas as pd

from config import (
    SAMPLE_SIZE, DATA_DIR, NUM_YEARS, MIN_VALID_YEARS,
    ANNUAL_RETURN_WINDOW, MIN_TRADING_DAYS_ANNUAL, REQUEST_SLEEP,
)
from sec_edgar import fetch_universe, load_company_metadata, extract_annual_facts
from returns import compute_returns_and_volatility
from panel import (
    compute_missing_components, impute_balance_sheet, add_financial_ratios,
    filter_transitions_and_duplicates, drop_earliest_year,
    save_panel, ANNUAL_COLS,
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
    print("Extracting annual XBRL data …")
    frames: list[pd.DataFrame] = []
    for ticker, meta in sorted(companies.items()):
        df = extract_annual_facts(meta, NUM_YEARS)
        if df.empty:
            print(f"  {ticker}: no data")
        elif len(df) < MIN_VALID_YEARS:
            print(f"  {ticker}: {len(df)} years (below minimum {MIN_VALID_YEARS})")
        else:
            frames.append(df)
            print(f"  {ticker}: {len(df)} years")
        time.sleep(REQUEST_SLEEP)

    if not frames:
        print("No data extracted — aborting.")
        return
    df_all = pd.concat(frames, ignore_index=True)
    print(f"\n  Total: {df_all['ticker'].nunique()} firms, {len(df_all)} firm-years\n")

    # 4. Save raw
    raw_path = DATA_DIR / "annual_financials.csv"
    df_all.to_csv(raw_path, index=False)
    print(f"  Raw data → {raw_path.name}\n")

    # 5. Clean transitions & duplicates
    print("Cleaning transitions & duplicates …")
    df_all = filter_transitions_and_duplicates(df_all)

    # 6. Impute
    print("Computing missing components …")
    df_all = compute_missing_components(df_all)
    print("Imputing balance-sheet gaps …")
    df_all = impute_balance_sheet(df_all, "year_end")

    # 7. Stock returns & volatility
    print("\nComputing stock returns & volatility (365-day window) …")
    df_all = compute_returns_and_volatility(
        df_all,
        date_col="year_end",
        return_col="return_next_year",
        vol_col="vol_next_year",
        window_days=ANNUAL_RETURN_WINDOW,
        min_trading_days=MIN_TRADING_DAYS_ANNUAL,
    )

    # 8. Financial ratios & winsorize
    print("\nDeriving financial ratios …")
    df_all = add_financial_ratios(df_all, "year_end")
    df_all = drop_earliest_year(df_all)

    # 9. Save
    print("\nSaving final panel …")
    save_panel(df_all, DATA_DIR / "annual_panel.csv", ANNUAL_COLS)
    print("\nDone.")


if __name__ == "__main__":
    main()
