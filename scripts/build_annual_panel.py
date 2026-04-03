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
    MIN_FIRM_COVERAGE,
)
from sec_edgar import fetch_universe, load_company_metadata, extract_annual_facts
from returns import compute_returns_and_volatility
from panel import (
    compute_missing_components, impute_balance_sheet, add_financial_ratios,
    filter_transitions_and_duplicates, drop_earliest_year,
    drop_low_return_coverage, cap_fiscal_year, winsorize_ratios,
    add_lagged_volatility, save_panel, ANNUAL_COLS,
    lock_firm_tags, save_tag_diagnostics,
)

# ── pipeline tracker ──────────────────────────────────────────────────────

_KEY_COLS = ["total_assets", "net_income", "total_liabilities",
             "operating_income", "operating_cash_flow",
             "return_next_year", "vol_next_year"]
_prev_snapshot: dict | None = None


def _snap(df: pd.DataFrame) -> dict:
    """Capture current state for comparison."""
    missing = {c: int(df[c].isna().sum()) for c in _KEY_COLS if c in df.columns}
    return {"rows": len(df), "firms": int(df["ticker"].nunique()), "missing": missing}


def _track(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Print a one-line stage summary and optional delta from previous stage."""
    global _prev_snapshot
    cur = _snap(df)
    header = f"  [{label}]  {cur['firms']} firms x {cur['rows']} rows"

    if _prev_snapshot is not None:
        dr = cur["rows"] - _prev_snapshot["rows"]
        df_firms = cur["firms"] - _prev_snapshot["firms"]
        parts = []
        if dr:
            parts.append(f"rows {dr:+d}")
        if df_firms:
            parts.append(f"firms {df_firms:+d}")
        # report changes in missing counts
        for col in _KEY_COLS:
            prev_m = _prev_snapshot["missing"].get(col)
            cur_m = cur["missing"].get(col)
            if prev_m is not None and cur_m is not None and cur_m != prev_m:
                delta = cur_m - prev_m
                parts.append(f"{col} NaN {delta:+d}")
            elif prev_m is None and cur_m is not None and cur_m > 0:
                parts.append(f"{col} NaN={cur_m}")
        if parts:
            header += "  (" + ", ".join(parts) + ")"

    print(header)
    _prev_snapshot = cur
    return df


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

    # Merge SIC and fiscal_year_end from metadata
    meta_df = pd.DataFrame(companies.values())[["ticker", "sic", "fiscal_year_end"]]
    df_all = df_all.merge(meta_df, on="ticker", how="left")

    # 4. Save raw
    raw_path = DATA_DIR / "annual_financials.csv"
    df_all.to_csv(raw_path, index=False)
    print(f"  Raw data -> {raw_path.name}\n")

    # ── Pipeline tracking starts here ──────────────────────────────────
    print("=" * 66)
    print("  PIPELINE STAGE TRACKER")
    print("=" * 66)
    df_all = _track(df_all, "raw extraction")

    # 4b. Per-firm tag locking (ensures within-firm XBRL tag consistency)
    print("\nLocking per-firm XBRL tags …")
    df_all = lock_firm_tags(df_all)
    save_tag_diagnostics(df_all, DATA_DIR / "diagnostics" / "tag_provenance.csv")
    df_all = _track(df_all, "tag locking")

    # 5. Clean transitions & duplicates
    print("\nCleaning transitions & duplicates …")
    df_all = filter_transitions_and_duplicates(df_all)
    df_all = _track(df_all, "transitions & dedup")

    # 6. Impute
    print("\nImputing missing data …")
    df_all = compute_missing_components(df_all)
    df_all = impute_balance_sheet(df_all, "year_end")
    df_all = _track(df_all, "imputation")

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
    df_all = _track(df_all, "returns & volatility")

    # 8. Financial ratios (raw, unwinsorized)
    print("\nDeriving financial ratios …")
    df_all = add_financial_ratios(df_all, "year_end")
    df_all = add_lagged_volatility(df_all, "vol_next_year", "year_end")
    df_all = _track(df_all, "ratios & lagged vol")

    df_all = drop_earliest_year(df_all)
    df_all = _track(df_all, "drop earliest year")

    # 8b. Cap fiscal year, then drop firms with insufficient return coverage
    print("\nFiltering panel …")
    df_all = cap_fiscal_year(df_all, max_fy=2024)
    df_all = _track(df_all, "cap FY <= 2024")

    df_all = drop_low_return_coverage(df_all, "vol_next_year", MIN_FIRM_COVERAGE)
    df_all = _track(df_all, "coverage filter")

    # 8c. Winsorize on the final sample
    df_all = winsorize_ratios(df_all)
    df_all = _track(df_all, "winsorize")

    print("=" * 66)

    # 9. Save
    print("\nSaving final panel …")
    save_panel(df_all, DATA_DIR / "annual_panel.csv", ANNUAL_COLS)
    print("\nDone.")


if __name__ == "__main__":
    main()
