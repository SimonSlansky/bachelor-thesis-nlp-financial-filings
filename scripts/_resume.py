"""Resume the annual panel pipeline from the saved raw CSV (steps 4b–9)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pandas as pd
from config import DATA_DIR, ANNUAL_RETURN_WINDOW, MIN_TRADING_DAYS_ANNUAL, MIN_FIRM_COVERAGE
from returns import compute_returns_and_volatility
from panel import (
    compute_missing_components, impute_balance_sheet, add_financial_ratios,
    filter_transitions_and_duplicates, drop_earliest_year,
    drop_low_return_coverage, cap_fiscal_year, winsorize_ratios,
    add_lagged_volatility, save_panel, ANNUAL_COLS,
    lock_firm_tags, save_tag_diagnostics,
)

df = pd.read_csv(DATA_DIR / "annual_financials.csv")
print(f"Loaded {df['ticker'].nunique()} firms, {len(df)} rows")

print("Step 4b: Locking per-firm XBRL tags …")
df = lock_firm_tags(df)
save_tag_diagnostics(df, DATA_DIR / "diagnostics" / "tag_provenance.csv")

print("\nStep 5: Cleaning …")
df = filter_transitions_and_duplicates(df)

print("Step 6: Imputing …")
df = compute_missing_components(df)
df = impute_balance_sheet(df, "year_end")

print("\nStep 7: Returns & volatility …")
df = compute_returns_and_volatility(
    df, date_col="year_end", return_col="return_next_year",
    vol_col="vol_next_year", window_days=ANNUAL_RETURN_WINDOW,
    min_trading_days=MIN_TRADING_DAYS_ANNUAL,
)

print("\nStep 8: Ratios …")
df = add_financial_ratios(df, "year_end")
df = add_lagged_volatility(df, "vol_next_year", "year_end")
df = drop_earliest_year(df)

print("\nStep 8b: Filtering panel …")
df = cap_fiscal_year(df, max_fy=2024)
df = drop_low_return_coverage(df, "vol_next_year", MIN_FIRM_COVERAGE)
df = winsorize_ratios(df)
print("\nStep 9: Saving …")
save_panel(df, DATA_DIR / "annual_panel.csv", ANNUAL_COLS)
print("Done.")
