#!/usr/bin/env python3
"""
Defense & Energy Companies - Quarterly Financial Analysis (Q1-Q3)
Analyzing quarterly (10-Q) financial data for defense contractors and energy companies with key financial metrics.

Note: This file is exported from the notebook. It expects the same data files and environment.
"""

import requests
import time
import re
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import yfinance as yf

# =============================================================================
# CONFIGURATION
# =============================================================================
HEADERS = {"User-Agent": "Simon Slansky simon.slansky@outlook.com"}
REQUEST_SLEEP = 0.15

# Data extraction
NUM_QUARTERS_TO_EXTRACT = 45  # or your desired number

# Filing validation
MAX_FILING_LAG_DAYS = 180

MIN_VALID_QUARTERS = 4

# Target companies (Defense & Energy)
TARGET_TICKERS = [
    # Tier 1: Prime defense contractors (čistá expozice)
    "LMT",   # Lockheed Martin
    "RTX",   # RTX (Raytheon – missiles, defense systems)
    "NOC",   # Northrop Grumman
    "GD",    # General Dynamics
    "LHX",   # L3Harris Technologies

    # Tier 2: Defense services & intelligence
    "BAH",   # Booz Allen Hamilton
    "LDOS",  # Leidos
    "CACI",  # CACI International
    "PSN",   # Parsons
    "HII",   # Huntington Ingalls (naval defense)

    # Tier 3: Aerospace / military platforms (high exposure)
    "TDG",   # TransDigm (aerospace & defense components)

    # Tier 4: Emerging / specialized defense tech
    "KTOS",  # Kratos Defense (unmanned aerial systems, drones)
    "AVAV",  # AeroVironment (tactical drones, loitering munitions)
    "MRCY",  # Mercury Systems (electronic warfare, secure computing)
    "TXT",   # Textron (military helicopters, armored vehicles)
    "CW"     # Curtiss-Wright (defense electronics, propulsion systems)
]


# Paths
BASE_DIR = Path.cwd().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

print(f"✓ Configuration loaded")
print(f"  Target companies: {len(TARGET_TICKERS)}")
print(f"  Data directory: {DATA_DIR}")

# =============================================================================
# STEP 2: LOAD SEC TICKERS
# =============================================================================
print("\n" + "="*80)
print("STEP 2: Loading SEC Tickers")
print("="*80)

# Load SEC company tickers
sec_json = requests.get(
    "https://www.sec.gov/files/company_tickers.json",
    headers=HEADERS
).json()

# Convert to DataFrame
sec_df = pd.DataFrame.from_dict(sec_json, orient="index")
sec_df["ticker"] = sec_df["ticker"].str.upper()
sec_df["cik"] = sec_df["cik_str"].astype(str).str.zfill(10)

print(f"✓ SEC tickers loaded: {len(sec_df)} companies")

# Filter to target tickers
sec_df = sec_df[sec_df["ticker"].isin(TARGET_TICKERS)].copy()

print(f"✓ Filtered to target companies: {len(sec_df)} tickers")
print(f"\nFound tickers:")
for ticker in sorted(sec_df["ticker"].tolist()):
    print(f"  {ticker}")

# =============================================================================
# STEP 3: LOAD SEC SUBMISSIONS METADATA
# =============================================================================
print("\n" + "="*80)
print("STEP 3: Loading SEC Submissions Metadata")
print("="*80)

def load_submissions(cik: str) -> dict | None:
    """Load SEC submissions metadata for a company."""
    try:
        r = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=HEADERS,
            timeout=10
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"  Error loading CIK {cik}: {e}")
    return None

# Load metadata for all target companies
companies = []
failed_ciks = []

print("Loading SEC submissions metadata...")
for _, row in sec_df.iterrows():
    ticker = row["ticker"]
    cik = row["cik"]

    sub = load_submissions(cik)
    if sub is None:
        failed_ciks.append((ticker, cik))
        print(f"  ⚠️ Failed to load: {ticker} (CIK {cik})")
        time.sleep(REQUEST_SLEEP)
        continue

    # Extract fiscal year-end (required)
    fye_raw = sub.get("fiscalYearEnd")
    if not isinstance(fye_raw, str) or len(fye_raw) < 2:
        print(f"  ⚠️ Missing fiscal year-end: {ticker}")
        failed_ciks.append((ticker, cik))
        time.sleep(REQUEST_SLEEP)
        continue

    try:
        fiscal_year_end = int(fye_raw[:2])  # Extract month from MMDD format (first 2 digits)
    except:
        print(f"  ⚠️ Invalid fiscal year-end format: {ticker} ({fye_raw})")
        failed_ciks.append((ticker, cik))
        time.sleep(REQUEST_SLEEP)
        continue

    # Get exchange info (optional - for informational purposes)
    exchanges = sub.get("exchanges", [])
    entity_type = sub.get("entityType", "unknown")

    companies.append({
        "ticker": ticker,
        "cik": cik,
        "company_name": sub.get("name", ticker),
        "exchanges": exchanges,
        "entity_type": entity_type,
        "fiscal_year_end": fiscal_year_end
    })

    print(f"  ✓ {ticker}: {sub.get('name', 'N/A')} | FYE: {fye_raw} | Exchange: {', '.join(exchanges) if exchanges else 'N/A'}")
    time.sleep(REQUEST_SLEEP)

target_companies = {c["ticker"]: c for c in companies}

print(f"\n{'='*80}")
print(f"✓ Successfully loaded: {len(target_companies)}/{len(TARGET_TICKERS)} companies")
if failed_ciks:
    print(f"⚠️ Failed to load {len(failed_ciks)} companies:")
    for ticker, cik in failed_ciks:
        print(f"    {ticker} (CIK {cik})")
print(f"{'='*80}")

# =============================================================================
# STEP 4: DEFINE FINANCIAL METRICS
# =============================================================================
print("\n" + "="*80)
print("STEP 4: Defining Financial Metrics")
print("="*80)

# Financial metrics with priority (lower number = higher priority)
METRICS_WITH_PRIORITY = [
    # Total Assets
    ('Assets', 'total_assets', 1),
    ('AssetsFairValueDisclosure', 'total_assets', 2),

    # Asset Components
    ('AssetsCurrent', 'assets_current', 1),
    ('AssetsNoncurrent', 'assets_noncurrent', 1),
    ('CashAndCashEquivalentsAtCarryingValue', 'cash', 1),
    ('Cash', 'cash', 2),
    ('AccountsReceivableNetCurrent', 'accounts_receivable_net', 1),
    ('AccountsAndNotesReceivableNet', 'accounts_receivable_net', 2),
    ('InventoryNet', 'inventory_net', 1),
    ('MarketableSecurities', 'marketable_securities', 1),
    ('PrepaidExpenseAndOtherAssetsCurrent', 'prepaid_expense_current', 1),
    ('PropertyPlantAndEquipmentNet', 'ppe_net', 1),
    ('IntangibleAssetsNetExcludingGoodwill', 'intangibles_net', 1),
    ('Goodwill', 'goodwill', 1),
    ('OtherAssetsNoncurrent', 'other_assets_noncurrent', 1),

    # Total Liabilities
    ('Liabilities', 'total_liabilities', 1),
    ('LiabilitiesCurrent', 'liabilities_current', 1),
    ('LiabilitiesNoncurrent', 'liabilities_noncurrent', 1),

    # Stockholders' Equity
    ('StockholdersEquity', 'stockholders_equity', 1),
    ('StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest', 'stockholders_equity', 2),

    # Net Income (Flow metric)
    ('NetIncomeLoss', 'net_income', 1),
    ('ProfitLoss', 'net_income', 2),
    ('NetIncomeLossAvailableToCommonStockholdersBasic', 'net_income', 3),
    ('IncomeLossFromContinuingOperations', 'net_income', 4),
]

# Classify metrics as flow (need start/end) or instant (point-in-time)
FLOW_METRICS = {"net_income"}
INSTANT_METRICS = {m for _, m, _ in METRICS_WITH_PRIORITY if m not in FLOW_METRICS}

print(f"✓ Defined {len(METRICS_WITH_PRIORITY)} XBRL tag mappings")
print(f"  Flow metrics (need date range): {FLOW_METRICS}")
print(f"  Instant metrics (point-in-time): {len(INSTANT_METRICS)} metrics")

# =============================================================================
# STEP 5: FISCAL CALENDAR HELPER FUNCTIONS
# =============================================================================
print("\n" + "="*80)
print("STEP 5: Fiscal Calendar Functions")
print("="*80)

def fiscal_quarter_and_year(end_date, fye_month):
    """Correctly determine fiscal quarter and year based on month distance from FYE."""
    end_month = end_date.month

    # Calculate month distance from fiscal year end
    diff = (end_month - fye_month) % 12

    # Map distance to fiscal quarter
    if diff == 3:
        fiscal_quarter = "Q1"
    elif diff == 6:
        fiscal_quarter = "Q2"
    elif diff == 9:
        fiscal_quarter = "Q3"
    elif diff == 0:
        fiscal_quarter = "Q4"
    else:
        # Not a fiscal quarter end
        fiscal_quarter = None

    # Determine fiscal year
    # If we're in months after FYE (diff > 0), we're in the new fiscal year
    if end_month > fye_month:
        fiscal_year = end_date.year + 1
    else:
        fiscal_year = end_date.year

    return fiscal_year, fiscal_quarter

print("✓ Fiscal calendar functions defined")

# =============================================================================
# STEP 6: XBRL FACT VALIDATION FUNCTIONS
# =============================================================================
print("\n" + "="*80)
print("STEP 6: XBRL Validation Functions")
print("="*80)

def is_valid_flow(row, fye_month):
    """Validate flow metrics with corrected duration window and Q4 exclusion."""
    # Must be from 10-Q filing
    if row.get("form") not in {"10-Q", "10-Q/A"}:
        return False

    # Must have valid end date
    end = pd.to_datetime(row.get("end"), errors="coerce")
    if pd.isna(end):
        return False

    # Must have start date
    start = row.get("start")
    if not start:
        return False

    # Duration should be ~90 days (quarterly) - FIXED: widened window from 70-110 to 65-120
    start = pd.to_datetime(start, errors="coerce")
    if pd.isna(start):
        return False

    days = (end - start).days
    if not (65 <= days <= 120):
        return False

    # FIXED: Exclude Q4 using proper fiscal quarter logic
    _, fiscal_quarter = fiscal_quarter_and_year(end, fye_month)
    if fiscal_quarter == "Q4":
        return False

    return True

def is_valid_instant(row, fye_month):
    """
    FIXED: Validate instant metrics with corrected Q4 exclusion.

    Instant metrics (like total assets) are point-in-time, no start date.
    """
    # Must be from 10-Q filing
    if row.get("form") != "10-Q":
        return False

    # Must have valid end date
    end = pd.to_datetime(row.get("end"), errors="coerce")
    if pd.isna(end):
        return False

    # Must NOT have start date (instant value)
    if row.get("start") is not None:
        return False

    # FIXED: Exclude Q4 using proper fiscal quarter logic
    _, fiscal_quarter = fiscal_quarter_and_year(end, fye_month)
    if fiscal_quarter == "Q4":
        return False

    return True

print("✓ XBRL validation functions defined")

# =============================================================================
# STEP 7: EXTRACT QUARTERLY FINANCIAL DATA
# =============================================================================
print("\n" + "="*80)
print("STEP 7: Data Extraction Function")
print("="*80)

def extract_quarters(meta: dict) -> pd.DataFrame:
    """Extract quarterly financial data with accession_number tracking."""
    try:
        r = requests.get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{meta['cik']}.json",
            headers=HEADERS,
            timeout=15
        )
        if r.status_code != 200:
            return pd.DataFrame()

        data = r.json()
        us_gaap = data.get("facts", {}).get("us-gaap", {})
        if not us_gaap:
            return pd.DataFrame()

    except Exception as e:
        print(f"    Error fetching XBRL for {meta['ticker']}: {e}")
        return pd.DataFrame()

    # Tracking counters
    stats = {
        'total_facts': 0,
        'after_metric_type_validation': 0,
        'after_fiscal_quarter_filter': 0,
        'after_filing_date_check': 0,
        'after_filing_lag_validation': 0,
        'unique_quarters': 0,
        'after_quarter_limit': 0,
        'accession_conflicts': 0,
    }

    # Dictionary to store records by (fiscal_year, fiscal_quarter) to handle date variations
    records = {}

    # Process each metric with priority handling
    for tag, metric, priority in METRICS_WITH_PRIORITY:
        units = us_gaap.get(tag, {}).get("units", {}).get("USD", [])

        for fact in units:
            stats['total_facts'] += 1

            # Validate based on metric type
            if metric in FLOW_METRICS:
                if not is_valid_flow(fact, meta["fiscal_year_end"]):
                    continue
            elif metric in INSTANT_METRICS:
                if not is_valid_instant(fact, meta["fiscal_year_end"]):
                    continue
            else:
                continue

            stats['after_metric_type_validation'] += 1

            # Get quarter end date and fiscal quarter
            end = pd.to_datetime(fact["end"])
            fy, fq = fiscal_quarter_and_year(end, meta["fiscal_year_end"])

            # Skip if not a valid fiscal quarter (Q1-Q3 only)
            if fq is None or fq == "Q4":
                continue

            stats['after_fiscal_quarter_filter'] += 1

            # Check filing date
            filing_date = pd.to_datetime(fact.get("filed"), errors="coerce")
            if pd.isna(filing_date):
                continue

            stats['after_filing_date_check'] += 1

            # Validate filing lag
            if (filing_date - end).days > MAX_FILING_LAG_DAYS:
                continue

            stats['after_filing_lag_validation'] += 1

            # Extract accession_number for tracking
            accession = fact.get("accn")
            if not accession:
                continue

            # Use (fiscal_year, fiscal_quarter) as key to handle date variations
            quarter_key = (fy, fq)

            # Create or update record for this quarter
            if quarter_key not in records:
                records[quarter_key] = {
                    "ticker": meta["ticker"],
                    "company_name": meta["company_name"],
                    "quarter_end": end,
                    "filing_date": filing_date,
                    "fiscal_year": fy,
                    "fiscal_quarter": fq,
                    "accession_number": accession,
                }
            else:
                existing = records[quarter_key]

                # CRITICAL FIX: DO NOT allow later filings to overwrite earlier ones
                # This prevents mixing metrics from different reports for the same quarter
                if filing_date > existing['filing_date']:
                    continue

                # If this is an earlier filing, replace everything
                if filing_date < existing['filing_date']:
                    # Track conflicts for reporting
                    if existing["accession_number"] != accession:
                        stats['accession_conflicts'] += 1

                    # Reset to use earlier filing
                    existing["quarter_end"] = end  # Update to earlier filing's end date
                    existing["filing_date"] = filing_date
                    existing["accession_number"] = accession
                    # Clear all metrics to repopulate from earlier filing
                    metrics_to_clear = [m for _, m, _ in METRICS_WITH_PRIORITY]
                    for m in metrics_to_clear:
                        if m in existing:
                            del existing[m]
                        priority_key = f"{m}__priority"
                        if priority_key in existing:
                            del existing[priority_key]
                else:
                    # Same filing date - update quarter_end to latest date if needed
                    if end > existing["quarter_end"]:
                        existing["quarter_end"] = end

            # Only set metric if not already set OR if this has higher priority
            rec = records[quarter_key]
            if metric not in rec or priority < rec.get(f"{metric}__priority", 999):
                rec[metric] = fact["val"]
                rec[f"{metric}__priority"] = priority

    stats['unique_quarters'] = len(records)

    df = pd.DataFrame(list(records.values()))

    # Remove priority tracking columns
    priority_cols = [col for col in df.columns if col.endswith("__priority")]
    if priority_cols:
        df = df.drop(columns=priority_cols)

    # Limit to most recent N quarters
    if not df.empty and len(df) > NUM_QUARTERS_TO_EXTRACT:
        df = df.sort_values('quarter_end', ascending=False).head(NUM_QUARTERS_TO_EXTRACT)
        df = df.sort_values('quarter_end')

    stats['after_quarter_limit'] = len(df)

    # Store stats in DataFrame attributes for aggregated reporting
    df.attrs['stats'] = stats
    return df

print("✓ Data extraction function defined")

# =============================================================================
# STEP 8: PROCESS ALL COMPANIES
# =============================================================================
print("\n" + "="*80)
print("STEP 8: Processing All Companies")
print("="*80)

all_quarters = []
extraction_stats = []

print("Extracting quarterly financial data...")
print("="*80)

for ticker, meta in tqdm(sorted(target_companies.items()), desc="Extracting companies", unit="company"):
    df = extract_quarters(meta)

    if df.empty:
        tqdm.write(f"{ticker}: ❌ No data")
        extraction_stats.append({
            'ticker': ticker,
            'quarters_found': 0,
            'status': 'NO_DATA'
        })
    elif len(df) < MIN_VALID_QUARTERS:
        tqdm.write(f"{ticker}: ⚠️  {len(df)} quarters (insufficient)")
        extraction_stats.append({
            'ticker': ticker,
            'quarters_found': len(df),
            'status': 'INSUFFICIENT'
        })
    else:
        # Extract stats from df.attrs if available
        stats = df.attrs.get('stats', {})
        conflicts = stats.get('accession_conflicts', 0)

        all_quarters.append(df)
        conflict_marker = f" [{conflicts} conflicts]" if conflicts > 0 else ""
        tqdm.write(f"{ticker}: ✓ {len(df)} quarters{conflict_marker}")

        extraction_stats.append({
            'ticker': ticker,
            'quarters_found': len(df),
            'status': 'SUCCESS',
            'accession_conflicts': conflicts,
            'total_facts': stats.get('total_facts', 0),
            'after_metric_validation': stats.get('after_metric_type_validation', 0),
            'after_fiscal_filter': stats.get('after_fiscal_quarter_filter', 0),
            'after_filing_lag': stats.get('after_filing_lag_validation', 0),
        })

    time.sleep(REQUEST_SLEEP)

print("\n" + "="*80)

# Combine all data
if len(all_quarters) == 0:
    df_all = pd.DataFrame()
    print("⚠️ WARNING: No companies passed extraction!")
else:
    df_all = pd.concat(all_quarters, ignore_index=True)
    print(f"✓ EXTRACTION COMPLETE")
    print(f"  Total firm-quarters: {len(df_all):,}")
    print(f"  Companies with data: {df_all['ticker'].nunique()}/{len(target_companies)}")
    print(f"  Date range: {df_all['quarter_end'].min().date()} to {df_all['quarter_end'].max().date()}")

# Show extraction summary
df_stats = pd.DataFrame(extraction_stats)
print(f"\nExtraction summary:")
print(df_stats['status'].value_counts().to_string())

if len(df_stats[df_stats['status'] != 'SUCCESS']) > 0:
    print(f"\nCompanies with issues:")
    for _, row in df_stats[df_stats['status'] != 'SUCCESS'].iterrows():
        print(f"  {row['ticker']:6s}: {row['status']:15s} ({row['quarters_found']} quarters)")

# Aggregated pipeline statistics
successful = df_stats[df_stats['status'] == 'SUCCESS']
if len(successful) > 0:
    print(f"\nAggregated Data Pipeline Statistics (across {len(successful)} successful companies):")
    print(f"  Total XBRL facts processed: {successful['total_facts'].sum():,}")
    print(f"  After metric validation: {successful['after_metric_validation'].sum():,} ({successful['after_metric_validation'].sum()/successful['total_facts'].sum()*100:.1f}% retained)")
    print(f"  After fiscal quarter filter (Q1-Q3): {successful['after_fiscal_filter'].sum():,} ({successful['after_fiscal_filter'].sum()/successful['after_metric_validation'].sum()*100:.1f}% retained)")
    print(f"  After filing lag validation: {successful['after_filing_lag'].sum():,} ({successful['after_filing_lag'].sum()/successful['after_fiscal_filter'].sum()*100:.1f}% retained)")
    print(f"  Final firm-quarters: {len(df_all):,}")
    total_conflicts = successful['accession_conflicts'].sum()
    if total_conflicts > 0:
        print(f"  ⚠️  Total accession conflicts resolved: {total_conflicts}")

print("="*80)

# =============================================================================
# STEP 9: DATA QUALITY CHECK
# =============================================================================
print("\n" + "="*80)
print("STEP 9: Data Quality Check")
print("="*80)

if not df_all.empty:
    print("DATA QUALITY CHECK")
    print("="*80)

    # Core metrics completeness
    core_metrics = ['total_assets', 'total_liabilities', 'net_income']

    print("\nMissing values by metric:")
    for metric in core_metrics:
        if metric in df_all.columns:
            missing = df_all[metric].isna().sum()
            pct = missing / len(df_all) * 100
            print(f"  {metric}: {missing}/{len(df_all)} ({pct:.1f}% missing)")
        else:
            print(f"  {metric}: Column not found!")

    # Missing by company
    print("\nMissing values by company (core metrics):")
    for ticker in sorted(df_all['ticker'].unique()):
        ticker_data = df_all[df_all['ticker'] == ticker]
        missing_counts = []
        for metric in core_metrics:
            if metric in df_all.columns:
                missing_counts.append(ticker_data[metric].isna().sum())
        print(f"  {ticker}: {sum(missing_counts)} total missing values across {len(ticker_data)} quarters")

    # Show sample of data
    print("\nSample data (first 5 rows):")
    display_cols = ['ticker', 'quarter_end', 'fiscal_quarter', 'filing_date',
                    'total_assets', 'total_liabilities', 'net_income']
    display_cols = [c for c in display_cols if c in df_all.columns]
    print(df_all[display_cols].head())

    print("="*80)
else:
    print("⚠️ No data to analyze - extraction failed for all companies")

# =============================================================================
# STEP 10: SAVE RAW EXTRACTION
# =============================================================================
print("\n" + "="*80)
print("STEP 10: Saving Raw Extraction")
print("="*80)

if not df_all.empty:
    output_path = DATA_DIR / 'quarterly_financials.csv'
    df_all.to_csv(output_path, index=False)
    print(f"✓ Raw data saved to: {output_path}")
else:
    print("⚠️ No data to save")

# =============================================================================
# STEP 11: COMPUTE MISSING METRICS FROM COMPONENTS
# =============================================================================
print("\n" + "="*80)
print("STEP 11: Computing Missing Metrics")
print("="*80)

if not df_all.empty:
    # Track missing values before computation
    missing_before_assets = df_all['total_assets'].isna().sum()
    missing_before_liab = df_all['total_liabilities'].isna().sum()

    # Initialize tag columns
    if 'total_assets__tag' not in df_all.columns:
        df_all['total_assets__tag'] = None
    if 'total_liabilities__tag' not in df_all.columns:
        df_all['total_liabilities__tag'] = None

    # =================================================================
    # COMPUTE MISSING TOTAL ASSETS
    # =================================================================

    # Strategy 1: AssetsCurrent + AssetsNoncurrent
    if all(col in df_all.columns for col in ['assets_current', 'assets_noncurrent']):
        mask = (
            df_all['total_assets'].isna() &
            df_all['assets_current'].notna() &
            df_all['assets_noncurrent'].notna()
        )
        df_all.loc[mask, 'total_assets'] = (
            df_all.loc[mask, 'assets_current'] +
            df_all.loc[mask, 'assets_noncurrent']
        )
        df_all.loc[mask, 'total_assets__tag'] = 'COMPUTED: AssetsCurrent + AssetsNoncurrent'

    # =================================================================
    # COMPUTE MISSING TOTAL LIABILITIES
    # =================================================================

    # Strategy 1: LiabilitiesCurrent + LiabilitiesNoncurrent
    if all(col in df_all.columns for col in ['liabilities_current', 'liabilities_noncurrent']):
        mask = (
            df_all['total_liabilities'].isna() &
            df_all['liabilities_current'].notna() &
            df_all['liabilities_noncurrent'].notna()
        )
        df_all.loc[mask, 'total_liabilities'] = (
            df_all.loc[mask, 'liabilities_current'] +
            df_all.loc[mask, 'liabilities_noncurrent']
        )
        df_all.loc[mask, 'total_liabilities__tag'] = 'COMPUTED: LiabilitiesCurrent + LiabilitiesNoncurrent'

    # Strategy 2: Assets - Equity
    if all(col in df_all.columns for col in ['total_assets', 'stockholders_equity']):
        mask = (
            df_all['total_liabilities'].isna() &
            df_all['total_assets'].notna() &
            df_all['stockholders_equity'].notna()
        )
        df_all.loc[mask, 'total_liabilities'] = (
            df_all.loc[mask, 'total_assets'] -
            df_all.loc[mask, 'stockholders_equity']
        )
        df_all.loc[mask, 'total_liabilities__tag'] = 'COMPUTED: Assets - Equity'

    # =================================================================
    # SUMMARY
    # =================================================================

    missing_after_assets = df_all['total_assets'].isna().sum()
    missing_after_liab = df_all['total_liabilities'].isna().sum()

    computed_assets = missing_before_assets - missing_after_assets
    computed_liab = missing_before_liab - missing_after_liab

    print("METRIC COMPUTATION SUMMARY")
    print("="*80)
    print(f"total_assets:")
    print(f"  Before: {missing_before_assets} missing")
    print(f"  Computed: {computed_assets}")
    print(f"  After: {missing_after_assets} missing")
    print(f"\ntotal_liabilities:")
    print(f"  Before: {missing_before_liab} missing")
    print(f"  Computed: {computed_liab}")
    print(f"  After: {missing_after_liab} missing")
    print(f"\nnet_income:")
    print(f"  No computation (keep as-is to avoid measurement error)")
    print(f"  Missing: {df_all['net_income'].isna().sum()}/{len(df_all)}")
    print("="*80)
else:
    print("⚠️ No data to process")

# =============================================================================
# STEP 12: TIME-SERIES IMPUTATION
# =============================================================================
print("\n" + "="*80)
print("STEP 12: Time-Series Imputation")
print("="*80)

# Sort by company and date for proper time-series operations
df_all = df_all.sort_values(['ticker', 'quarter_end']).reset_index(drop=True)

# Balance sheet metrics to impute
balance_sheet_metrics = ['total_assets', 'total_liabilities']

# Apply forward-fill then backward-fill (limit=1 means at most 1 quarter)
for metric in balance_sheet_metrics:
    if metric in df_all.columns:
        before = df_all[metric].isna().sum()
        df_all[metric] = df_all.groupby('ticker')[metric].ffill(limit=1)
        df_all[metric] = df_all.groupby('ticker')[metric].bfill(limit=1)
        after = df_all[metric].isna().sum()
        filled = before - after
        print(f"✓ {metric}: filled {filled} values (±1 quarter)")

# =============================================================================
# STEP 13: CALCULATE STOCK RETURNS
# =============================================================================
print("\n" + "="*80)
print("STEP 13: Calculating Stock Returns")
print("="*80)

if not df_all.empty:
    # Configuration - FIXED: Corrected comment for RETURN_WINDOW_DAYS
    RETURN_WINDOW_DAYS = 63  # ~63 trading days ≈ 1 calendar quarter (3 months)
    POST_FILING_LAG = 2
    MIN_TRADING_DAYS = 40
    MIN_FIRM_COVERAGE = 0.6

    results = []
    firm_stats = []

    min_date = pd.to_datetime(df_all['quarter_end'].min())
    max_date = pd.to_datetime(df_all['quarter_end'].max())

    print("Calculating stock returns...")
    print("="*80)

    for ticker in tqdm(sorted(df_all['ticker'].unique()), desc="Calculating returns", unit="company"):
        # Download stock data
        stock = yf.download(
            ticker,
            start=min_date,
            end=max_date + pd.Timedelta(days=180),
            progress=False,
            auto_adjust=True
        )

        if stock.empty or 'Close' not in stock.columns:
            tqdm.write(f"{ticker}: No price data")
            continue

        stock = stock[['Close']].dropna()
        firm_quarters = df_all[df_all['ticker'] == ticker]
        usable = 0

        for _, row in firm_quarters.iterrows():
            filing_date = pd.to_datetime(row['filing_date'], errors='coerce')
            if pd.isna(filing_date):
                continue

            start_date = filing_date + pd.Timedelta(days=POST_FILING_LAG)
            end_date = start_date + pd.Timedelta(days=RETURN_WINDOW_DAYS)

            window = stock.loc[start_date:end_date]
            if len(window) < MIN_TRADING_DAYS:
                continue

            log_return = float(np.log(window.iloc[-1]['Close'] / window.iloc[0]['Close']))
            usable += 1

            results.append({
                'ticker': ticker,
                'quarter_end': row['quarter_end'],
                'return_next_q': log_return
            })

        firm_stats.append({
            'ticker': ticker,
            'quarters_total': len(firm_quarters),
            'quarters_with_returns': usable,
            'coverage_ratio': usable / len(firm_quarters) if len(firm_quarters) > 0 else 0
        })

        tqdm.write(f"{ticker}: {usable}/{len(firm_quarters)} quarters")

    df_stock = pd.DataFrame(results)
    df_coverage = pd.DataFrame(firm_stats)

    if df_stock.empty:
        print("\n⚠️ No stock returns computed")
        df_stock = pd.DataFrame(columns=["ticker", "quarter_end", "return_next_q"])
    else:
        # Keep only firms with sufficient coverage
        usable_firms = df_coverage[df_coverage['coverage_ratio'] >= MIN_FIRM_COVERAGE]['ticker']
        df_stock = df_stock[df_stock['ticker'].isin(usable_firms)]

        print("\n" + "="*80)
        print(f"✓ Returns calculated")
        print(f"  Firms with returns: {df_stock['ticker'].nunique()}")
        print(f"  Total observations: {len(df_stock)}")
        print(f"  Coverage threshold: {MIN_FIRM_COVERAGE*100:.0f}%")
        print("="*80)
else:
    print("⚠️ No financial data - skipping returns")
    df_stock = pd.DataFrame(columns=["ticker", "quarter_end", "return_next_q"])

# =============================================================================
# STEP 14: MERGE STOCK RETURNS WITH FINANCIAL DATA
# =============================================================================
print("\n" + "="*80)
print("STEP 14: Merging Stock Returns")
print("="*80)

if not df_all.empty and not df_stock.empty:
    df_panel = df_all.merge(
        df_stock[['ticker', 'quarter_end', 'return_next_q']],
        on=['ticker', 'quarter_end'],
        how='left'
    )
    print(f"✓ Merged stock returns")
    print(f"  Panel: {len(df_panel)} rows")
    print(f"  Rows with returns: {df_panel['return_next_q'].notna().sum()}")
elif not df_all.empty:
    df_panel = df_all.copy()
    df_panel['return_next_q'] = np.nan
    print("⚠️ No stock returns to merge")
else:
    df_panel = pd.DataFrame()
    print("⚠️ No data to merge")

# =============================================================================
# STEP 15: CALCULATE FINANCIAL METRICS
# =============================================================================
print("\n" + "="*80)
print("STEP 15: Calculating Financial Metrics")
print("="*80)

if not df_panel.empty:
    df_panel = df_panel.sort_values(['ticker', 'quarter_end']).reset_index(drop=True)

    # =================================================================
    # 1) FIRM SIZE: Log of Total Assets
    # =================================================================
    df_panel['log_total_assets'] = np.where(
    df_panel['total_assets'] > 0,
    np.log(df_panel['total_assets']),
    np.nan
    )

    # =================================================================
    # 2) CAPITAL STRUCTURE / RISK: Leverage Ratio
    # =================================================================
    # Avoid division by zero
    df_panel['leverage'] = np.where(
        df_panel['total_assets'] > 0,
        df_panel['total_liabilities'] / df_panel['total_assets'],
        np.nan
    )

    # =================================================================
    # 3) PROFITABILITY: Return on Assets (ROA)
    # =================================================================
    # Avoid division by zero
    df_panel['roa'] = np.where(
        df_panel['total_assets'] > 0,
        df_panel['net_income'] / df_panel['total_assets'],
        np.nan
    )

    # =================================================================
    # 4) INVESTMENT DYNAMICS: Asset Growth Rate (lagged)
    # =================================================================
    # Calculate lagged total assets (within each firm)
    df_panel['total_assets_lag'] = df_panel.groupby('ticker')['total_assets'].shift(1)

    # Calculate asset growth: (Assets_t - Assets_t-1) / Assets_t-1
    df_panel['asset_growth'] = np.where(
        df_panel['total_assets_lag'] > 0,
        (df_panel['total_assets'] - df_panel['total_assets_lag']) / df_panel['total_assets_lag'],
        np.nan
    )

    # Drop temporary lagged column (not needed in final dataset)
    df_panel = df_panel.drop(columns=['total_assets_lag'])

    # =================================================================
    # SUMMARY OF NEW METRICS
    # =================================================================
    print("="*80)
    print("CALCULATED FINANCIAL METRICS")
    print("="*80)

    metrics_summary = {
        'log_total_assets': (
            f"{df_panel['log_total_assets'].notna().sum()}/{len(df_panel)} "
            f"({df_panel['log_total_assets'].notna().mean()*100:.1f}%)"
        ),
        'leverage': (
            f"{df_panel['leverage'].notna().sum()}/{len(df_panel)} "
            f"({df_panel['leverage'].notna().mean()*100:.1f}%)"
        ),
        'roa': (
            f"{df_panel['roa'].notna().sum()}/{len(df_panel)} "
            f"({df_panel['roa'].notna().mean()*100:.1f}%)"
        ),
        'asset_growth': (
            f"{df_panel['asset_growth'].notna().sum()}/{len(df_panel)} "
            f"({df_panel['asset_growth'].notna().mean()*100:.1f}%)"
        )
    }

    for metric, coverage in metrics_summary.items():
        print(f"  {metric}: {coverage}")

    print("\nDescriptive Statistics:")
    print(df_panel[['log_total_assets', 'leverage', 'roa', 'asset_growth']].describe())
    print("="*80)
else:
    print("⚠️ No panel data to process metrics")

# =============================================================================
# STEP 16: SAVE FINAL PANEL DATASET
# =============================================================================
print("\n" + "="*80)
print("STEP 16: Saving Final Dataset")
print("="*80)

if not df_panel.empty:
    # Select essential columns for final dataset
    essential_cols = [
        'ticker',
        'company_name',
        'quarter_end',
        'filing_date',
        'fiscal_year',
        'fiscal_quarter',
        'accession_number',
        # Primary financial data (as-reported or computed)
        'net_income',
        'total_assets',
        'total_liabilities',
        # Derived financial metrics
        'log_total_assets',
        'leverage',
        'roa',
        'asset_growth',
        # Market data
        'return_next_q',
    ]

    # Keep only columns that exist
    final_cols = [col for col in essential_cols if col in df_panel.columns]
    df_final = df_panel[final_cols].copy()

    # Save to CSV
    output_path = DATA_DIR / 'panel_data_with_returns.csv'
    df_final.to_csv(output_path, index=False)

    print("="*80)
    print("FINAL PANEL DATASET")
    print("="*80)
    print(f"✓ Saved to: {output_path}")
    print(f"\nDimensions:")
    print(f"  Companies: {df_final['ticker'].nunique()}")
    print(f"  Quarters: {len(df_final)}")
    print(f"  Date range: {df_final['quarter_end'].min()} to {df_final['quarter_end'].max()}")

    print(f"\nData completeness:")
    print(f"  filing_date: {df_final['filing_date'].notna().sum()}/{len(df_final)} ({df_final['filing_date'].notna().mean()*100:.1f}%)")
    print(f"  accession_number: {df_final['accession_number'].notna().sum()}/{len(df_final)} ({df_final['accession_number'].notna().mean()*100:.1f}%)")
    print(f"  net_income: {df_final['net_income'].notna().sum()}/{len(df_final)} ({df_final['net_income'].notna().mean()*100:.1f}%)")
    print(f"  total_assets: {df_final['total_assets'].notna().sum()}/{len(df_final)} ({df_final['total_assets'].notna().mean()*100:.1f}%)")
    print(f"  total_liabilities: {df_final['total_liabilities'].notna().sum()}/{len(df_final)} ({df_final['total_liabilities'].notna().mean()*100:.1f}%)")
    if 'return_next_q' in df_final.columns:
        print(f"  return_next_q: {df_final['return_next_q'].notna().sum()}/{len(df_final)} ({df_final['return_next_q'].notna().mean()*100:.1f}%)")
    print(f"  log_total_assets: {df_final['log_total_assets'].notna().sum()}/{len(df_final)} ({df_final['log_total_assets'].notna().mean()*100:.1f}%)")
    print(f"  leverage: {df_final['leverage'].notna().sum()}/{len(df_final)} ({df_final['leverage'].notna().mean()*100:.1f}%)")
    print(f"  roa: {df_final['roa'].notna().sum()}/{len(df_final)} ({df_final['roa'].notna().mean()*100:.1f}%)")
    print(f"  asset_growth: {df_final['asset_growth'].notna().sum()}/{len(df_final)} ({df_final['asset_growth'].notna().mean()*100:.1f}%)")

    print("="*80)

    # Display sample
    print("\nSample data (first 10 rows):")
    display_cols = ['ticker', 'quarter_end', 'fiscal_quarter', 'accession_number', 'log_total_assets', 'leverage', 'roa', 'return_next_q']
    display_cols = [c for c in display_cols if c in df_final.columns]
    print(df_final[display_cols].head(10))
else:
    print("⚠️ No panel data to save")

print("\n" + "="*80)
print("SCRIPT COMPLETE")
print("="*80)
