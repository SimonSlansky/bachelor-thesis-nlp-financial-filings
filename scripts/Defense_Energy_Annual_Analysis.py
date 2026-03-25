"""
Defense & Energy Companies - Annual Financial Analysis

Analyzing annual (10-K) financial data for defense contractors and energy companies
with key financial metrics.

Key Features:
- Uses 10-K filings instead of 10-Q
- Annual fiscal year data
- 365-day duration for flow metrics
- 12-month forward returns
- Annual UCDP conflict data alignment


CRITICAL DATA QUALITY FIXES APPLIED:
1. Instant Metric Validation - Accept trivial durations (≤5 days)
2. Multi-Unit USD Support - Scan ["USD", "iso4217:USD", "usd"]
3. Amendment Filing Priority - Prefer 10-K/A over 10-K
4. Returns Minimum Trading Days - 200 days minimum
5. Transition Period Filtering - Remove partial-year filings
6. Asset Growth Calculation - Extract extra year for lagged calculation
"""

import requests
import time
import re
import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf

# =============================================================================
# CONFIGURATION
# =============================================================================
HEADERS = {"User-Agent": "Simon Slansky simon.slansky@outlook.com"}
REQUEST_SLEEP = 0.15

# Data extraction
NUM_YEARS_TO_EXTRACT = 16  # Extract 16 years to allow asset_growth calculation (will drop earliest year)

# Filing validation
MAX_FILING_LAG_DAYS = 180

MIN_VALID_YEARS = 5

# Target companies (Defense & Energy)
TARGET_TICKERS = [
    # Tier 1: Prime defense contractors
    "LMT",   # Lockheed Martin
    "RTX",   # RTX (Raytheon)
    "NOC",   # Northrop Grumman
    "GD",    # General Dynamics
    "LHX",   # L3Harris Technologies

    # Tier 2: Defense services & intelligence
    "BAH",   # Booz Allen Hamilton
    "LDOS",  # Leidos
    "CACI",  # CACI International
    "PSN",   # Parsons
    "HII",   # Huntington Ingalls

    # Tier 3: Aerospace / military platforms
    "TDG",   # TransDigm

    # Tier 4: Emerging / specialized defense tech
    "KTOS",  # Kratos Defense
    "AVAV",  # AeroVironment
    "MRCY",  # Mercury Systems
    "TXT",   # Textron
    "CW"     # Curtiss-Wright
]

# Paths
BASE_DIR = Path.cwd().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

print(f"✓ Configuration loaded")
print(f"  Target companies: {len(TARGET_TICKERS)}")
print(f"  Years to extract: {NUM_YEARS_TO_EXTRACT} (earliest year used only for asset_growth calculation)")
print(f"  Data directory: {DATA_DIR}")

# =============================================================================
# FINANCIAL METRICS DEFINITION
# =============================================================================

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

    # Net Income (Flow metric - annual)
    ('NetIncomeLoss', 'net_income', 1),
    ('ProfitLoss', 'net_income', 2),
    ('NetIncomeLossAvailableToCommonStockholdersBasic', 'net_income', 3),
    ('IncomeLossFromContinuingOperations', 'net_income', 4),
    ('NetIncomeLossAttributableToParent', 'net_income', 5),
    ('IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest', 'net_income', 6)
]

# Classify metrics as flow (need start/end) or instant (point-in-time)
FLOW_METRICS = {"net_income"}
INSTANT_METRICS = {m for _, m, _ in METRICS_WITH_PRIORITY if m not in FLOW_METRICS}

print(f"✓ Defined {len(METRICS_WITH_PRIORITY)} XBRL tag mappings")
print(f"  Flow metrics (need date range): {FLOW_METRICS}")
print(f"  Instant metrics (point-in-time): {len(INSTANT_METRICS)} metrics")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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


def determine_fiscal_year(end_date, fye_month):
    """Determine fiscal year from year-end date and fiscal year-end month."""
    end_month = end_date.month

    # If year-end is after FYE month, fiscal year is next calendar year
    if end_month > fye_month:
        fiscal_year = end_date.year + 1
    else:
        fiscal_year = end_date.year

    return fiscal_year


def is_valid_flow(row, fye_month):
    """Validate FLOW metrics (annual only, 10-K).

    Rules:
    - Must come from 10-K or 10-K/A
    - Must have valid start and end dates
    - Duration must be close to 1 year (350–380 days)
    - Explicitly reject transition / stub periods
    """
    # Must be from 10-K filing
    if row.get("form") not in {"10-K", "10-K/A"}:
        return False

    # Must have valid end date
    end = pd.to_datetime(row.get("end"), errors="coerce")
    if pd.isna(end):
        return False

    # Must have start date
    start = row.get("start")
    if not start:
        return False

    # Duration should be ~365 days (annual)
    start = pd.to_datetime(start, errors="coerce")
    if pd.isna(start):
        return False

    days = (end - start).days
    if not (350 <= days <= 380):
        return False

    return True


def is_valid_instant(row, fye_month):
    """Validate INSTANT metrics (balance sheet items).

    CRITICAL LOGIC:
    - SEC XBRL may include start dates for instant facts
    - Accept:
        1) No start date
        2) Trivial duration (≤ 5 days)
    - Reject:
        - Durations > 5 days (likely flow / period aggregates)

    NOTE: We do NOT enforce strict fiscal year-end month alignment
    because some companies report balance sheets with 1-2 day shifts
    due to weekends/holidays, which is acceptable for annual data.
    """
    if row.get("form") not in {"10-K", "10-K/A"}:
        return False

    end = pd.to_datetime(row.get("end"), errors="coerce")
    if pd.isna(end):
        return False

    start = row.get("start")

    # No start date = pure instant (ideal case)
    if start is None:
        return True

    # Has start date - check if duration is trivial
    start = pd.to_datetime(start, errors="coerce")
    if pd.isna(start):
        return True

    days = (end - start).days

    # Reject if duration > 5 days (likely a flow metric)
    if days > 5:
        return False

    return True


print("✓ XBRL validation functions defined")
print("  ✓ Flow metrics: annual only (350–380 days)")
print("  ✓ Instant metrics: accept trivial durations (≤5 days)")
print("  ✓ Transition / stub periods explicitly rejected")
print("  ⚠️  FYE alignment check REMOVED to prevent excessive data loss")

# =============================================================================
# STEP 7: EXTRACT ANNUAL FINANCIAL DATA
# =============================================================================
def extract_annual_data(meta: dict) -> pd.DataFrame:
    """Extract annual financial data from 10-K filings.

    GUARANTEES:
    - Annual-only flow metrics (no transition / stub periods)
    - Instant metrics accept SEC trivial-duration artifacts
    - Multi-unit USD support
    - Amendment (10-K/A) priority over 10-K
    - One observation per fiscal year-end
    """
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
        'after_filing_date_check': 0,
        'after_filing_lag_validation': 0,
        'unique_years': 0,
        'after_year_limit': 0
    }

    # Dictionary to store records by year_end date
    records = {}

    # Process each metric with priority handling
    for tag, metric, priority in METRICS_WITH_PRIORITY:
        tag_data = us_gaap.get(tag, {}).get("units", {})

        # CRITICAL FIX: Multi-unit USD support to prevent data loss
        facts = []
        for unit_key in ["USD", "iso4217:USD", "usd"]:
            if unit_key in tag_data:
                facts.extend(tag_data[unit_key])

        if not facts:
            continue

        for fact in facts:
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

            # Get year end date and fiscal year
            end = pd.to_datetime(fact["end"])
            fy = determine_fiscal_year(end, meta["fiscal_year_end"])

            # Check filing date
            filing_date = pd.to_datetime(fact.get("filed"), errors="coerce")
            if pd.isna(filing_date):
                continue

            stats['after_filing_date_check'] += 1

            # Validate filing lag
            if (filing_date - end).days > MAX_FILING_LAG_DAYS:
                continue

            stats['after_filing_lag_validation'] += 1

            # CRITICAL FIX: Amendment filing priority
            form_type = fact.get("form", "")
            is_amendment = form_type.endswith("/A")

            # Create or update record for this year
            if end not in records:
                records[end] = {
                    "ticker": meta["ticker"],
                    "company_name": meta["company_name"],
                    "year_end": end,
                    "filing_date": filing_date,
                    "fiscal_year": fy,
                    "__is_amendment": is_amendment
                }
            else:
                # Amendment filing priority: 10-K/A over 10-K, then latest filing
                existing_is_amendment = records[end].get("__is_amendment", False)
                existing_filing_date = records[end]['filing_date']

                # Skip if current is not amendment but existing is
                if not is_amendment and existing_is_amendment:
                    continue
                # Replace if current is amendment but existing is not
                elif is_amendment and not existing_is_amendment:
                    records[end]['filing_date'] = filing_date
                    records[end]['__is_amendment'] = True
                # If both same type, keep earlier filing
                elif filing_date > existing_filing_date:
                    continue

            # Only set metric if not already set OR if this has higher priority
            rec = records[end]
            if metric not in rec or priority < rec.get(f"{metric}__priority", 999):
                rec[metric] = fact["val"]
                rec[f"{metric}__priority"] = priority

    stats['unique_years'] = len(records)

    df = pd.DataFrame(list(records.values()))

    # Remove priority tracking columns and amendment flag
    cols_to_drop = [col for col in df.columns if col.endswith("__priority") or col == "__is_amendment"]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Limit to most recent N years
    if not df.empty and len(df) > NUM_YEARS_TO_EXTRACT:
        df = df.sort_values('year_end', ascending=False).head(NUM_YEARS_TO_EXTRACT)
        df = df.sort_values('year_end')  # Restore chronological order

    stats['after_year_limit'] = len(df)

    # Print statistics
    print(f"    Data cleaning pipeline:")
    print(f"      1. Total XBRL facts: {stats['total_facts']:,}")

    if stats['total_facts'] > 0:
        print(f"      2. After metric type validation: {stats['after_metric_type_validation']:,} "
              f"({stats['after_metric_type_validation']/stats['total_facts']*100:.1f}% of raw)")

    if stats['after_metric_type_validation'] > 0:
        print(f"      3. After filing date check: {stats['after_filing_date_check']:,} "
              f"({stats['after_filing_date_check']/stats['after_metric_type_validation']*100:.1f}% retained)")

    if stats['after_filing_date_check'] > 0:
        print(f"      4. After filing lag validation (≤{MAX_FILING_LAG_DAYS} days): {stats['after_filing_lag_validation']:,} "
              f"({stats['after_filing_lag_validation']/stats['after_filing_date_check']*100:.1f}% retained)")

    print(f"      5. Unique years created: {stats['unique_years']}")
    print(f"      6. After limiting to {NUM_YEARS_TO_EXTRACT} most recent: {stats['after_year_limit']}")

    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""

    # Step 1: Load SEC Tickers
    print("\n" + "="*80)
    print("STEP 1: Loading SEC Tickers")
    print("="*80)

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

    # Step 2: Load SEC Submissions Metadata
    print("\n" + "="*80)
    print("STEP 2: Loading SEC Submissions Metadata")
    print("="*80)

    companies = []
    failed_ciks = []

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
            fiscal_year_end = int(fye_raw[:2])  # Extract month from MMDD format
        except:
            print(f"  ⚠️ Invalid fiscal year-end format: {ticker} ({fye_raw})")
            failed_ciks.append((ticker, cik))
            time.sleep(REQUEST_SLEEP)
            continue

        # Get exchange info (optional)
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

        print(f"  ✓ {ticker}: {sub.get('name', 'N/A')} | FYE: {fye_raw}")
        time.sleep(REQUEST_SLEEP)

    target_companies = {c["ticker"]: c for c in companies}

    print(f"\n✓ Successfully loaded: {len(target_companies)}/{len(TARGET_TICKERS)} companies")
    if failed_ciks:
        print(f"⚠️ Failed to load {len(failed_ciks)} companies:")
        for ticker, cik in failed_ciks:
            print(f"    {ticker} (CIK {cik})")

    # Step 3: Extract Annual Financial Data
    print("\n" + "="*80)
    print("STEP 3: Extracting Annual Financial Data")
    print("="*80)

    all_years = []
    extraction_stats = []

    for ticker, meta in sorted(target_companies.items()):
        print(f"\nProcessing {ticker} ({meta['company_name']})...")

        df = extract_annual_data(meta)

        if df.empty:
            print(f"  ⚠️ No annual data found")
            extraction_stats.append({
                'ticker': ticker,
                'years_found': 0,
                'status': 'NO_DATA'
            })
        elif len(df) < MIN_VALID_YEARS:
            print(f"  ⚠️ Insufficient data: {len(df)} years (min {MIN_VALID_YEARS} required)")
            extraction_stats.append({
                'ticker': ticker,
                'years_found': len(df),
                'status': 'INSUFFICIENT'
            })
        else:
            all_years.append(df)
            print(f"  ✓ Extracted {len(df)} years")
            extraction_stats.append({
                'ticker': ticker,
                'years_found': len(df),
                'status': 'SUCCESS'
            })

        time.sleep(REQUEST_SLEEP)

    # Combine all data
    if len(all_years) == 0:
        df_all = pd.DataFrame()
        print("⚠️ WARNING: No companies passed extraction!")
        return
    else:
        df_all = pd.concat(all_years, ignore_index=True)
        print(f"\n✓ EXTRACTION COMPLETE")
        print(f"  Total firm-years: {len(df_all)}")
        print(f"  Companies with data: {df_all['ticker'].nunique()}/{len(target_companies)}")
        print(f"  Date range: {df_all['year_end'].min()} to {df_all['year_end'].max()}")

    # Step 4: Data Quality Check
    print("\n" + "="*80)
    print("STEP 4: Data Quality Check")
    print("="*80)

    core_metrics = ['total_assets', 'total_liabilities', 'net_income']

    print("\nMissing values by metric:")
    for metric in core_metrics:
        if metric in df_all.columns:
            missing = df_all[metric].isna().sum()
            pct = missing / len(df_all) * 100
            print(f"  {metric}: {missing}/{len(df_all)} ({pct:.1f}% missing)")

    # Step 5: Save Raw Extraction
    print("\n" + "="*80)
    print("STEP 5: Saving Raw Extraction")
    print("="*80)

    output_path = DATA_DIR / 'annual_financials.csv'
    df_all.to_csv(output_path, index=False)
    print(f"✓ Raw data saved to: {output_path}")

    # Step 5.5: Filter Out Transition Periods & Resolve Duplicates
    print("\n" + "="*80)
    print("STEP 5.5: Filter Out Transition Periods & Resolve Duplicates")
    print("="*80)

    df_all = df_all.sort_values(['ticker', 'year_end']).reset_index(drop=True)

    before_transition_filter = len(df_all)

    # Calculate days between consecutive year-ends
    df_all['year_end_dt'] = pd.to_datetime(df_all['year_end'])
    df_all['days_since_last'] = df_all.groupby('ticker')['year_end_dt'].diff().dt.days
    df_all['days_from_365'] = abs(df_all['days_since_last'] - 365)
    df_all['has_net_income'] = df_all['net_income'].notna().astype(int)

    # Detect duplicate filing dates
    df_all['filing_date_str'] = df_all['filing_date'].astype(str)
    df_all['is_duplicate'] = df_all.duplicated(subset=['ticker', 'filing_date_str'], keep=False)

    duplicates = df_all[df_all['is_duplicate']].copy()

    if len(duplicates) > 0:
        print("DUPLICATE FILING DATES DETECTED")
        print("="*80)
        print(f"Found {len(duplicates)} observations with duplicate filing dates\n")

        for (ticker, filing_date), group in duplicates.groupby(['ticker', 'filing_date_str']):
            print(f"\n{ticker} | filing_date={filing_date}:")
            for _, row in group.iterrows():
                net_inc_status = "✓" if row['has_net_income'] else "✗"
                days_status = f"{row['days_since_last']:.0f} days" if pd.notna(row['days_since_last']) else "N/A"
                print(f"    year_end={row['year_end']} | net_income={net_inc_status} | "
                      f"period={days_status} | deviation={row['days_from_365']:.0f}")

        print("\n⚠️  For each duplicate, will keep the observation with:")
        print("     1. Non-null net_income (if available)")
        print("     2. Period closest to 365 days (full annual period)")
        print("="*80)

        # Score duplicates: prioritize complete data, then annual periods
        df_all['score'] = df_all['has_net_income'] * 10000 - df_all['days_from_365']

        # Keep best observation per duplicate group
        df_all['keep'] = False
        for (ticker, filing_date), group in df_all[df_all['is_duplicate']].groupby(['ticker', 'filing_date_str']):
            best_idx = group['score'].idxmax()
            df_all.loc[best_idx, 'keep'] = True

        # Mark non-duplicates as keep
        df_all.loc[~df_all['is_duplicate'], 'keep'] = True

        removed_duplicates = (~df_all['keep']).sum()
        df_all = df_all[df_all['keep']].copy()

        print(f"\n✓ Resolved {removed_duplicates} duplicate observations")
    else:
        print("✓ No duplicate filing dates found\n")
        removed_duplicates = 0

    # Filter transition periods (non-annual periods)
    df_all['is_transition'] = (
        (df_all['days_since_last'] < 300) |
        (df_all['days_since_last'] > 400)
    )

    df_all['is_transition'] = df_all['is_transition'].fillna(False)

    transition_periods = df_all[df_all['is_transition']]
    if len(transition_periods) > 0:
        print("\nREMAINING TRANSITION PERIODS")
        print("="*80)
        print(f"Found {len(transition_periods)} additional transition period(s):\n")
        for _, row in transition_periods.iterrows():
            print(f"  {row['ticker']}: year_end={row['year_end']}, "
                  f"filing_date={row['filing_date']}, "
                  f"days_since_last={row['days_since_last']:.0f}")
        print("\n⚠️  These will also be excluded (non-annual periods)")
        print("="*80)

        df_all = df_all[~df_all['is_transition']].copy()
        removed_transitions = len(transition_periods)
    else:
        removed_transitions = 0

    # Clean up helper columns
    cols_to_drop = ['year_end_dt', 'days_since_last', 'days_from_365',
                    'has_net_income', 'filing_date_str', 'is_duplicate', 'is_transition']
    if 'score' in df_all.columns:
        cols_to_drop.extend(['score', 'keep'])

    df_all = df_all.drop(columns=cols_to_drop)

    after_transition_filter = len(df_all)
    total_removed = before_transition_filter - after_transition_filter

    print(f"\n✓ TRANSITION FILTER COMPLETE")
    print(f"  Duplicate filing dates resolved: {removed_duplicates}")
    print(f"  Non-annual periods removed: {removed_transitions}")
    print(f"  Total removed: {total_removed}")
    print(f"  Panel size: {before_transition_filter} → {after_transition_filter} observations\n")

    # Step 6: Compute Missing Metrics from Components
    print("\n" + "="*80)
    print("STEP 6: Computing Missing Metrics from Components")
    print("="*80)

    missing_before_assets = df_all['total_assets'].isna().sum()
    missing_before_liab = df_all['total_liabilities'].isna().sum()

    # Initialize tag columns
    if 'total_assets__tag' not in df_all.columns:
        df_all['total_assets__tag'] = None
    if 'total_liabilities__tag' not in df_all.columns:
        df_all['total_liabilities__tag'] = None

    # Compute missing total assets
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

    # Compute missing total liabilities
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

    missing_after_assets = df_all['total_assets'].isna().sum()
    missing_after_liab = df_all['total_liabilities'].isna().sum()

    computed_assets = missing_before_assets - missing_after_assets
    computed_liab = missing_before_liab - missing_after_liab

    print(f"total_assets: {computed_assets} computed, {missing_after_assets} still missing")
    print(f"total_liabilities: {computed_liab} computed, {missing_after_liab} still missing")

    # Step 7: Time-Series Imputation
    print("\n" + "="*80)
    print("STEP 7: Time-Series Imputation")
    print("="*80)

    df_all = df_all.sort_values(['ticker', 'year_end']).reset_index(drop=True)

    balance_sheet_metrics = ['total_assets', 'total_liabilities']

    for metric in balance_sheet_metrics:
        if metric in df_all.columns:
            before = df_all[metric].isna().sum()
            df_all[metric] = df_all.groupby('ticker')[metric].ffill(limit=1)
            df_all[metric] = df_all.groupby('ticker')[metric].bfill(limit=1)
            after = df_all[metric].isna().sum()
            filled = before - after
            print(f"✓ {metric}: filled {filled} values (±1 year)")

    # Step 8: Time-window filtering (no external coverage filter applied)
    print("\n" + "="*80)
    print("STEP 8: Time-window filtering")
    print("="*80)

    # Previously this step filtered to match external UCDP coverage limits.
    # UCDP integration has been removed. Keep all available years from filings.
    print(f"✓ No external coverage filter applied; keeping all years ({len(df_all)} rows)")

    # Step 9: Calculate Stock Returns
    print("\n" + "="*80)
    print("STEP 9: Calculating Stock Returns")
    print("="*80)

    RETURN_WINDOW_DAYS = 365  # 365 calendar days ≈ 252 trading days
    POST_FILING_LAG = 2
    MIN_TRADING_DAYS = 200  # CRITICAL FIX: Require minimum trading days for valid returns
    MIN_FIRM_COVERAGE = 0.6

    results = []
    firm_stats = []

    min_date = pd.to_datetime(df_all['year_end'].min())
    max_date = pd.to_datetime(df_all['year_end'].max())

    print(f"⚠️  CRITICAL FIX APPLIED: MIN_TRADING_DAYS = {MIN_TRADING_DAYS}")
    print(f"   Return window: {RETURN_WINDOW_DAYS} calendar days ≈ 252 trading days")
    print(f"   This ensures returns are calculated from adequate trading data\n")

    for ticker in sorted(df_all['ticker'].unique()):
        print(f"Processing {ticker}...", end=" ")

        stock = yf.download(
            ticker,
            start=min_date,
            end=max_date + pd.Timedelta(days=450),
            progress=False,
            auto_adjust=True
        )

        if stock.empty or 'Close' not in stock.columns:
            print("No price data")
            continue

        stock = stock[['Close']].dropna()
        firm_years = df_all[df_all['ticker'] == ticker]
        usable = 0

        for _, row in firm_years.iterrows():
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
                'year_end': row['year_end'],
                'return_next_year': log_return
            })

        firm_stats.append({
            'ticker': ticker,
            'years_total': len(firm_years),
            'years_with_returns': usable,
            'coverage_ratio': usable / len(firm_years) if len(firm_years) > 0 else 0
        })

        print(f"{usable}/{len(firm_years)} years")

    df_stock = pd.DataFrame(results)
    df_coverage = pd.DataFrame(firm_stats)

    if not df_stock.empty:
        usable_firms = df_coverage[df_coverage['coverage_ratio'] >= MIN_FIRM_COVERAGE]['ticker']
        df_stock = df_stock[df_stock['ticker'].isin(usable_firms)]

        print(f"\n✓ Returns calculated")
        print(f"  Firms with returns: {df_stock['ticker'].nunique()}")
        print(f"  Total observations: {len(df_stock)}")

    # Step 10: Merge Stock Returns
    print("\n" + "="*80)
    print("STEP 10: Merging Stock Returns")
    print("="*80)

    if not df_stock.empty:
        df_panel = df_all.merge(
            df_stock[['ticker', 'year_end', 'return_next_year']],
            on=['ticker', 'year_end'],
            how='left'
        )
        print(f"✓ Merged stock returns")
        print(f"  Panel: {len(df_panel)} rows")
        print(f"  Rows with returns: {df_panel['return_next_year'].notna().sum()}")
    else:
        df_panel = df_all.copy()
        df_panel['return_next_year'] = np.nan

    # Step 11: Conflict data columns (UCDP integration removed)
    print("\n" + "="*80)
    print("STEP 11: Conflict data columns (placeholder)")
    print("="*80)

    # UCDP integration and associated data files were removed per project cleanup.
    # Create placeholder columns with NaN so downstream code keeps working.
    df_panel['num_events'] = np.nan
    df_panel['num_active_conflicts'] = np.nan
    df_panel['num_fatalities'] = np.nan
    print("✓ UCDP integration removed; conflict columns added as NaN placeholders")

    # Step 12: Calculate Financial Metrics
    print("\n" + "="*80)
    print("STEP 12: Calculating Financial Metrics")
    print("="*80)

    df_panel = df_panel.sort_values(['ticker', 'year_end']).reset_index(drop=True)

    # Firm size
    df_panel['log_total_assets'] = np.where(
        df_panel['total_assets'] > 0,
        np.log(df_panel['total_assets']),
        np.nan
    )

    # Leverage
    df_panel['leverage'] = np.where(
        df_panel['total_assets'] > 0,
        df_panel['total_liabilities'] / df_panel['total_assets'],
        np.nan
    )

    # ROA
    df_panel['roa'] = np.where(
        df_panel['total_assets'] > 0,
        df_panel['net_income'] / df_panel['total_assets'],
        np.nan
    )

    # Asset growth
    df_panel['total_assets_lag'] = df_panel.groupby('ticker')['total_assets'].shift(1)

    df_panel['asset_growth'] = np.where(
        df_panel['total_assets_lag'] > 0,
        (df_panel['total_assets'] - df_panel['total_assets_lag']) / df_panel['total_assets_lag'],
        np.nan
    )

    df_panel = df_panel.drop(columns=['total_assets_lag'])

    # Winsorize asset_growth at 1% and 99% percentiles to handle outliers
    if 'asset_growth' in df_panel.columns:
        asset_growth_notna = df_panel['asset_growth'].dropna()
        if len(asset_growth_notna) > 0:
            p01 = asset_growth_notna.quantile(0.01)
            p99 = asset_growth_notna.quantile(0.99)
            df_panel['asset_growth'] = df_panel['asset_growth'].clip(lower=p01, upper=p99)
            print(f"✓ Winsorized asset_growth at 1% ({p01:.3f}) and 99% ({p99:.3f}) percentiles")

    # CRITICAL FIX: Drop earliest year per company (used only for asset_growth calculation)
    before_drop = len(df_panel)

    earliest_years = df_panel.groupby('ticker')['year_end'].min().reset_index()
    earliest_years.columns = ['ticker', 'earliest_year']

    df_panel = df_panel.merge(earliest_years, on='ticker', how='left')
    df_panel = df_panel[df_panel['year_end'] != df_panel['earliest_year']].copy()
    df_panel = df_panel.drop(columns=['earliest_year'])

    after_drop = len(df_panel)
    dropped = before_drop - after_drop

    print(f"✓ Dropped {dropped} earliest years (1 per company) used for asset_growth calculation")
    print(f"  Panel size: {before_drop} → {after_drop} observations\n")

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

    # Step 13: Save Final Panel Dataset
    print("\n" + "="*80)
    print("STEP 13: Saving Final Panel Dataset")
    print("="*80)

    essential_cols = [
        'ticker', 'company_name', 'year_end', 'filing_date',
        'fiscal_year',
        'net_income', 'total_assets', 'total_liabilities',
        'return_next_year',
        'num_events', 'num_active_conflicts', 'num_fatalities',
        'log_total_assets', 'leverage', 'roa', 'asset_growth'
    ]

    final_cols = [col for col in essential_cols if col in df_panel.columns]
    df_final = df_panel[final_cols].copy()

    output_path = DATA_DIR / 'annual_panel_data_with_returns.csv'
    df_final.to_csv(output_path, index=False)

    print(f"✓ Saved to: {output_path}")
    print(f"\nDimensions:")
    print(f"  Companies: {df_final['ticker'].nunique()}")
    print(f"  Years: {len(df_final)}")
    print(f"  Date range: {df_final['year_end'].min()} to {df_final['year_end'].max()}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
