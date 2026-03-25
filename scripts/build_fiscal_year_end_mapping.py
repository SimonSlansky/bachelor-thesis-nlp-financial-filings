#!/usr/bin/env python3
"""
Build fiscal year-end mapping for ALL SEC EDGAR companies
using the authoritative submissions API.

Outputs:
- fiscal_year_end_mapping.json   (ticker -> month)
- fiscal_year_end_failures.json  (ticker -> metadata)
"""

import json
import requests
import time
from pathlib import Path
from collections import Counter
import pandas as pd

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
CONFIG_DIR.mkdir(exist_ok=True)

OUT_MAP = CONFIG_DIR / "fiscal_year_end_mapping.json"
OUT_FAIL = CONFIG_DIR / "fiscal_year_end_failures.json"

HEADERS = {
    "User-Agent": "Simon Slansky simon.slansky@outlook.com"
}

SLEEP = 0.2  # 5 req/sec (safe)

# -------------------------------
# Load tickers
# -------------------------------
def load_companies():
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=HEADERS,
        timeout=30
    )
    r.raise_for_status()

    data = r.json()
    return [
        {
            "ticker": v["ticker"],
            "cik": str(v["cik_str"]).zfill(10),
            "name": v["title"]
        }
        for v in data.values()
    ]


# -------------------------------
# Fetch submissions
# -------------------------------
def fetch_submissions(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        raise ValueError("submissions_not_found")
    return r.json()


# -------------------------------
# Extract fiscal year end
# -------------------------------
def extract_fye_month(sub):
    """
    Extract fiscal year-end month from SEC submissions JSON.

    Priority:
    1) Top-level 'fiscalYearEnd' (authoritative)
    2) Infer from most recent 10-K period
    """

    # --- PRIMARY: top-level fiscalYearEnd ---
    fye = sub.get("fiscalYearEnd")

    if fye and len(fye) == 4 and fye.isdigit():
        return int(fye[:2]), "submissions_fiscalYearEnd"

    # --- FALLBACK: infer from most recent 10-K ---
    recent = sub.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    periods = recent.get("period", [])

    for form, period in zip(forms, periods):
        if form == "10-K" and period:
            dt = pd.to_datetime(period, errors="coerce")
            if pd.notna(dt):
                return dt.month, "inferred_from_10K"

    raise ValueError("fiscal_year_end_not_found")


# -------------------------------
# Main build
# -------------------------------
def main():
    companies = load_companies()
    print(f"Loaded {len(companies)} companies")

    mapping = {}
    failures = {}

    for i, c in enumerate(companies, 1):
        ticker = c["ticker"]
        cik = c["cik"]

        try:
            sub = fetch_submissions(cik)
            month, source = extract_fye_month(sub)
            mapping[ticker] = month

            print(
                f"[{i:>5}/{len(companies)}] "
                f"{ticker:<6} ✓ FYE = {month:02d}  ({source})"
            )

        except Exception as e:
            mapping[ticker] = None
            failures[ticker] = {
                "cik": cik,
                "name": c["name"],
                "reason": str(e)
            }

            print(
                f"[{i:>5}/{len(companies)}] "
                f"{ticker:<6} ✗ FAILED → {e}"
            )

        time.sleep(SLEEP)

    # Save
    OUT_MAP.write_text(json.dumps(mapping, indent=2))
    OUT_FAIL.write_text(json.dumps(failures, indent=2))

    # Summary
    ok = sum(v is not None for v in mapping.values())
    print("\nDONE")
    print(f"Success: {ok}")
    print(f"Failed: {len(failures)}")

    dist = Counter(v for v in mapping.values() if v)
    print("\nFiscal year-end distribution:")
    for m in sorted(dist):
        print(f"  Month {m}: {dist[m]}")


if __name__ == "__main__":
    main()