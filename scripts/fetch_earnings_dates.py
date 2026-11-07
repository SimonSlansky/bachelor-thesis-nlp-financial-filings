"""Earnings-announcement dates from EDGAR 8-K filings with Item 2.02.

Item 2.02 ("Results of Operations and Financial Condition") is the 8-K
item under which US issuers furnish earnings releases (required since
2003), so its filing dates proxy for earnings-announcement dates.  The
full filing history is read from the Submissions API (recent block plus
the older paginated files).

Writes data/earnings_dates.csv  (ticker, cik, ea_date).  Resumable.
Usage:  python scripts/fetch_earnings_dates.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, SEC_HEADERS, REQUEST_SLEEP

OUT = DATA_DIR / "earnings_dates.csv"


def _get(url: str) -> dict | None:
    for attempt in range(4):
        try:
            r = requests.get(url, headers=SEC_HEADERS, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(2 ** attempt)
    return None


def _ea_from_block(b: dict) -> list[str]:
    forms, dates, items = b.get("form", []), b.get("filingDate", []), b.get("items", [])
    return [d for f, d, it in zip(forms, dates, items)
            if f in ("8-K", "8-K/A") and "2.02" in (it or "")]


def main() -> None:
    cmap = pd.read_csv(DATA_DIR / "cik_map.csv", dtype=str)
    tickers = set(pd.read_csv(DATA_DIR / "annual_panel.csv")["ticker"])
    cmap = cmap[cmap["ticker"].isin(tickers)]
    done = set(pd.read_csv(OUT)["ticker"]) if OUT.exists() else set()
    for i, (t, cik) in enumerate(zip(cmap["ticker"], cmap["cik"]), 1):
        if t in done:
            continue
        sub = _get(f"https://data.sec.gov/submissions/CIK{cik}.json")
        time.sleep(REQUEST_SLEEP)
        if sub is None:
            print(f"  {t}: submissions unavailable")
            continue
        dates = _ea_from_block(sub.get("filings", {}).get("recent", {}))
        for f in sub.get("filings", {}).get("files", []):
            b = _get(f"https://data.sec.gov/submissions/{f['name']}")
            time.sleep(REQUEST_SLEEP)
            if b:
                dates += _ea_from_block(b)
        rows = pd.DataFrame({"ticker": t, "cik": cik,
                             "ea_date": sorted(set(dates))})
        if rows.empty:  # keep a marker row so the firm is not refetched
            rows = pd.DataFrame({"ticker": [t], "cik": [cik], "ea_date": [None]})
        rows.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        if i % 50 == 0:
            print(f"  [{i}/{len(cmap)}] {t}: {len(dates)} EA dates")
    df = pd.read_csv(OUT)
    print(f"Saved {OUT.name}: {df['ticker'].nunique()} firms, "
          f"{df['ea_date'].notna().sum():,} dates")


if __name__ == "__main__":
    main()
