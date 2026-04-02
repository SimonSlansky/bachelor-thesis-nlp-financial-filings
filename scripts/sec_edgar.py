"""SEC EDGAR API: ticker resolution, company metadata, XBRL fact extraction."""

import time
import pandas as pd
import requests

from config import (
    SEC_HEADERS, REQUEST_SLEEP, MAX_FILING_LAG_DAYS,
    METRICS_WITH_PRIORITY, FLOW_METRICS,
    EXCLUDED_SIC_RANGE,
    ANNUAL_FLOW_RANGE,
)

# ── public helpers ─────────────────────────────────────────────────────────

def fetch_universe(n: int | None = None) -> pd.DataFrame:
    """Fetch the SEC company-tickers list (ordered by market cap).

    Returns a DataFrame with columns [ticker, cik].
    If *n* is given, return only the top *n* entries.
    """
    data = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS,
        timeout=15,
    ).json()
    # Preserve SEC ordering (market-cap ranked indices)
    rows = [data[str(i)] for i in range(len(data))]
    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].str.upper()
    df["cik"] = df["cik_str"].astype(str).str.zfill(10)
    df = df[["ticker", "cik"]]
    if n is not None:
        df = df.head(n)
    return df.reset_index(drop=True)


def load_company_metadata(sec_df: pd.DataFrame) -> dict[str, dict]:
    """Fetch SEC submissions for each row and return {ticker: metadata}.

    Skips non-operating entities (ETFs, funds) and excluded SIC industries.
    Metadata dict keys: ticker, cik, company_name, fiscal_year_end, sic.
    """
    companies: dict[str, dict] = {}
    skipped_type = skipped_sic = 0
    for _, row in sec_df.iterrows():
        ticker, cik = row["ticker"], row["cik"]
        sub = _fetch_submissions(cik)
        if sub is None:
            print(f"  ✗ {ticker} — failed to load submissions")
            time.sleep(REQUEST_SLEEP)
            continue

        # Filter: only operating companies (skip ETFs, funds, shells)
        if sub.get("entityType") != "operating":
            skipped_type += 1
            time.sleep(REQUEST_SLEEP)
            continue

        # Filter: exclude financial-sector SIC codes
        sic_raw = sub.get("sic", "")
        try:
            sic = int(sic_raw)
        except (ValueError, TypeError):
            sic = 0
        if sic in EXCLUDED_SIC_RANGE:
            skipped_sic += 1
            time.sleep(REQUEST_SLEEP)
            continue

        fye_raw = sub.get("fiscalYearEnd", "")
        if not isinstance(fye_raw, str) or len(fye_raw) < 2:
            print(f"  ✗ {ticker} — missing fiscal-year-end")
            time.sleep(REQUEST_SLEEP)
            continue
        try:
            fye_month = int(fye_raw[:2])
        except ValueError:
            print(f"  ✗ {ticker} — invalid fiscal-year-end: {fye_raw}")
            time.sleep(REQUEST_SLEEP)
            continue

        companies[ticker] = {
            "ticker": ticker,
            "cik": cik,
            "company_name": sub.get("name", ticker),
            "fiscal_year_end": fye_month,
            "sic": sic,
        }
        print(f"  ✓ {ticker}: {sub.get('name', '')}  FYE month={fye_month}")
        time.sleep(REQUEST_SLEEP)

    if skipped_type or skipped_sic:
        print(f"  Skipped: {skipped_type} non-operating, {skipped_sic} financial-sector")
    return companies


# ── XBRL extraction ───────────────────────────────────────────────────────

def extract_annual_facts(meta: dict, n_years: int) -> pd.DataFrame:
    """Extract annual (10-K) financial facts for one company.

    Returns one row per fiscal year-end with columns for each metric.

    Two-pass approach:
      Pass 1 — collect ALL valid facts grouped by (end_date, tag).
      Pass 2 — for each end_date, decide amendment vs original, then
               resolve metric priorities.
    This avoids the bug where an amendment discovered mid-loop clears
    metrics populated by earlier tags that the loop won't revisit.
    """
    us_gaap = _fetch_company_facts(meta)
    if us_gaap is None:
        return pd.DataFrame()

    fye = meta["fiscal_year_end"]

    # ── Pass 1: collect all valid facts ────────────────────────────────────
    # Key: (end_date, tag) → list of fact dicts
    from collections import defaultdict
    raw_facts: dict[pd.Timestamp, list[tuple[str, str, int, dict]]] = defaultdict(list)

    for tag, metric, priority in METRICS_WITH_PRIORITY:
        for fact in _usd_facts(us_gaap, tag):
            if metric in FLOW_METRICS:
                if not _valid_annual_flow(fact):
                    continue
            else:
                if not _valid_annual_instant(fact):
                    continue

            end = pd.to_datetime(fact["end"])
            filing_date = pd.to_datetime(fact.get("filed"), errors="coerce")
            if pd.isna(filing_date) or (filing_date - end).days > MAX_FILING_LAG_DAYS:
                continue

            raw_facts[end].append((tag, metric, priority, fact))

    # ── Pass 2: resolve amendments and priorities per end_date ─────────────
    records: dict[pd.Timestamp, dict] = {}

    for end, fact_list in raw_facts.items():
        # Determine whether we have an amendment for this period
        has_amendment = any(f.get("form", "").endswith("/A") for _, _, _, f in fact_list)

        # If amendment exists, keep only amendment facts; otherwise only originals
        if has_amendment:
            fact_list = [(t, m, p, f) for t, m, p, f in fact_list
                         if f.get("form", "").endswith("/A")]
        else:
            fact_list = [(t, m, p, f) for t, m, p, f in fact_list
                         if not f.get("form", "").endswith("/A")]

        if not fact_list:
            continue

        # filing_date = earliest amendment date (when corrected info first
        # became public), used to anchor the return-measurement window.
        all_filed = [pd.to_datetime(f.get("filed"), errors="coerce")
                     for _, _, _, f in fact_list]
        earliest_filing = min(d for d in all_filed if pd.notna(d))

        rec = {
            "ticker": meta["ticker"],
            "company_name": meta["company_name"],
            "year_end": end,
            "filing_date": earliest_filing,
            "fiscal_year": _fiscal_year(end, fye),
            "accession_number": None,  # set below from winning fact
        }

        # Resolve metric priorities: lowest priority number wins.
        # Among same-priority facts, prefer the LATEST filing — the
        # most recent amendment supersedes all prior values.
        for tag, metric, priority, fact in fact_list:
            fd = pd.to_datetime(fact.get("filed"), errors="coerce")
            cur_pri = rec.get(f"_p_{metric}", 999)
            cur_fd = rec.get(f"_fd_{metric}")
            if priority < cur_pri or (priority == cur_pri and cur_fd is not None and fd > cur_fd):
                rec[metric] = fact["val"]
                rec[f"_p_{metric}"] = priority
                rec[f"_tag_{metric}"] = tag
                rec[f"_fd_{metric}"] = fd
                rec["accession_number"] = fact.get("accn")

        records[end] = rec

    df = pd.DataFrame(records.values())
    if df.empty:
        return df
    # Keep tag provenance columns (prefixed _tag_) but drop internal helper cols
    df = df[[c for c in df.columns
             if not c.startswith("_p_") and not c.startswith("_fd_")]]
    df = df.sort_values("year_end").tail(n_years).reset_index(drop=True)
    return df


# ── private helpers ────────────────────────────────────────────────────────

def _fetch_submissions(cik: str) -> dict | None:
    try:
        r = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=SEC_HEADERS, timeout=10,
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _fetch_company_facts(meta: dict) -> dict | None:
    try:
        r = requests.get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{meta['cik']}.json",
            headers=SEC_HEADERS, timeout=15,
        )
        if r.status_code != 200:
            return None
        return r.json().get("facts", {}).get("us-gaap", {}) or None
    except Exception:
        return None


def _usd_facts(us_gaap: dict, tag: str) -> list[dict]:
    """Return USD- or USD/shares-denominated facts for *tag*."""
    units = us_gaap.get(tag, {}).get("units", {})
    facts: list[dict] = []
    for key in ("USD", "iso4217:USD", "usd",
                "USD/shares", "iso4217:USD/shares"):
        facts.extend(units.get(key, []))
    return facts


def _fiscal_year(end_date: pd.Timestamp, fye_month: int) -> int:
    m = end_date.month
    # Tolerate a few days' overshoot past the nominal FYE-month boundary.
    # E.g. FYE=Jan, end_date=Feb 2 → still the same fiscal year.
    # For FYE=Dec, early-January dates (52/53-week filers) belong to the
    # prior calendar year's fiscal year.
    next_m = (fye_month % 12) + 1
    adjusted = False
    if m == next_m and end_date.day <= 5:
        m = fye_month
        adjusted = True
    fy = end_date.year + 1 if m > fye_month else end_date.year
    # When the tolerance fires and the raw month was in a later calendar year
    # than the FYE month implies, subtract 1 (only matters for FYE=Dec→Jan).
    if adjusted and end_date.month < fye_month:
        fy -= 1
    return fy


def _fiscal_quarter_and_year(end_date: pd.Timestamp, fye_month: int) -> tuple[int, str | None]:
    diff = (end_date.month - fye_month) % 12
    qmap = {3: "Q1", 6: "Q2", 9: "Q3", 0: "Q4"}
    fq = qmap.get(diff)
    fy = _fiscal_year(end_date, fye_month)
    return fy, fq


# ── XBRL fact validators (annual) ─────────────────────────────────────────

def _valid_annual_flow(fact: dict) -> bool:
    if fact.get("form") not in ("10-K", "10-K/A"):
        return False
    start = pd.to_datetime(fact.get("start"), errors="coerce")
    end = pd.to_datetime(fact.get("end"), errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return False
    lo, hi = ANNUAL_FLOW_RANGE
    return lo <= (end - start).days <= hi


def _valid_annual_instant(fact: dict) -> bool:
    if fact.get("form") not in ("10-K", "10-K/A"):
        return False
    if pd.isna(pd.to_datetime(fact.get("end"), errors="coerce")):
        return False
    start = fact.get("start")
    if start is None:
        return True
    s = pd.to_datetime(start, errors="coerce")
    if pd.isna(s):
        return True
    return (pd.to_datetime(fact["end"]) - s).days <= 5
