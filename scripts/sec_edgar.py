"""SEC EDGAR API: ticker resolution, company metadata, XBRL fact extraction."""

import time
import pandas as pd
import requests

from config import (
    SEC_HEADERS, REQUEST_SLEEP, MAX_FILING_LAG_DAYS,
    METRICS_WITH_PRIORITY, FLOW_METRICS,
    EXCLUDED_SIC_RANGE,
    ANNUAL_FLOW_RANGE, SHARE_TAGS,
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
    facts_all = _fetch_company_facts(meta)
    if facts_all is None:
        return pd.DataFrame()
    us_gaap = facts_all.get("us-gaap") or {}
    if not us_gaap:
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

    # ── Pass 2: point-in-time resolution per end_date ─────────────────────
    # The panel must only contain information that was public when the
    # ORIGINAL 10-K was filed, because the volatility window is anchored on
    # that date.  Therefore:
    #   * values, filing_date and accession come from the original 10-K;
    #   * a 10-K/A is used only when no original 10-K facts exist for the
    #     period (then everything, including the date, comes from the 10-K/A,
    #     so values and timing stay consistent);
    #   * restated values published later in a 10-K/A are NOT mixed in
    #     (that would be look-ahead).
    records: dict[pd.Timestamp, dict] = {}

    for end, fact_list in raw_facts.items():
        originals = [x for x in fact_list
                     if not x[3].get("form", "").endswith("/A")]
        amendments = [x for x in fact_list
                      if x[3].get("form", "").endswith("/A")]
        use_amend = not originals
        chosen = amendments if use_amend else originals
        if not chosen:
            continue

        filed = [pd.to_datetime(f.get("filed"), errors="coerce")
                 for _, _, _, f in chosen]
        filed_ok = [d for d in filed if pd.notna(d)]
        if not filed_ok:
            continue
        first_filed = min(filed_ok)
        # Keep only facts from the first filing of this form type (same
        # day); later same-form facts would be look-ahead.
        chosen = [x for x, d in zip(chosen, filed) if d == first_filed]
        accns = pd.Series([f.get("accn") for _, _, _, f in chosen]).dropna()
        accn = accns.mode().iloc[0] if len(accns) else None

        rec = {
            "ticker": meta["ticker"],
            "company_name": meta["company_name"],
            "year_end": end,
            "filing_date": first_filed,
            "fiscal_year": _fiscal_year(end, fye),
            "accession_number": accn,
            "form_type": "10-K/A" if use_amend else "10-K",
            "has_amendment": bool(amendments) and not use_amend,
        }

        # Resolve metric priorities: lowest priority number wins; facts
        # come from the single chosen filing, so values are point-in-time.
        for tag, metric, priority, fact in chosen:
            if fact.get("accn") != accn:
                continue
            cur_pri = rec.get(f"_p_{metric}", 999)
            if priority < cur_pri:
                rec[metric] = fact["val"]
                rec[f"_p_{metric}"] = priority
                rec[f"_tag_{metric}"] = tag

        records[end] = rec

    # Share counts, matched to the chosen filing by accession number.
    shares_by_accn = _shares_by_accession(facts_all)
    for rec in records.values():
        val, src = shares_by_accn.get(rec.get("accession_number"), (None, None))
        rec["shares_out"] = val
        rec["_tag_shares_out"] = src

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
        return r.json().get("facts", {}) or None
    except Exception:
        return None


def _shares_by_accession(facts_all: dict) -> dict[str, tuple[float, str]]:
    """{accession: (shares outstanding, source tag)} from 10-K / 10-K/A facts.

    Tags are tried in ``SHARE_TAGS`` order; within one accession and tag the
    latest-dated instant is used (cover-page date for dei).  Multi-class
    filers report one dei fact per class with the same date; these are summed.
    """
    out: dict[str, tuple[float, str]] = {}
    for ns, tag in SHARE_TAGS:
        units = (facts_all.get(ns) or {}).get(tag, {}).get("units", {})
        facts = units.get("shares", [])
        by_accn: dict[str, dict[str, float]] = {}
        for f in facts:
            if f.get("form") not in ("10-K", "10-K/A"):
                continue
            accn, end, val = f.get("accn"), f.get("end"), f.get("val")
            if not accn or end is None or val is None:
                continue
            by_accn.setdefault(accn, {})
            by_accn[accn][end] = by_accn[accn].get(end, 0.0) + float(val)
        for accn, by_end in by_accn.items():
            if accn in out:
                continue
            v = by_end[max(by_end)]
            if v > 0:
                out[accn] = (v, f"{ns}:{tag}")
    return out


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
