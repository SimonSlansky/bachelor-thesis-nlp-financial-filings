"""Direct comparison of equivalent XBRL tags when BOTH are present for the same firm-year.

This is the gold-standard test: when SEC filings contain values under BOTH
tags for the same period, we can directly measure the difference.

The script queries the SEC EDGAR company-facts API for every firm in our
sample and collects periods where both tags coexist.

Run:  python tests/test_tag_overlap.py
"""

import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import numpy as np
import pandas as pd
import requests

from config import (
    DATA_DIR, SEC_HEADERS, REQUEST_SLEEP,
    EQUIVALENT_TAG_GROUPS, ANNUAL_FLOW_RANGE,
)

raw = pd.read_csv(DATA_DIR / "annual_financials.csv")
panel_tickers = sorted(raw["ticker"].unique())

# Build CIK lookup from SEC
print("Loading SEC ticker→CIK map …")
data = requests.get(
    "https://www.sec.gov/files/company_tickers.json",
    headers=SEC_HEADERS, timeout=15,
).json()
cik_map = {}
for v in data.values():
    cik_map[v["ticker"].upper()] = str(v["cik_str"]).zfill(10)

# Tag pairs to compare
TAG_PAIRS = {
    "operating_cash_flow": {
        "tags": [
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ],
        "flow": True,
    },
    "net_income": {
        "tags": [
            "NetIncomeLoss",
            "ProfitLoss",
        ],
        "flow": True,
    },
}


def _usd_facts(us_gaap: dict, tag: str) -> list[dict]:
    units = us_gaap.get(tag, {}).get("units", {})
    facts = []
    for key in ("USD", "iso4217:USD", "usd"):
        facts.extend(units.get(key, []))
    return facts


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
    return pd.notna(pd.to_datetime(fact.get("end"), errors="coerce"))


def fetch_company_facts(cik: str) -> dict | None:
    try:
        r = requests.get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
            headers=SEC_HEADERS, timeout=15,
        )
        return r.json().get("facts", {}).get("us-gaap", {}) if r.status_code == 200 else None
    except Exception:
        return None


# ── Main loop: fetch every firm and find overlapping tag periods ──────────

all_comparisons: list[dict] = []
firms_checked = 0
firms_with_overlap = 0

print(f"Scanning {len(panel_tickers)} firms for tag overlaps …\n")

for ticker in panel_tickers:
    cik = cik_map.get(ticker)
    if not cik:
        continue

    us_gaap = fetch_company_facts(cik)
    if us_gaap is None:
        time.sleep(REQUEST_SLEEP)
        continue

    firms_checked += 1
    firm_has_overlap = False

    for metric, cfg in TAG_PAIRS.items():
        tag_a, tag_b = cfg["tags"]
        is_flow = cfg["flow"]
        validator = _valid_annual_flow if is_flow else _valid_annual_instant

        # Collect valid annual facts per tag, keyed by end_date
        def _collect(tag: str) -> dict[str, list[dict]]:
            by_end: dict[str, list[dict]] = {}
            for fact in _usd_facts(us_gaap, tag):
                if not validator(fact):
                    continue
                end = fact["end"]
                by_end.setdefault(end, []).append(fact)
            return by_end

        facts_a = _collect(tag_a)
        facts_b = _collect(tag_b)

        # Find overlapping end_dates
        overlap_dates = set(facts_a.keys()) & set(facts_b.keys())
        if not overlap_dates:
            continue

        firm_has_overlap = True
        for end_date in sorted(overlap_dates):
            # Take the value from the latest filing for each tag
            best_a = max(facts_a[end_date], key=lambda f: f.get("filed", ""))
            best_b = max(facts_b[end_date], key=lambda f: f.get("filed", ""))

            val_a = best_a["val"]
            val_b = best_b["val"]
            diff = val_a - val_b
            pct_diff = abs(diff) / max(abs(val_a), abs(val_b), 1) * 100

            all_comparisons.append({
                "ticker": ticker,
                "end_date": end_date,
                "metric": metric,
                "tag_a": tag_a.split("NetCash")[-1] if "NetCash" in tag_a else tag_a,
                "val_a": val_a,
                "tag_b": tag_b.split("NetCash")[-1] if "NetCash" in tag_b else tag_b,
                "val_b": val_b,
                "diff": diff,
                "pct_diff": pct_diff,
                "filed_a": best_a.get("filed"),
                "filed_b": best_b.get("filed"),
            })

    if firm_has_overlap:
        firms_with_overlap += 1

    if firms_checked % 50 == 0:
        print(f"  … {firms_checked} firms checked, {firms_with_overlap} with overlaps, "
              f"{len(all_comparisons)} comparisons so far")

    time.sleep(REQUEST_SLEEP)

# ── Results ───────────────────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print(f"  RESULTS: {firms_checked} firms checked, {firms_with_overlap} with at least one overlap")
print(f"  Total firm-year-metric comparisons: {len(all_comparisons)}")
print(f"{'=' * 70}\n")

if not all_comparisons:
    print("No overlapping tag periods found.")
    sys.exit(0)

df = pd.DataFrame(all_comparisons)
df.to_csv(DATA_DIR / "diagnostics" / "tag_overlap_comparison.csv", index=False)

for metric in df["metric"].unique():
    sub = df[df["metric"] == metric]
    print(f"── {metric} ({len(sub)} comparisons across {sub['ticker'].nunique()} firms) ──")

    exact = (sub["diff"] == 0).sum()
    tiny = (sub["pct_diff"] < 0.01).sum()
    small = (sub["pct_diff"] < 1.0).sum()
    medium = ((sub["pct_diff"] >= 1.0) & (sub["pct_diff"] < 5.0)).sum()
    large = (sub["pct_diff"] >= 5.0).sum()

    print(f"  Exactly equal:     {exact:4d}  ({exact/len(sub)*100:.1f}%)")
    print(f"  < 0.01% diff:      {tiny:4d}  ({tiny/len(sub)*100:.1f}%)")
    print(f"  < 1% diff:         {small:4d}  ({small/len(sub)*100:.1f}%)")
    print(f"  1–5% diff:         {medium:4d}  ({medium/len(sub)*100:.1f}%)")
    print(f"  ≥ 5% diff:         {large:4d}  ({large/len(sub)*100:.1f}%)")
    print()
    print(f"  pct_diff distribution:")
    print(f"    mean   = {sub['pct_diff'].mean():.4f}%")
    print(f"    median = {sub['pct_diff'].median():.4f}%")
    print(f"    p90    = {sub['pct_diff'].quantile(.90):.4f}%")
    print(f"    p99    = {sub['pct_diff'].quantile(.99):.4f}%")
    print(f"    max    = {sub['pct_diff'].max():.4f}%")

    # Show the largest differences
    if large > 0:
        print(f"\n  Largest differences (≥5%):")
        top = sub.nlargest(min(10, large), "pct_diff")
        for _, r in top.iterrows():
            print(f"    {r['ticker']:6s} {r['end_date']}  "
                  f"{r['val_a']:>18,.0f} vs {r['val_b']:>18,.0f}  "
                  f"diff={r['pct_diff']:.2f}%")
    print()

print(f"Full comparison saved to data/diagnostics/tag_overlap_comparison.csv")
