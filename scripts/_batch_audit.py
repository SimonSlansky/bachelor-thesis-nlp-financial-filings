"""Batch extraction + self-audit for N random firm-years.

Usage:  python _batch_audit.py [--seed 42] [--n 100]

Extracts Item 1A and Item 7 for a random sample, runs heuristic quality
checks, and dumps a summary TSV plus detailed diagnostics for failures.
"""

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import pandas as pd

from config import DATA_DIR, REQUEST_SLEEP
from text_parser import (
    load_cik_map,
    _build_primary_doc_map,
    extract_filing_text,
)

FILING_TIMEOUT = 120  # seconds per filing

# ---------------------------------------------------------------------------
# Heuristic quality checks
# ---------------------------------------------------------------------------

# Patterns that indicate misidentified content
_FINANCIAL_STMT_RE = re.compile(
    r"\b(?:balance\s+sheets?|statements?\s+of\s+(?:income|operations|cash\s+flows)"
    r"|stockholders.?\s+equity|notes?\s+to\s+(?:consolidated\s+)?financial\s+statements)\b",
    re.IGNORECASE,
)
_NOTES_HEADER_RE = re.compile(
    r"(?m)^(?:Note|NOTE)\s+\d+[\s.:\-\u2014]",
)
_ITEM_HEADER_RE = re.compile(
    r"\bItem\s+\d+[A-Za-z]?\b", re.IGNORECASE,
)


def audit_section(text: str | None, section_key: str) -> dict:
    """Return a diagnostic dict for one extracted section."""
    result = {
        "status": "MISSING",
        "word_count": 0,
        "flags": [],
    }
    if not text:
        return result

    wc = len(text.split())
    result["word_count"] = wc
    result["head_80"] = text[:80].replace("\n", " ")
    result["tail_80"] = text[-80:].replace("\n", " ")

    # --- Flag 1: Suspiciously short ---
    if wc < 300:
        result["flags"].append("SHORT")
        result["status"] = "WARN"
        return result

    # --- Flag 2: Suspiciously long (possible bleed) ---
    if section_key == "item_1a" and wc > 25_000:
        result["flags"].append("VERY_LONG_1A")
    if section_key == "item_7" and wc > 40_000:
        result["flags"].append("VERY_LONG_7")

    # --- Flag 3: Contains financial statement headers (bleed into Item 8) ---
    # Only flag if we find headers like "CONSOLIDATED BALANCE SHEETS" as
    # standalone-ish lines, not just references like "see our financial statements".
    # Require 2+ different *types* of fin-stmt headers (a real bleed into Item 8
    # would have balance sheets + income statement + cash flows, not just one kind).
    _FS_TYPES = {
        "balance": re.compile(
            r"(?m)^\s*(?:CONSOLIDATED\s+)?BALANCE\s+SHEETS?\s*$", re.I),
        "income": re.compile(
            r"(?m)^\s*(?:CONSOLIDATED\s+)?STATEMENTS?\s+OF\s+(?:INCOME|OPERATIONS|COMPREHENSIVE)\s*$", re.I),
        "cashflow": re.compile(
            r"(?m)^\s*(?:CONSOLIDATED\s+)?STATEMENTS?\s+OF\s+CASH\s+FLOWS?\s*$", re.I),
        "notes": re.compile(
            r"(?m)^\s*NOTES?\s+TO\s+(?:THE\s+)?(?:CONSOLIDATED\s+)?FINANCIAL\s+STATEMENTS\s*$", re.I),
    }
    second_half = text[len(text) // 2:]
    fs_types_found = sum(1 for pat in _FS_TYPES.values() if pat.search(second_half))
    if fs_types_found >= 2:
        result["flags"].append("FINANCIAL_STMT_BLEED")

    # --- Flag 4: Contains numbered notes (bleed into notes) ---
    notes = _NOTES_HEADER_RE.findall(text)
    if len(notes) >= 3:
        result["flags"].append("NOTES_BLEED")

    # --- Flag 5: Contains many other Item headers (bleed across sections) ---
    # MD&A commonly references "Item 1A", "Item 7A", "Item 8", etc.
    # Only flag as bleed if we see 8+ distinct Item references (suggests
    # we've picked up another Part's TOC or multiple full sections).
    item_headers = _ITEM_HEADER_RE.findall(text)
    body_items = [h for h in item_headers
                  if text.index(h) > 200]
    if len(set(h.lower() for h in body_items)) >= 8:
        result["flags"].append("MULTI_ITEM_BLEED")

    # --- Flag 6: Head doesn't look right ---
    head_lower = text[:300].lower()
    if section_key == "item_1a":
        if not any(kw in head_lower for kw in ["risk", "item 1a", "item 1(a)"]):
            result["flags"].append("BAD_START_1A")
    elif section_key == "item_7":
        if not any(kw in head_lower for kw in
                   ["management", "discussion", "md&a", "item 7",
                    "overview", "results of operations"]):
            result["flags"].append("BAD_START_7")

    # --- Flag 7: High numeric density (table residue) ---
    numeric_chars = len(re.findall(r"[\d$%,.]", text))
    if len(text) > 0 and numeric_chars / len(text) > 0.15:
        result["flags"].append("HIGH_NUMERIC")

    # Overall status
    if result["flags"]:
        result["status"] = "WARN"
    else:
        result["status"] = "OK"

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    panel = pd.read_csv(DATA_DIR / "annual_panel.csv")
    cik_map = load_cik_map()

    # Filter to rows with accession numbers and CIK
    panel = panel.dropna(subset=["accession_number"])
    panel = panel[panel["ticker"].isin(cik_map)]

    random.seed(args.seed)
    sample = panel.sample(n=min(args.n, len(panel)), random_state=args.seed)
    print(f"=== Batch audit: {len(sample)} firm-years  (seed={args.seed}) ===\n")

    # Pre-fetch doc maps
    unique_ciks = {cik_map[t] for t in sample["ticker"].unique()}
    doc_maps: dict[str, dict] = {}
    print(f"Fetching doc maps for {len(unique_ciks)} firms...")
    for cik in sorted(unique_ciks):
        doc_maps[cik] = _build_primary_doc_map(cik)
        time.sleep(REQUEST_SLEEP)

    results = []
    ok = warn = fail = 0

    for i, (_, row) in enumerate(sample.iterrows()):
        ticker = row["ticker"]
        fy = int(row["fiscal_year"])
        accn = row["accession_number"]
        cik = cik_map[ticker]

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(extract_filing_text, cik, accn,
                                     doc_maps.get(cik))
                sections = future.result(timeout=FILING_TIMEOUT)
        except TimeoutError:
            print(f"  [{i+1:3d}] ✗ {ticker:6s} FY{fy}  TIMEOUT (>{FILING_TIMEOUT}s)")
            results.append({
                "ticker": ticker, "fiscal_year": fy,
                "item_1a_status": "TIMEOUT", "item_7_status": "TIMEOUT",
                "item_1a_wc": 0, "item_7_wc": 0,
                "item_1a_flags": "TIMEOUT", "item_7_flags": "",
            })
            fail += 1
            continue
        except Exception as e:
            print(f"  [{i+1:3d}] {ticker} FY{fy}: ERROR — {e}")
            results.append({
                "ticker": ticker, "fiscal_year": fy,
                "item_1a_status": "ERROR", "item_7_status": "ERROR",
                "item_1a_wc": 0, "item_7_wc": 0,
                "item_1a_flags": "ERROR", "item_7_flags": str(e)[:100],
            })
            fail += 1
            continue
        time.sleep(REQUEST_SLEEP)

        a1 = audit_section(sections["item_1a"], "item_1a")
        a7 = audit_section(sections["item_7"], "item_7")

        rec = {
            "ticker": ticker,
            "fiscal_year": fy,
            "item_1a_status": a1["status"],
            "item_1a_wc": a1["word_count"],
            "item_1a_flags": "|".join(a1["flags"]) if a1["flags"] else "",
            "item_7_status": a7["status"],
            "item_7_wc": a7["word_count"],
            "item_7_flags": "|".join(a7["flags"]) if a7["flags"] else "",
        }
        results.append(rec)

        # Determine combined status for this filing
        statuses = {a1["status"], a7["status"]}
        if "MISSING" in statuses or "ERROR" in statuses:
            fail += 1
            marker = "✗"
        elif "WARN" in statuses:
            warn += 1
            marker = "⚠"
        else:
            ok += 1
            marker = "✓"

        flags_str = ""
        if a1["flags"]:
            flags_str += f" 1A:[{','.join(a1['flags'])}]"
        if a7["flags"]:
            flags_str += f" 7:[{','.join(a7['flags'])}]"

        print(f"  [{i+1:3d}] {marker} {ticker:6s} FY{fy}  "
              f"1A={a1['word_count']:>6,}w  7={a7['word_count']:>6,}w"
              f"{flags_str}")

    # Summary
    total = len(results)
    missing_1a = sum(1 for r in results if r["item_1a_status"] == "MISSING")
    missing_7 = sum(1 for r in results if r["item_7_status"] == "MISSING")

    print(f"\n{'='*60}")
    print(f"TOTAL: {total}  |  OK: {ok}  |  WARN: {warn}  |  FAIL/MISSING: {fail}")
    print(f"Missing Item 1A: {missing_1a}  |  Missing Item 7: {missing_7}")

    # Flag breakdown
    all_flags: dict[str, int] = {}
    for r in results:
        for f in (r["item_1a_flags"] + "|" + r["item_7_flags"]).split("|"):
            if f:
                all_flags[f] = all_flags.get(f, 0) + 1
    if all_flags:
        print("\nFlag breakdown:")
        for flag, count in sorted(all_flags.items(), key=lambda x: -x[1]):
            print(f"  {flag:30s} {count}")

    # Print details for WARN/FAIL cases
    problems = [r for r in results
                if r["item_1a_status"] in ("WARN", "MISSING", "ERROR")
                or r["item_7_status"] in ("WARN", "MISSING", "ERROR")]
    if problems:
        print(f"\n{'='*60}")
        print(f"PROBLEM DETAILS ({len(problems)} filings):\n")
        for r in problems:
            print(f"  {r['ticker']} FY{r['fiscal_year']}:")
            print(f"    1A: {r['item_1a_status']:7s} {r['item_1a_wc']:>6,}w  flags={r['item_1a_flags']}")
            print(f"     7: {r['item_7_status']:7s} {r['item_7_wc']:>6,}w  flags={r['item_7_flags']}")

    # Save to JSONL for deeper analysis
    out_path = DATA_DIR / "batch_audit_results.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
