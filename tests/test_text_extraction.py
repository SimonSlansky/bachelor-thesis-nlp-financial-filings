"""
Investigate approaches to extracting Item 1A (Risk Factors) and Item 7 (MD&A)
from SEC 10-K HTML filings.

Tests three approaches in order of minimalism:
  1. Submissions API section-document lookup  (zero parsing needed?)
  2. Plain-text regex with full section titles  (strip HTML → regex)
  3. HTML bold-element detection               (BeautifulSoup structural)

Tested across multiple companies × multiple years to assess reliability.
"""
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from config import SEC_HEADERS, REQUEST_SLEEP

# ── Test sample ───────────────────────────────────────────────────────────
# Diverse companies: different filers, industries, filing software, years
# (ticker, cik, accession_number, fiscal_year)
SAMPLES = [
    # Tech giants (Workiva/Inline XBRL, modern filings)
    ("AAPL", "0000320193", "0000320193-24-000123", 2024),
    ("MSFT", "0000789019", "0000950170-24-087843", 2024),
    # Healthcare / consumer staples
    ("JNJ",  "0000200406", "0000200406-25-000038", 2024),
    ("WMT",  "0000104169", "0000104169-24-000056", 2024),
    # Energy
    ("XOM",  "0000034088", "0000034088-25-000010", 2024),
    # Older filings (test across XBRL eras)
    ("AAPL", "0000320193", "0000320193-22-000108", 2022),
    ("AAPL", "0000320193", "0000320193-20-000096", 2020),
    # Agilent (company "A") — diverse filer across years
    ("A",    "0001090872", "0001090872-16-000082", 2016),
    ("A",    "0001090872", "0001090872-24-000049", 2024),
]


def _accn_nodash(accn: str) -> str:
    return accn.replace("-", "")


def _base_url(cik: str, accn: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{_accn_nodash(accn)}"


# ═══════════════════════════════════════════════════════════════════════════
# Approach 0: Can the Submissions API give us the primary document directly?
# ═══════════════════════════════════════════════════════════════════════════

def test_submissions_primary_doc():
    """Check if the submissions API provides primaryDocument for our accessions."""
    print("=" * 80)
    print("APPROACH 0: Submissions API → primaryDocument lookup")
    print("=" * 80)

    # Cache submissions per CIK to avoid redundant fetches
    cache: dict[str, dict] = {}
    results = []

    for ticker, cik, accn, fy in SAMPLES:
        if cik not in cache:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            r = requests.get(url, headers=SEC_HEADERS, timeout=15)
            cache[cik] = r.json() if r.status_code == 200 else {}
            time.sleep(REQUEST_SLEEP)

        sub = cache[cik]
        recent = sub.get("filings", {}).get("recent", {})
        accns = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        found = False
        for i, a in enumerate(accns):
            if a == accn:
                results.append((ticker, fy, primary_docs[i]))
                print(f"  {ticker} FY{fy}: primaryDocument = {primary_docs[i]}")
                found = True
                break

        if not found:
            # Might be in older filings (paginated)
            results.append((ticker, fy, None))
            print(f"  {ticker} FY{fy}: NOT in recent filings (may need pagination)")

    print(f"\n  Found primary doc for {sum(1 for _,_,d in results if d):}/{len(results)} samples")
    print("  ⚠ Submissions API only holds ~1000 recent filings; older ones need filing pages")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Approach 1: Filing index JSON → identify primary document by size/name
# ═══════════════════════════════════════════════════════════════════════════

def get_primary_doc_from_index(cik: str, accn: str) -> str | None:
    """Get the primary 10-K HTML filename from the filing's index.json."""
    url = f"{_base_url(cik, accn)}/index.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=15)
    if r.status_code != 200:
        return None
    items = r.json().get("directory", {}).get("item", [])
    # Strategy: the primary 10-K document is the largest .htm file
    # (exhibits are smaller; XBRL files are .xsd/.xml)
    htm_files = [(it["name"], int(it.get("size") or 0))
                 for it in items if it["name"].endswith(".htm")]
    if not htm_files:
        return None
    return max(htm_files, key=lambda x: x[1])[0]


def test_index_approach():
    """Check reliability of finding primary doc via filing index JSON."""
    print("\n" + "=" * 80)
    print("APPROACH 1: Filing index.json → largest .htm = primary 10-K doc")
    print("=" * 80)

    results = []
    for ticker, cik, accn, fy in SAMPLES:
        doc = get_primary_doc_from_index(cik, accn)
        url = f"{_base_url(cik, accn)}/{doc}" if doc else "N/A"
        print(f"  {ticker} FY{fy}: {doc or 'FAILED'}")
        results.append((ticker, fy, doc))
        time.sleep(REQUEST_SLEEP)

    success = sum(1 for _, _, d in results if d)
    print(f"\n  Found primary doc for {success}/{len(results)} samples")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Approach 2: Plain-text regex extraction (strip HTML → find sections)
# ═══════════════════════════════════════════════════════════════════════════

# Section header patterns: require the FULL section title after the item number
# to distinguish from TOC entries (which just have "Item 1A.  22") and
# cross-references ("see Item 1A of this Form").
SECTION_PATTERNS = {
    "item_1a": re.compile(
        r"item\s+1a\.?\s{0,20}(?:&#160;|\xa0|\s)*risk\s+factors",
        re.IGNORECASE,
    ),
    "item_7": re.compile(
        r"item\s+7\.?\s{0,20}(?:&#160;|\xa0|\s)*management.{0,5}s?\s+discussion",
        re.IGNORECASE,
    ),
    "item_7a": re.compile(
        r"item\s+7a\.?\s{0,20}(?:&#160;|\xa0|\s)*quantitative",
        re.IGNORECASE,
    ),
    "item_8": re.compile(
        r"item\s+8\.?\s{0,20}(?:&#160;|\xa0|\s)*financial\s+statements",
        re.IGNORECASE,
    ),
    "item_1b": re.compile(
        r"item\s+1b\.?\s{0,20}(?:&#160;|\xa0|\s)*unresolved\s+staff",
        re.IGNORECASE,
    ),
    "item_2": re.compile(
        r"item\s+2\.?\s{0,20}(?:&#160;|\xa0|\s)*properties",
        re.IGNORECASE,
    ),
}


def strip_html_to_text(html: str) -> str:
    """Convert HTML to plain text, preserving meaningful whitespace."""
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style elements
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    # Normalize whitespace but keep newlines for structure
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    return text


def find_section_boundaries_plaintext(text: str) -> dict[str, list[int]]:
    """Find all matches for each section header in plain text.

    Returns {section_name: [list of char positions]}.
    """
    boundaries = {}
    for name, pat in SECTION_PATTERNS.items():
        matches = [m.start() for m in pat.finditer(text)]
        boundaries[name] = matches
    return boundaries


def extract_section_plaintext(text: str, start_section: str, end_sections: list[str],
                              boundaries: dict) -> str | None:
    """Extract text between start_section (last match) and first subsequent end_section.

    Uses LAST match of start_section to skip TOC entries.
    """
    starts = boundaries.get(start_section, [])
    if not starts:
        return None
    start_pos = starts[-1]  # last match = actual section header (after TOC)

    # Find the earliest end boundary that comes AFTER start_pos
    end_pos = len(text)
    for end_sec in end_sections:
        ends = boundaries.get(end_sec, [])
        for e in ends:
            if e > start_pos + 100:  # must be well past the header itself
                end_pos = min(end_pos, e)
                break

    section = text[start_pos:end_pos].strip()
    return section if len(section) > 200 else None  # reject trivially short


def test_plaintext_regex():
    """Test plain-text regex extraction across all samples."""
    print("\n" + "=" * 80)
    print("APPROACH 2: Strip HTML → plain-text regex (LAST match = real header)")
    print("=" * 80)

    results = []
    for ticker, cik, accn, fy in SAMPLES:
        doc = get_primary_doc_from_index(cik, accn)
        time.sleep(REQUEST_SLEEP)
        if not doc:
            print(f"  {ticker} FY{fy}: SKIP (no primary doc)")
            results.append((ticker, fy, None, None))
            continue

        url = f"{_base_url(cik, accn)}/{doc}"
        r = requests.get(url, headers=SEC_HEADERS, timeout=60)
        time.sleep(REQUEST_SLEEP)
        if r.status_code != 200:
            print(f"  {ticker} FY{fy}: SKIP (HTTP {r.status_code})")
            results.append((ticker, fy, None, None))
            continue

        text = strip_html_to_text(r.text)
        bounds = find_section_boundaries_plaintext(text)

        # Extract Item 1A (ends at Item 1B or Item 2)
        item1a = extract_section_plaintext(text, "item_1a", ["item_1b", "item_2"], bounds)
        # Extract Item 7 (ends at Item 7A or Item 8)
        item7 = extract_section_plaintext(text, "item_7", ["item_7a", "item_8"], bounds)

        i1a_words = len(item1a.split()) if item1a else 0
        i7_words = len(item7.split()) if item7 else 0

        status1a = f"{i1a_words:>6,} words" if item1a else "FAILED"
        status7 = f"{i7_words:>6,} words" if item7 else "FAILED"

        print(f"  {ticker} FY{fy}: Item 1A = {status1a} | Item 7 = {status7}")

        # Sanity check: show first 100 chars of extracted text
        if item1a:
            clean = item1a[:150].replace("\n", " ").strip()
            print(f"         1A preview: {clean}")
        if item7:
            clean = item7[:150].replace("\n", " ").strip()
            print(f"         7  preview: {clean}")

        results.append((ticker, fy, i1a_words, i7_words))

        # Count section header matches for debugging
        for sec_name in ["item_1a", "item_7", "item_7a", "item_8"]:
            n = len(bounds.get(sec_name, []))
            if n != 1:
                print(f"         ⚠ {sec_name}: {n} matches (expected 1–2, using last)")

    # Summary
    ok_1a = sum(1 for _, _, a, _ in results if a and a > 500)
    ok_7 = sum(1 for _, _, _, b in results if b and b > 500)
    print(f"\n  Item 1A extracted: {ok_1a}/{len(results)}")
    print(f"  Item 7  extracted: {ok_7}/{len(results)}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Approach 3: HTML bold-element detection (structural parsing)
# ═══════════════════════════════════════════════════════════════════════════

def find_section_by_bold_html(html: str, pattern: re.Pattern) -> list[int]:
    """Find positions where pattern appears inside a bold/heavy-weight element.

    This filters out TOC links and cross-references, which are NOT bold.
    """
    matches = []
    # Search raw HTML for pattern; then check if surrounding style has font-weight:700/bold
    for m in pattern.finditer(html.lower()):
        # Look backwards ~500 chars for style context
        context = html[max(0, m.start() - 500):m.start()].lower()
        # Check for bold indicators
        is_bold = (
            "font-weight:700" in context
            or "font-weight:bold" in context
            or "<b>" in context[-50:]
            or "<strong>" in context[-100:]
        )
        if is_bold:
            matches.append(m.start())
    return matches


def test_bold_html_detection():
    """Test bold-element detection across all samples."""
    print("\n" + "=" * 80)
    print("APPROACH 3: HTML bold-element detection (font-weight:700 / <b>)")
    print("=" * 80)

    results = []
    for ticker, cik, accn, fy in SAMPLES:
        doc = get_primary_doc_from_index(cik, accn)
        time.sleep(REQUEST_SLEEP)
        if not doc:
            print(f"  {ticker} FY{fy}: SKIP (no primary doc)")
            results.append((ticker, fy, 0, 0))
            continue

        url = f"{_base_url(cik, accn)}/{doc}"
        r = requests.get(url, headers=SEC_HEADERS, timeout=60)
        time.sleep(REQUEST_SLEEP)
        if r.status_code != 200:
            print(f"  {ticker} FY{fy}: SKIP (HTTP {r.status_code})")
            results.append((ticker, fy, 0, 0))
            continue

        html = r.text
        bold_1a = find_section_by_bold_html(html, SECTION_PATTERNS["item_1a"])
        bold_7 = find_section_by_bold_html(html, SECTION_PATTERNS["item_7"])
        bold_7a = find_section_by_bold_html(html, SECTION_PATTERNS["item_7a"])
        bold_8 = find_section_by_bold_html(html, SECTION_PATTERNS["item_8"])

        print(f"  {ticker} FY{fy}: bold Item 1A={len(bold_1a)}, "
              f"Item 7={len(bold_7)}, Item 7A={len(bold_7a)}, Item 8={len(bold_8)}")

        results.append((ticker, fy, len(bold_1a), len(bold_7)))

    ok_1a = sum(1 for _, _, a, _ in results if a == 1)
    ok_7 = sum(1 for _, _, _, b in results if b == 1)
    print(f"\n  Exactly 1 bold Item 1A: {ok_1a}/{len(results)}")
    print(f"  Exactly 1 bold Item 7 : {ok_7}/{len(results)}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Cross-validation: compare approaches and check text quality
# ═══════════════════════════════════════════════════════════════════════════

def test_text_quality_checks():
    """Verify extracted text makes logical sense."""
    print("\n" + "=" * 80)
    print("QUALITY CHECKS: verify extracted sections are genuine")
    print("=" * 80)

    # Use one well-known filing for deep inspection
    ticker, cik, accn, fy = "AAPL", "0000320193", "0000320193-24-000123", 2024
    doc = get_primary_doc_from_index(cik, accn)
    time.sleep(REQUEST_SLEEP)
    url = f"{_base_url(cik, accn)}/{doc}"
    r = requests.get(url, headers=SEC_HEADERS, timeout=60)
    text = strip_html_to_text(r.text)
    bounds = find_section_boundaries_plaintext(text)

    item1a = extract_section_plaintext(text, "item_1a", ["item_1b", "item_2"], bounds)
    item7 = extract_section_plaintext(text, "item_7", ["item_7a", "item_8"], bounds)

    print(f"  Full document: {len(text):,} chars, {len(text.split()):,} words")
    print()

    for name, section in [("Item 1A", item1a), ("Item 7", item7)]:
        if not section:
            print(f"  {name}: EXTRACTION FAILED")
            continue
        words = section.split()
        # Check 1: reasonable length
        print(f"  {name}: {len(words):,} words, {len(section):,} chars")

        # Check 2: contains expected keywords
        lower = section.lower()
        if name == "Item 1A":
            keywords = ["risk", "could", "may", "adverse", "material"]
        else:
            keywords = ["revenue", "operating", "financial", "results", "management"]
        found = [kw for kw in keywords if kw in lower]
        print(f"    Expected keywords present: {found}")
        missing = [kw for kw in keywords if kw not in lower]
        if missing:
            print(f"    ⚠ Missing keywords: {missing}")

        # Check 3: does NOT contain financial statement tables (sign of over-extraction)
        table_markers = ["consolidated balance sheet", "consolidated statements of operations",
                         "notes to consolidated financial statements"]
        leaks = [m for m in table_markers if m in lower]
        if leaks:
            print(f"    ⚠ POSSIBLE OVER-EXTRACTION: contains '{leaks[0]}'")
        else:
            print(f"    ✓ No financial statement table leakage detected")

        # Check 4: starts with the section title
        first_line = section[:200].replace("\n", " ").strip()
        if "item" in first_line.lower()[:20]:
            print(f"    ✓ Starts with section header")
        else:
            print(f"    ⚠ Does not start with 'Item': {first_line[:80]}")

        # Check 5: last 100 chars — should NOT end mid-sentence
        tail = section[-200:].replace("\n", " ").strip()
        print(f"    Last 100 chars: ...{tail[-100:]}")
        print()


# ═══════════════════════════════════════════════════════════════════════════
# Scalability check: how fast is download + extraction?
# ═══════════════════════════════════════════════════════════════════════════

def test_timing():
    """Time the full pipeline for one filing."""
    print("\n" + "=" * 80)
    print("TIMING: full pipeline for one filing")
    print("=" * 80)

    import time as t
    ticker, cik, accn, fy = "AAPL", "0000320193", "0000320193-24-000123", 2024

    t0 = t.perf_counter()
    doc = get_primary_doc_from_index(cik, accn)
    t1 = t.perf_counter()
    print(f"  Index lookup:  {t1-t0:.2f}s")

    time.sleep(REQUEST_SLEEP)
    url = f"{_base_url(cik, accn)}/{doc}"
    r = requests.get(url, headers=SEC_HEADERS, timeout=60)
    t2 = t.perf_counter()
    print(f"  HTML download: {t2-t1:.2f}s  ({len(r.text)/1024:.0f} KB)")

    text = strip_html_to_text(r.text)
    t3 = t.perf_counter()
    print(f"  HTML→text:     {t3-t2:.2f}s")

    bounds = find_section_boundaries_plaintext(text)
    item1a = extract_section_plaintext(text, "item_1a", ["item_1b", "item_2"], bounds)
    item7 = extract_section_plaintext(text, "item_7", ["item_7a", "item_8"], bounds)
    t4 = t.perf_counter()
    print(f"  Section regex: {t4-t3:.2f}s")
    print(f"  TOTAL:         {t4-t0:.2f}s per filing")
    print(f"  Estimated for ~6,700 filings: {(t4-t0)*6700/3600:.1f} hours "
          f"(+ {0.15*6700/3600:.1f}h rate-limit sleep)")


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("SEC 10-K Section Extraction — Approach Investigation")
    print("=" * 80)
    print()

    # Run all tests
    test_submissions_primary_doc()
    idx_results = test_index_approach()
    pt_results = test_plaintext_regex()
    bold_results = test_bold_html_detection()
    test_text_quality_checks()
    test_timing()

    # Final recommendation
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATION")
    print("=" * 80)
    print("""
  Approach 0 (Submissions API): gives primaryDocument but limited to recent filings.
  Approach 1 (Index JSON → largest .htm): reliable doc identification, works for all years.
  Approach 2 (Plain-text regex, last match): simplest extraction, check results above.
  Approach 3 (Bold HTML detection): more precise but harder to generalize.

  Recommended pipeline:
    1. Get primary doc name from index.json (Approach 1)
    2. Download HTML, strip to text (BeautifulSoup)
    3. Find section boundaries via full-title regex (Approach 2)
    4. Use LAST match to skip TOC/cross-references
    5. Validate: word count > 500, expected keywords present
""")
