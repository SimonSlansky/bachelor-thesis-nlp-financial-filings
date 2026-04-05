"""
SEC 10-K section extraction — v3 (final robust version).

Combines the two best ideas from v1 and v2:
  1.  Inline-tag unwrapping  (fixes 'RIS K FACTORS' → 'RISK FACTORS')
  2.  Subtitle-gated longest-gap heuristic:
        a) Find ALL matches of broad pattern 'Item X'
        b) FILTER: keep only those with a subtitle keyword nearby
           (e.g. "risk" for Item 1A, "management|discussion" for Item 7)
        c) Among filtered matches, pick the one with the LONGEST gap
           to the next different-Item header

This correctly handles:
  - TOC entries  ("Item 1A .........20"   → no subtitle keyword → filtered out)
  - Cross-references embedded in text     → subtitle may be present but gap is short
  - Page footers  (MSFT: "Item 1A")       → no subtitle keyword → filtered out
  - HTML word splits  ("RIS K")           → keyword search is lenient
  - Forward-looking disclaimers ("Item 1A of this Form 10-K") → no subtitle → out
  - Cross-ref stubs  (XOM Item 7 = 38 words) → detected by word-count gate
"""

import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from config import SEC_HEADERS, REQUEST_SLEEP

# ── Test sample ───────────────────────────────────────────────────────────
SAMPLES = [
    ("AAPL", "0000320193", "0000320193-24-000123", 2024),
    ("MSFT", "0000789019", "0000950170-24-087843", 2024),
    ("JNJ",  "0000200406", "0000200406-25-000038", 2024),
    ("WMT",  "0000104169", "0000104169-24-000056", 2024),
    ("XOM",  "0000034088", "0000034088-25-000010", 2024),
    ("AAPL", "0000320193", "0000320193-22-000108", 2022),
    ("AAPL", "0000320193", "0000320193-20-000096", 2020),
    ("A",    "0001090872", "0001090872-16-000082", 2016),
    ("A",    "0001090872", "0001090872-24-000049", 2024),
]

INLINE_TAGS = {"span", "a", "b", "i", "em", "strong", "font", "sup", "sub",
               "u", "s", "small", "big", "mark", "abbr", "cite", "code"}

# ── Broad item-number patterns (no subtitle required) ────────────────────
_ITEM_RE = {
    "1":   re.compile(r"\bitem\s+1\b\.?(?!\s*[0-9a-z])", re.IGNORECASE),
    "1a":  re.compile(r"\bitem\s+1\s*a\b\.?",  re.IGNORECASE),
    "1b":  re.compile(r"\bitem\s+1\s*b\b\.?",  re.IGNORECASE),
    "1c":  re.compile(r"\bitem\s+1\s*c\b\.?",  re.IGNORECASE),
    "2":   re.compile(r"\bitem\s+2\b\.?",       re.IGNORECASE),
    "3":   re.compile(r"\bitem\s+3\b\.?",       re.IGNORECASE),
    "4":   re.compile(r"\bitem\s+4\b\.?",       re.IGNORECASE),
    "5":   re.compile(r"\bitem\s+5\b\.?",       re.IGNORECASE),
    "6":   re.compile(r"\bitem\s+6\b\.?",       re.IGNORECASE),
    "7":   re.compile(r"\bitem\s+7\b\.?(?!\s*a)", re.IGNORECASE),
    "7a":  re.compile(r"\bitem\s+7\s*a\b\.?",  re.IGNORECASE),
    "8":   re.compile(r"\bitem\s+8\b\.?",       re.IGNORECASE),
    "9":   re.compile(r"\bitem\s+9\b\.?(?!\s*[a-c])", re.IGNORECASE),
    "9a":  re.compile(r"\bitem\s+9\s*a\b\.?",  re.IGNORECASE),
    "9b":  re.compile(r"\bitem\s+9\s*b\b\.?",  re.IGNORECASE),
    "10":  re.compile(r"\bitem\s+10\b\.?",      re.IGNORECASE),
    "11":  re.compile(r"\bitem\s+11\b\.?",      re.IGNORECASE),
    "12":  re.compile(r"\bitem\s+12\b\.?",      re.IGNORECASE),
    "13":  re.compile(r"\bitem\s+13\b\.?",      re.IGNORECASE),
    "14":  re.compile(r"\bitem\s+14\b\.?",      re.IGNORECASE),
    "15":  re.compile(r"\bitem\s+15\b\.?",      re.IGNORECASE),
}

# ── Subtitle keywords expected near the REAL section header ──────────────
# Searched within 150 chars AFTER the "Item X" match.
_SUBTITLE_KEYWORDS = {
    "1a": ["risk", "ris k"],                              # "Risk Factors" (incl. HTML split)
    "7":  ["management", "discussion", "md&a", "md a"],   # "Management's Discussion…"
}

# ── End-item boundaries for extraction ───────────────────────────────────
_END_ITEMS = {
    "1a": ["1b", "1c", "2"],
    "7":  ["7a", "8"],
}


# ═══════════════════════════════════════════════════════════════════════════
# Document handling
# ═══════════════════════════════════════════════════════════════════════════

def get_primary_doc(cik: str, accn: str) -> str | None:
    accn_nd = accn.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn_nd}/index.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=15)
    if r.status_code != 200:
        return None
    items = r.json().get("directory", {}).get("item", [])
    htm = [(it["name"], int(it.get("size") or 0))
           for it in items if it["name"].endswith(".htm")]
    return max(htm, key=lambda x: x[1])[0] if htm else None


def download_filing_html(cik: str, accn: str, doc_name: str) -> str:
    accn_nd = accn.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn_nd}/{doc_name}"
    r = requests.get(url, headers=SEC_HEADERS, timeout=60)
    r.raise_for_status()
    return r.text


def html_to_text(html: str) -> str:
    """Convert 10-K HTML to plain text with inline-tag unwrapping."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all("ix:header"):
        tag.decompose()
    # Unwrap inline tags so adjacent text merges (fixes "RIS K" → "RISK")
    for tag in soup.find_all(list(INLINE_TAGS)):
        tag.unwrap()
    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Section extraction: subtitle-gated longest-gap
# ═══════════════════════════════════════════════════════════════════════════

def _all_item_positions(text: str) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {}
    for item_id, pat in _ITEM_RE.items():
        positions[item_id] = sorted(m.start() for m in pat.finditer(text))
    return positions


def _is_section_header(text: str, match_start: int, match_end: int,
                       keywords: list[str], target_item: str = "") -> bool:
    """Return True if match is an actual section header, not a cross-reference.

    Section headers look like:  Item 1A.   Risk Factors   (The Company's…)
    Cross-references look like: Item 1A of this Form 10-K under the heading…

    Distinguishing rule: the text immediately after "Item X" (within 40 chars,
    ignoring dots, spaces, nbsps) must contain a subtitle keyword AND must NOT
    contain a cross-reference preposition.
    """
    # Text right after the match (e.g. after "Item 1A")
    # Normalize newlines → spaces (HTML stripping can split words across lines)
    after = text[match_end:match_end + 60].lower().replace("\n", " ")

    # Cross-reference giveaways in immediate vicinity (first 30 chars)
    XREF_PHRASES = ["of this", "of our", "of the", "of its", "under the",
                    "and elsewhere", "to be a", "for a ", "for additional",
                    "in this", " in part"]
    near = after[:30]
    if any(phrase in near for phrase in XREF_PHRASES):
        return False

    # Comma or closing paren right after "Item X" = cross-reference syntax
    first_char = after.lstrip(" .").lstrip()[:1]
    if first_char in (",", ")"):
        return False

    # Must have subtitle keyword within 40 chars
    title_zone = after[:40]
    if not any(kw in title_zone for kw in keywords):
        return False

    # TOC detection: if another "Item X" header appears within 200 chars
    # after this match, we're in a table of contents (items listed sequentially).
    # Real section headers are followed by body text, not the next item.
    toc_zone = text[match_end:match_end + 200].replace("\n", " ")
    other_item_count = 0
    for item_id, pat in _ITEM_RE.items():
        if item_id != target_item:
            if pat.search(toc_zone):
                other_item_count += 1
    if other_item_count >= 2:  # 2+ other items nearby → TOC
        return False

    return True


def _is_crossref(text: str, match_start: int) -> bool:
    """Return True if an Item match is a cross-reference (not a section header).

    Cross-references are preceded/followed by phrases like:
      "in Part II, Item 8 of this Form", "see Item 7A", "included in Item 8"
    Section headers stand alone (preceded by newline / blank).
    """
    # Check text AFTER the match (within 25 chars)
    m = None
    for pat in _ITEM_RE.values():
        m = pat.match(text, match_start)
        if m:
            break
    if m:
        after = text[m.end():m.end() + 30].lower().replace("\n", " ").strip()
        # If followed by preposition → cross-ref
        if any(after.startswith(w) for w in
               ["of this", "of our", "of the", "of its", "under",
                "in this", "in our", "in the", "and ", "to be",
                "for a", "for additional", ",", ")"]):
            return True

    # Check text BEFORE the match (within 25 chars)
    before = text[max(0, match_start - 25):match_start].lower().strip()
    if any(before.endswith(w) for w in
           ["see", "in", "in part ii,", "in part i,", "part ii,",
            "part i,", "included in", "with"]):
        return True

    return False


def _longest_gap_start(text: str, candidates: list[int],
                       positions: dict[str, list[int]],
                       target_item: str) -> int | None:
    """Among candidates, return the one with the longest gap to next different item."""
    if not candidates:
        return None

    other = sorted(
        p for item_id, ps in positions.items()
        if item_id != target_item
        for p in ps
    )

    best, best_gap = None, -1
    for cand in candidates:
        gap = len(text) - cand
        for op in other:
            if op > cand + 50:
                gap = op - cand
                break
        if gap > best_gap:
            best_gap = gap
            best = cand
    return best


def extract_section(text: str, target_item: str,
                    positions: dict[str, list[int]]) -> tuple[str | None, dict]:
    """Extract target section with subtitle-gated longest-gap heuristic."""
    end_items = _END_ITEMS[target_item]
    all_cands = positions.get(target_item, [])
    keywords = _SUBTITLE_KEYWORDS.get(target_item, [])
    meta = {"target": target_item, "n_candidates": len(all_cands)}

    if not all_cands:
        meta["status"] = "no_matches"
        return None, meta

    # Stage 1: filter to candidates that are actual section headers (not cross-refs)
    if keywords:
        titled = []
        for p in all_cands:
            m = _ITEM_RE[target_item].search(text, p)
            if m and _is_section_header(text, m.start(), m.end(), keywords, target_item):
                titled.append(p)
    else:
        titled = all_cands

    meta["n_titled"] = len(titled)

    # Stage 2: among titled candidates, try extracting each and pick the
    # one that yields the LONGEST section. This beats plain longest-gap
    # because it ignores candidates past all end boundaries.
    def _section_length(candidate: int) -> int:
        """Compute section word count for a candidate start position."""
        end = len(text)
        for end_item in end_items:
            for pos in positions.get(end_item, []):
                if pos > candidate + 100 and not _is_crossref(text, pos):
                    end = min(end, pos)
                    break
        section = text[candidate:end].strip()
        wc = len(section.split())
        # Penalize sections that run to end-of-document (no valid end found)
        if end == len(text):
            wc = min(wc, 200)  # cap at stub-level
        return wc

    if titled:
        start = max(titled, key=_section_length)
        meta["selection"] = "subtitle+maxlen"
    else:
        # Fallback: no subtitle matches (should be rare)
        start = _longest_gap_start(text, all_cands,
                                   positions, target_item)
        meta["selection"] = "gap_only"

    if start is None:
        meta["status"] = "no_matches"
        return None, meta

    meta["start_pos"] = start
    meta["start_pct"] = round(100 * start / len(text), 1)

    # Find end boundary — skip cross-references
    end = len(text)
    for end_item in end_items:
        for pos in positions.get(end_item, []):
            if pos > start + 100 and not _is_crossref(text, pos):
                end = min(end, pos)
                break

    section = text[start:end].strip()
    wc = len(section.split())
    meta["word_count"] = wc
    meta["end_pos"] = end

    # Detect stubs / cross-references
    if wc < 200:
        lower = section.lower()
        if any(p in lower for p in
               ["reference is made", "refer to", "incorporated by reference",
                "included elsewhere", "see the section"]):
            meta["status"] = "cross_reference"
        else:
            meta["status"] = "stub"
        return section, meta

    meta["status"] = "ok"
    return section, meta


# ═══════════════════════════════════════════════════════════════════════════
# Test runner
# ═══════════════════════════════════════════════════════════════════════════

def run_all_tests():
    print("SEC 10-K Section Extraction — v3 (subtitle-gated longest-gap)")
    print("=" * 80)
    print()

    all_results = []

    for ticker, cik, accn, fy in SAMPLES:
        label = f"{ticker} FY{fy}"
        doc = get_primary_doc(cik, accn)
        time.sleep(REQUEST_SLEEP)
        if not doc:
            print(f"  {label}: SKIP (no primary doc)")
            continue

        html = download_filing_html(cik, accn, doc)
        time.sleep(REQUEST_SLEEP)
        text = html_to_text(html)
        positions = _all_item_positions(text)

        print(f"  {label} ({len(text):,} chars)")

        for target, sec_name in [("1a", "Item 1A"), ("7", "Item 7")]:
            section, meta = extract_section(text, target, positions)
            status = meta["status"]
            wc = meta.get("word_count", 0)
            sel = meta.get("selection", "?")
            n_cand = meta.get("n_candidates", 0)
            n_titled = meta.get("n_titled", 0)

            if status == "ok":
                preview = section[:180].replace("\n", " ").strip()
                pct = meta.get("start_pct", "?")
                print(f"    {sec_name}: ✓ {wc:>6,} words @ {pct}%  "
                      f"[{n_titled}/{n_cand} titled, {sel}]")
                print(f"      → {preview[:120]}")

                # Quality checks
                lower = section.lower()
                if sec_name == "Item 1A":
                    chk = ["risk", "could", "may", "adverse"]
                else:
                    chk = ["revenue", "operating", "financial", "results"]
                found = sum(1 for k in chk if k in lower)
                if found < 2:
                    print(f"      ⚠ Only {found}/{len(chk)} expected keywords found")

                # Over-extraction
                leaks = ["consolidated balance sheet",
                         "notes to consolidated financial statements"]
                for lk in leaks:
                    if lk in lower:
                        print(f"      ⚠ Over-extraction: '{lk}' found in extracted text")
                        break

            elif status in ("cross_reference", "stub"):
                sym = "⤷" if status == "cross_reference" else "⚠"
                print(f"    {sec_name}: {sym} {status.upper()} ({wc} words)")
                if section:
                    print(f"      → {section[:150].replace(chr(10), ' ')}")
            else:
                print(f"    {sec_name}: ✗ NO MATCHES")

            all_results.append({
                "ticker": ticker, "fy": fy, "section": sec_name,
                "status": status, "words": wc, "selection": sel,
            })

        print()

    # ── Summary ────────────────────────────────────────────────────────────
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for sec_name in ["Item 1A", "Item 7"]:
        rows = [r for r in all_results if r["section"] == sec_name]
        ok = sum(1 for r in rows if r["status"] == "ok")
        stub = sum(1 for r in rows if r["status"] in ("stub", "cross_reference"))
        fail = sum(1 for r in rows if r["status"] == "no_matches")
        print(f"\n  {sec_name}: {ok}/{len(rows)} OK, {stub} stubs, {fail} failures")
        for r in rows:
            sym = {"ok": "✓", "stub": "⚠", "cross_reference": "⤷",
                   "no_matches": "✗"}.get(r["status"], "?")
            print(f"    {sym} {r['ticker']:>4s} FY{r['fy']}: {r['words']:>6,} words "
                  f"[{r['status']}] ({r['selection']})")

    # Comparison with expected ranges
    print("\n  Expected word-count ranges (academic literature):")
    print("    Item 1A (Risk Factors): 3,000–25,000 words")
    print("    Item 7  (MD&A):         2,000–20,000 words")

    print()
    print("=" * 80)
    print("ASSESSMENT")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
