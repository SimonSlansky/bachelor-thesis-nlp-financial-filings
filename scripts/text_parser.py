"""SEC 10-K section text extraction (Item 1A Risk Factors, Item 7 MD&A).

Downloads 10-K HTML from EDGAR, strips to plain text, and extracts sections
using a subtitle-gated max-length heuristic.  Designed to run on the full
panel (~6 700 filings).
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, NavigableString

from config import SEC_HEADERS, REQUEST_SLEEP, DATA_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_SECTION_WORDS = 200          # below this → stub / cross-reference

_INLINE_TAGS = frozenset(
    ["span", "a", "b", "i", "em", "strong", "font", "sup", "sub",
     "u", "s", "small", "big", "mark", "abbr", "cite", "code"]
)

# Broad item-number patterns (no subtitle required)
_ITEM_RE: dict[str, re.Pattern] = {
    "1a":  re.compile(r"\bitem\s+1[\s.]*a\b\.?",  re.IGNORECASE),
    "1b":  re.compile(r"\bitem\s+1[\s.]*b\b\.?",  re.IGNORECASE),
    "1c":  re.compile(r"\bitem\s+1[\s.]*c\b\.?",  re.IGNORECASE),
    "2":   re.compile(r"\bitem\s+2\b\.?",          re.IGNORECASE),
    "7":   re.compile(r"\bitem\s+7\b\.?(?![\s.]*a)", re.IGNORECASE),
    "7a":  re.compile(r"\bitem\s+7[\s.]*a\b\.?",  re.IGNORECASE),
    "8":   re.compile(r"\bitem\s+8\b\.?",          re.IGNORECASE),
}

# Subtitle keywords expected right after the section header
_SUBTITLE_KW: dict[str, list[str]] = {
    "1a": ["risk", "ris k"],
    "7":  ["management", "discussion", "md&a", "md a"],
}

# Which items terminate each section
_END_ITEMS: dict[str, list[str]] = {
    "1a": ["1b", "1c", "2"],
    "7":  ["7a", "8"],
}

_EXHIBIT_RE = re.compile(r"exhibit\s+\(?(\d+(?:\.\d+)?)\)?", re.IGNORECASE)
_MARKER_CHAR = "\x00"

# Title-only patterns for exhibits (no "Item X" prefix)
_TITLE_RE: dict[str, re.Pattern] = {
    "1a":  re.compile(r"\brisk\s+factors\b",                          re.IGNORECASE),
    "1b":  re.compile(r"\bunresolved\s+staff\s+comments\b",           re.IGNORECASE),
    "2":   re.compile(r"\bproperties\b",                              re.IGNORECASE),
    "7":   re.compile(r"\bmanagement.s\s+discussion",                 re.IGNORECASE),
    "7a":  re.compile(r"\bquantitative\s+and\s+qualitative",          re.IGNORECASE),
    "8":   re.compile(r"\b(?:consolidated\s+)?financial\s+statements", re.IGNORECASE),
}


# ---------------------------------------------------------------------------
# SEC EDGAR helpers
# ---------------------------------------------------------------------------

def _build_primary_doc_map(cik: str, include_older: bool = False) -> dict[str, str]:
    """Fetch Submissions API and return {accession_number: primaryDocument}.

    By default only returns the "recent" array (≤1000 filings).
    Set *include_older* to also fetch older filing batches (slower).
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=15)
    if r.status_code != 200:
        return {}
    data = r.json()
    filings = data.get("filings", {})
    recent = filings.get("recent", {})
    accns = list(recent.get("accessionNumber", []))
    docs = list(recent.get("primaryDocument", []))

    if include_older:
        for file_entry in filings.get("files", []):
            fname = file_entry.get("name", "")
            if not fname:
                continue
            batch_url = f"https://data.sec.gov/submissions/{fname}"
            br = requests.get(batch_url, headers=SEC_HEADERS, timeout=15)
            time.sleep(REQUEST_SLEEP)
            if br.status_code != 200:
                continue
            batch = br.json()
            accns.extend(batch.get("accessionNumber", []))
            docs.extend(batch.get("primaryDocument", []))

    return dict(zip(accns, docs))


def _download_html(cik: str, accn: str, doc: str) -> str:
    accn_nd = accn.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn_nd}/{doc}"
    r = requests.get(url, headers=SEC_HEADERS, timeout=60)
    r.raise_for_status()
    return r.text


def _fetch_exhibit_map(cik: str, accn: str) -> dict[str, str]:
    """Parse filing index → {exhibit_type: filename}.

    E.g. {'EX-99.1': 'clx-20240630_d2.htm', 'EX-21': 'ex21.htm'}.
    """
    accn_nd = accn.replace("-", "")
    url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
           f"{accn_nd}/{accn}-index.htm")
    r = requests.get(url, headers=SEC_HEADERS, timeout=15)
    if r.status_code != 200:
        return {}
    soup = BeautifulSoup(r.text, "html.parser")
    exhibit_map: dict[str, str] = {}
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 4:
            dtype = cells[3].get_text(strip=True)   # Type column
            doc_cell = cells[2]                      # Document column
            # Prefer the <a> href/text (avoids "iXBRL" annotation)
            link = doc_cell.find("a")
            fname = (link.get("href", link.get_text(strip=True))
                     if link else doc_cell.get_text(strip=True))
            # href may be a relative path — keep just the filename
            fname = fname.rsplit("/", 1)[-1]
            if dtype.startswith("EX-"):
                exhibit_map[dtype] = fname
    return exhibit_map


# ---------------------------------------------------------------------------
# HTML-aware section header detection
# ---------------------------------------------------------------------------

def _has_bold_style(tag) -> bool:
    """True if *tag* or its immediate children have bold styling."""
    if tag.find(["b", "strong"]):
        return True
    for el in [tag] + list(tag.find_all(["span", "font"], recursive=False)):
        style = (el.get("style", "") or "").lower()
        if re.search(r"font-weight:\s*(?:bold|700)", style):
            return True
    return False


def _has_large_font(tag, threshold: float = 12.0) -> bool:
    """True if *tag* or its children use a font-size >= *threshold* pt
    or the legacy HTML <font size> attribute >= 3."""
    for el in [tag] + list(tag.find_all(["span", "font"], recursive=False)):
        style = (el.get("style", "") or "").lower()
        m = re.search(r"font-size:\s*([\d.]+)\s*pt", style)
        if m and float(m.group(1)) >= threshold:
            return True
        size_attr = el.get("size", "")
        if size_attr and str(size_attr).isdigit() and int(size_attr) >= 3:
            return True
    return False


def _find_html_headers(soup) -> dict[str, list]:
    """Detect real section headers in the HTML DOM.

    Matches _ITEM_RE first (plain key like "1a"), then _TITLE_RE (prefixed
    key like "t_1a") so item-based detections take priority over title-only.

    A header must be bold/ALL-CAPS/large-font, ≤25 words, not a TOC link,
    and not a "(Continued)" running header.
    """
    results: dict[str, list] = {}
    for tag in soup.find_all(
        ["p", "div", "td", "h1", "h2", "h3", "h4", "h5", "h6"]
    ):
        txt = tag.get_text(separator="", strip=True)
        wc = len(txt.split())
        if wc > 25 or wc < 2:
            continue
        if "continued" in txt.lower():
            continue
        if not (_has_bold_style(tag) or (txt.upper() == txt and wc >= 2)
               or _has_large_font(tag)):
            continue
        if any(a.get("href") and a.get_text(strip=True)
               for a in tag.find_all("a")):
            continue
        matched = False
        for key, pat in _ITEM_RE.items():
            if pat.search(txt):
                results.setdefault(key, []).append(tag)
                matched = True
                break
        if not matched:
            for key, pat in _TITLE_RE.items():
                if pat.search(txt):
                    results.setdefault(f"t_{key}", []).append(tag)
                    break
    return results


# ---------------------------------------------------------------------------
# HTML → plain text with header position tracking
# ---------------------------------------------------------------------------

def _parse_html(html: str) -> tuple[str, dict[str, list[int]]]:
    """Parse HTML, detect section headers, and convert to plain text.

    Returns (plain_text, {key: [positions]}).  Keys include both item-based
    ("1a", "7") and title-based ("t_1a", "t_7") detections.
    """
    soup = BeautifulSoup(html, "html.parser")

    # --- Phase 1: find headers in the original DOM ---
    headers = _find_html_headers(soup)

    # --- Phase 2: insert unique markers at each header ---
    markers: dict[str, str] = {}          # {marker_text: item_key}
    idx = 0
    for key, elements in headers.items():
        for el in elements:
            marker = f"{_MARKER_CHAR}{key}_{idx}{_MARKER_CHAR}"
            el.insert(0, NavigableString(marker))
            markers[marker] = key
            idx += 1

    # --- Phase 3: convert to plain text ---
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all("ix:header"):
        tag.decompose()
    for tag in soup.find_all(list(_INLINE_TAGS)):
        tag.unwrap()
    soup.smooth()
    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    text = text.strip()

    # --- Phase 4: extract positions from markers, then clean ---
    marker_locs: list[tuple[int, int, str]] = []
    for marker, key in markers.items():
        pos = text.find(marker)
        if pos >= 0:
            marker_locs.append((pos, len(marker), key))
    marker_locs.sort()

    positions: dict[str, list[int]] = {}
    cum_removed = 0
    for pos, mlen, key in marker_locs:
        positions.setdefault(key, []).append(pos - cum_removed)
        cum_removed += mlen

    for marker in markers:
        text = text.replace(marker, "")

    return text, positions


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

def _extract(text: str, target: str,
             positions: dict[str, list[int]],
             allow_stub: bool = False) -> str | None:
    """Extract a section using HTML-verified header positions.

    Tries item-based positions first ("1a"), then title-based ("t_1a").
    End markers prefer item-based positions for precision.
    """
    keywords = _SUBTITLE_KW.get(target, [])
    end_items = _END_ITEMS[target]

    def _section_end(start: int) -> int:
        end = len(text)
        for ei in end_items:
            # Item-based end markers preferred; title-based fallback
            ends = sorted(positions.get(ei, []) or
                          positions.get(f"t_{ei}", []))
            for ep in ends:
                if ep > start + 100:
                    end = min(end, ep)
                    break
        return end

    def _section_len(start: int) -> int:
        end = _section_end(start)
        wc = len(text[start:end].split())
        return wc if end < len(text) else min(wc, MIN_SECTION_WORDS - 1)

    def _is_toc(p: int) -> bool:
        nearby = 0
        for k, plist in positions.items():
            if k in (target, f"t_{target}"):
                continue
            for op in plist:
                if abs(op - p) < 300:
                    nearby += 1
        return nearby >= 2

    # Try item-based candidates, then title-based
    for key in (target, f"t_{target}"):
        candidates = list(positions.get(key, []))
        if not candidates:
            continue

        if keywords:
            candidates = [p for p in candidates
                          if any(kw in text[p:p + 150].lower()
                                 for kw in keywords)]
        if not candidates:
            continue

        non_toc = [p for p in candidates if not _is_toc(p)]
        if non_toc:
            candidates = non_toc

        best = max(candidates, key=_section_len)
        section = text[best:_section_end(best)].strip()

        if len(section.split()) >= MIN_SECTION_WORDS:
            return section
        if allow_stub and section:
            return section

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_filing_text(cik: str, accn: str,
                        doc_map: dict[str, str] | None = None) -> dict[str, str | None]:
    """Download one 10-K and extract Item 1A and Item 7.

    Falls back to exhibit files for "incorporated by reference" stubs.
    Returns {"item_1a": text|None, "item_7": text|None}.
    """
    doc = (doc_map or {}).get(accn)
    if not doc:
        dm = _build_primary_doc_map(cik, include_older=True)
        doc = dm.get(accn)
        time.sleep(REQUEST_SLEEP)
    if not doc:
        return {"item_1a": None, "item_7": None}

    html = _download_html(cik, accn, doc)
    time.sleep(REQUEST_SLEEP)
    text, positions = _parse_html(html)

    result: dict[str, str | None] = {}
    exhibit_map: dict[str, str] | None = None   # lazy-fetched

    for target, out_key in [("1a", "item_1a"), ("7", "item_7")]:
        section = _extract(text, target, positions)

        # Exhibit fallback for "incorporated by reference" stubs
        if section is None:
            stub = _extract(text, target, positions, allow_stub=True)
            if stub and "incorporat" in stub.lower():
                m = _EXHIBIT_RE.search(stub)
                ex_numbers = [m.group(1)] if m else ["13"]
                if exhibit_map is None:
                    exhibit_map = _fetch_exhibit_map(cik, accn)
                    time.sleep(REQUEST_SLEEP)
                for ex_num in ex_numbers:
                    ex_doc = exhibit_map.get(f"EX-{ex_num}")
                    if ex_doc:
                        try:
                            ex_html = _download_html(cik, accn, ex_doc)
                            time.sleep(REQUEST_SLEEP)
                            ex_text, ex_pos = _parse_html(ex_html)
                            section = _extract(ex_text, target, ex_pos)
                        except Exception:
                            pass
                    if section is not None:
                        break

        result[out_key] = section

    return result


def build_text_dataset(panel: pd.DataFrame, cik_map: dict[str, str]) -> pd.DataFrame:
    """Extract Item 1A & Item 7 for every filing in the panel.

    Args:
        panel:   DataFrame with columns [ticker, accession_number, fiscal_year].
        cik_map: {ticker: cik} mapping (10-digit zero-padded).

    Returns:
        DataFrame with columns [ticker, fiscal_year, item_1a, item_7].
        Saved incrementally to data/text_sections.csv for resume.
    """
    out_path = DATA_DIR / "text_sections.csv"

    # Resume: load already-extracted rows
    done: set[str] = set()
    if out_path.exists():
        prev = pd.read_csv(out_path, usecols=["ticker", "fiscal_year"],
                           on_bad_lines="skip")
        done = set(prev["ticker"] + "_" + prev["fiscal_year"].astype(str))
        print(f"  Resuming: {len(done)} filings already extracted")

    # Skip 10-K/A amendments (cover pages only, no extractable text)
    if "form_type" in panel.columns:
        panel = panel[panel["form_type"] != "10-K/A"]

    rows = panel[["ticker", "accession_number", "fiscal_year"]].dropna(
        subset=["accession_number"]
    ).drop_duplicates()
    total = len(rows)
    skipped = extracted = failed = 0

    # Pre-fetch primary-document maps (one Submissions API call per firm)
    doc_maps: dict[str, dict[str, str]] = {}  # {cik: {accn: doc}}
    unique_ciks = {cik_map[t] for t in rows["ticker"].unique() if t in cik_map}
    print(f"  Fetching primary-document index for {len(unique_ciks)} firms …")
    for cik in sorted(unique_ciks):
        doc_maps[cik] = _build_primary_doc_map(cik)
        time.sleep(REQUEST_SLEEP)

    for i, (_, r) in enumerate(rows.iterrows()):
        key = f"{r.ticker}_{int(r.fiscal_year)}"
        if key in done:
            skipped += 1
            continue

        cik = cik_map.get(r.ticker)
        if not cik:
            failed += 1
            continue

        try:
            sections = extract_filing_text(cik, r.accession_number,
                                           doc_maps.get(cik))
        except Exception as e:
            print(f"  ✗ {r.ticker} FY{int(r.fiscal_year)}: {e}")
            sections = {"item_1a": None, "item_7": None}
            failed += 1

        rec = {
            "ticker": r.ticker,
            "fiscal_year": int(r.fiscal_year),
            "item_1a": sections["item_1a"],
            "item_7": sections["item_7"],
        }

        # Append incrementally (one row at a time for crash safety)
        pd.DataFrame([rec]).to_csv(
            out_path, mode="a", header=not out_path.exists(), index=False,
        )
        extracted += 1
        done.add(key)

        i1a_w = len(sections["item_1a"].split()) if sections["item_1a"] else 0
        i7_w = len(sections["item_7"].split()) if sections["item_7"] else 0
        if (extracted % 50 == 0) or (i == total - 1):
            print(f"  [{extracted + skipped}/{total}] "
                  f"{r.ticker} FY{int(r.fiscal_year)}: "
                  f"1A={i1a_w:,}w  7={i7_w:,}w")

    print(f"\n  Done: {extracted} extracted, {skipped} resumed, {failed} failed"
          f" / {total} total")

    return pd.read_csv(out_path)


def load_cik_map() -> dict[str, str]:
    """Fetch the SEC ticker→CIK mapping (cached to disk)."""
    cache_path = DATA_DIR / "cik_map.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path, dtype=str)
        return dict(zip(df["ticker"], df["cik"]))

    data = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS, timeout=15,
    ).json()
    rows = [{"ticker": data[str(i)]["ticker"].upper(),
             "cik": str(data[str(i)]["cik_str"]).zfill(10)}
            for i in range(len(data))]
    df = pd.DataFrame(rows).drop_duplicates(subset="ticker", keep="first")
    df.to_csv(cache_path, index=False)
    print(f"  CIK map: {len(df)} tickers → {cache_path.name}")
    return dict(zip(df["ticker"], df["cik"]))
