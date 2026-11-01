"""Back-fill per-filing metadata that the original text run did not store.

For every (ticker, fiscal_year, accession) in data/annual_panel.csv:

* ``sic_at_filing``  — SIC code in the EDGAR header of THAT filing
                        (historical, not today's SIC; #72).
* ``src_1a`` / ``src_7`` — DISABLED by owner decision (no 10-K re-download).
                        Columns stay empty; #26 is handled as a documented
                        limitation.  ``--with-section-source`` re-enables it
                        (downloads every 10-K again — do not use).

Writes data/filing_metadata.csv.  Resumable; 1 small header request per filing.
Usage:  python scripts/backfill_filing_metadata.py
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, SEC_HEADERS, REQUEST_SLEEP
import text_parser as tp

OUT = DATA_DIR / "filing_metadata.csv"
_SIC_RE = re.compile(r"STANDARD INDUSTRIAL CLASSIFICATION:[^\[\n]*\[(\d{3,4})\]")


def _get(url: str) -> str | None:
    for attempt in range(4):
        try:
            r = requests.get(url, headers=SEC_HEADERS, timeout=30)
            if r.status_code == 200:
                return r.text
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(2 ** attempt)
    return None


def sic_at_filing(cik: str, accn: str) -> int | None:
    nod = accn.replace("-", "")
    html = _get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{nod}/"
                f"{accn}-index-headers.html")
    if not html:
        return None
    m = _SIC_RE.search(html)
    return int(m.group(1)) if m else None


def section_source(cik: str, accn: str, doc_map: dict) -> tuple[str, str]:
    doc = doc_map.get(accn)
    if not doc:
        return "none", "none"
    html = tp._download_html(cik, accn, doc)
    text, pos = tp._parse_html(html)
    out = []
    for target in ("1a", "7"):
        sec = tp._extract(text, target, pos)
        out.append("main" if sec is not None else "exhibit_or_none")
    return out[0], out[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-section-source", action="store_true",
                    help="re-download every 10-K (DISABLED by default)")
    a = ap.parse_args()
    a.sic_only = not a.with_section_source

    panel = pd.read_csv(DATA_DIR / "annual_panel.csv",
                        usecols=["ticker", "fiscal_year", "accession_number"])
    panel = panel.dropna(subset=["accession_number"]).drop_duplicates()
    cmap = dict(pd.read_csv(DATA_DIR / "cik_map.csv", dtype=str)[["ticker", "cik"]].values)
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(prev["accession_number"])
    doc_maps: dict[str, dict] = {}
    n = len(panel)
    for i, r in enumerate(panel.itertuples(index=False), 1):
        if r.accession_number in done or r.ticker not in cmap:
            continue
        cik = cmap[r.ticker]
        sic = sic_at_filing(cik, r.accession_number)
        time.sleep(REQUEST_SLEEP)
        s1a = s7 = None
        if not a.sic_only:
            if cik not in doc_maps:
                doc_maps[cik] = tp._build_primary_doc_map(cik, include_older=True)
                time.sleep(REQUEST_SLEEP)
            try:
                s1a, s7 = section_source(cik, r.accession_number, doc_maps[cik])
            except Exception:
                s1a = s7 = "error"
            time.sleep(REQUEST_SLEEP)
        pd.DataFrame([{"ticker": r.ticker, "fiscal_year": int(r.fiscal_year),
                       "accession_number": r.accession_number,
                       "sic_at_filing": sic, "src_1a": s1a, "src_7": s7}]
                     ).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        if i % 100 == 0:
            print(f"  [{i}/{n}] {r.ticker} FY{int(r.fiscal_year)} sic={sic} 1A={s1a}", flush=True)
    print(f"Saved {OUT.name}")


if __name__ == "__main__":
    main()
