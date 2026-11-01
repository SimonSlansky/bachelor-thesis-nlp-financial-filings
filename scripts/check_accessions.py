"""Accession changes between the text-era panel and the rebuilt panel (#18).

The stored Item 1A / Item 7 text (data/text_sections.csv, keyed by ticker and
fiscal year) was extracted from the accessions in data/text_accessions.csv
(panel of git commit 280b73c, the commit that last changed text_sections.csv).
The rebuilt panel selects accessions point-in-time.  For every (ticker, FY)
whose accession changed, the EDGAR Submissions API (JSON only; no filing
documents) gives the form type and period of report of the text accession:

  text still valid  <=> text accession is a 10-K (not 10-K/A) whose period
                        of report is within 10 days of the new year_end
  otherwise         text_mismatch = 1 (excluded from text regressions)

Writes data/accession_changes.csv.  No 10-K document is downloaded.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, SEC_HEADERS, REQUEST_SLEEP  # noqa: E402

CACHE = Path.home() / "scratch" / "bak" / "submissions"
OUT = DATA_DIR / "accession_changes.csv"


def _get(url):
    for k in range(4):
        try:
            r = requests.get(url, headers=SEC_HEADERS, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(2 ** k)
    return None


def filings_index(cik: str) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{cik}.json"
    if p.exists():
        return pd.read_json(p)
    sub = _get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    time.sleep(REQUEST_SLEEP)
    blocks = [sub["filings"]["recent"]] if sub else []
    for f in (sub or {}).get("filings", {}).get("files", []):
        b = _get(f"https://data.sec.gov/submissions/{f['name']}")
        time.sleep(REQUEST_SLEEP)
        if b:
            blocks.append(b)
    cols = ["accessionNumber", "form", "reportDate", "filingDate"]
    df = pd.concat([pd.DataFrame({c: b.get(c, []) for c in cols}) for b in blocks],
                   ignore_index=True) if blocks else pd.DataFrame(columns=cols)
    df.to_json(p)
    return df


def main():
    txt = pd.read_csv(DATA_DIR / "text_accessions.csv")
    new = pd.read_csv(DATA_DIR / "annual_panel.csv",
                      usecols=["ticker", "fiscal_year", "year_end", "accession_number",
                               "form_type"])
    m = new.merge(txt, on=["ticker", "fiscal_year"], how="left")
    m["has_text"] = m["accession_text"].notna()
    m["changed"] = m["has_text"] & (m["accession_text"] != m["accession_number"])
    cmap = dict(pd.read_csv(DATA_DIR / "cik_map.csv", dtype=str)[["ticker", "cik"]].values)
    info = []
    for tk, g in m[m["changed"]].groupby("ticker"):
        idx = filings_index(cmap[tk]).set_index("accessionNumber")
        for _, r in g.iterrows():
            f = idx.loc[r.accession_text] if r.accession_text in idx.index else None
            info.append({"ticker": tk, "fiscal_year": r.fiscal_year,
                         "text_form": None if f is None else f["form"],
                         "text_report_date": None if f is None else f["reportDate"],
                         "text_filing_date": None if f is None else f["filingDate"]})
    if info:
        m = m.merge(pd.DataFrame(info), on=["ticker", "fiscal_year"], how="left")
    else:
        m[["text_form", "text_report_date", "text_filing_date"]] = None
    gap = (pd.to_datetime(m["text_report_date"], errors="coerce")
           - pd.to_datetime(m["year_end"])).dt.days.abs()
    m["same_period_10k"] = m["changed"] & m["text_form"].eq("10-K") & (gap <= 10)
    m["text_mismatch"] = (m["changed"] & ~m["same_period_10k"]).astype(int)
    m.to_csv(OUT, index=False)
    n_text = int(m["has_text"].sum())
    print(f"firm-years with text: {n_text:,}; accession changed: {int(m['changed'].sum())}; "
          f"same-period 10-K: {int(m['same_period_10k'].sum())}; text_mismatch: "
          f"{int(m['text_mismatch'].sum())} ({m['text_mismatch'].sum() / n_text:.2%})")
    print(m.loc[m["changed"], "text_form"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
