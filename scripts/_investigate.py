"""Test DGX FY2016 end boundary fix and full regression."""
import time, re
from text_parser import (load_cik_map, _build_primary_doc_map, extract_filing_text)
from config import DATA_DIR, REQUEST_SLEEP
import pandas as pd

cik_map = load_cik_map()
df = pd.read_csv(DATA_DIR / "annual_panel.csv")

# --- DGX FY2016 ---
print("=== DGX FY2016 ===")
sub = df[(df["ticker"] == "DGX") & (df["fiscal_year"] == 2016)]
cik = cik_map["DGX"]
accn = sub.iloc[0]["accession_number"]
dm = _build_primary_doc_map(cik, include_older=True)
time.sleep(REQUEST_SLEEP)
res = extract_filing_text(cik, accn, dm)
time.sleep(REQUEST_SLEEP)
txt = res["item_7"]
if txt:
    print(f"Words: {len(txt.split())}")
    for kw in ["Liquidity and Capital Resources", "Contractual Obligations",
                "Critical Accounting", "Quantitative and Qualitative"]:
        found = kw.lower() in txt.lower()
        print(f"  {kw}: {'FOUND' if found else 'NOT FOUND'}")
    print(f"  Last 150: ...{txt[-150:]}")
else:
    print("MISSING")

# --- Full regression ---
print("\n=== REGRESSION CHECK ===")
cases = [
    ("SPGI", 2023), ("PHM", 2017), ("HAL", 2017), ("RMD", 2014),
    ("ADSK", 2019), ("MSCI", 2019), ("BIIB", 2015), ("DGX", 2018),
    ("AVY", 2020), ("IBM", 2023), ("GE", 2024), ("DGX", 2016),
    # Also test filings from validation
    ("ALLE", 2017), ("DHR", 2011),
]
for ticker, fy in cases:
    sub = df[(df["ticker"] == ticker) & (df["fiscal_year"] == fy)]
    if len(sub) == 0:
        print(f"{ticker} FY{fy}: NO DATA")
        continue
    cik = cik_map[ticker]
    dm = _build_primary_doc_map(cik, include_older=True)
    accn = sub.iloc[0]["accession_number"]
    time.sleep(REQUEST_SLEEP)
    res = extract_filing_text(cik, accn, dm)
    time.sleep(REQUEST_SLEEP)
    for k, v in res.items():
        if v:
            wc = len(v.split())
            status = "OK" if wc >= 200 else "SHORT"
            print(f"{ticker} FY{fy} {k}: {wc}w {status}")
        else:
            print(f"{ticker} FY{fy} {k}: MISSING")
