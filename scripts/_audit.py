"""Comprehensive extraction audit – prints detailed diagnostics per section."""
import time, re, sys
from text_parser import extract_filing_text, load_cik_map, _build_primary_doc_map
from config import DATA_DIR, REQUEST_SLEEP
import pandas as pd

SEED = int(sys.argv[1]) if sys.argv[1:] and sys.argv[1].isdigit() else 401
N = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 10

_NUM_TOKEN = re.compile(r"^[\d$%,.()\-–−]+$")

cik_map = load_cik_map()
df = pd.read_csv(DATA_DIR / "annual_panel.csv")
rows = df[["ticker", "accession_number", "fiscal_year"]].dropna(
    subset=["accession_number"]
).drop_duplicates()
sample = rows.sample(N, random_state=SEED)

for _, r in sample.iterrows():
    tk, accn, fy = r["ticker"], r["accession_number"], int(r["fiscal_year"])
    cik = cik_map.get(tk)
    if not cik:
        continue
    dm = _build_primary_doc_map(cik, include_older=True)
    time.sleep(REQUEST_SLEEP)
    ex = extract_filing_text(cik, accn, dm)
    time.sleep(REQUEST_SLEEP)
    for sec in ["item_1a", "item_7"]:
        txt = ex[sec]
        if not txt:
            print(f"=== {tk} FY{fy} {sec} [MISSING] ===\n")
            continue
        words = txt.split()
        wc = len(words)
        # --- Artifact counts ---
        solo_dollar = len(re.findall(r"(?<!\w)\$\s*(?!\d)", txt))
        solo_pct = len(re.findall(r"(?<!\d)%(?!\d)", txt))
        toc = txt.lower().count("table of contents")
        html_tags = len(re.findall(r"<[a-zA-Z/][^>]*>", txt))
        nbsp = txt.count("\xa0")
        num_clusters = len(re.findall(r"(?:\d[\d,.]+\s+){4,}", txt))
        page_footers = len(re.findall(
            r"(?m)^\s*(?:\d{4}\s+)?(?:Form\s+10-K|Annual\s+Report)\s+\d{1,3}\s*$", txt))
        exhibit_refs = len(re.findall(r"\bExhibit\s+\d", txt))
        sig_block = len(re.findall(r"(?i)(?:pursuant to the requirements|power of attorney|/s/)", txt))
        # Garbled lines: short numeric-only lines in sequence
        lines = txt.split("\n")
        numeric_only_lines = sum(1 for ln in lines if ln.strip() and _NUM_TOKEN.match(ln.strip()))
        # Very short or very long
        flag = ""
        if wc < 500:
            flag = " *** SHORT"
        elif wc > 25000:
            flag = " *** VERY LONG"
        if sig_block >= 3:
            flag += " *** SIGNATURE BLOCK?"
        if html_tags > 0:
            flag += " *** HTML REMNANTS"
        if toc > 0:
            flag += " *** TOC LEAK"
        if num_clusters > 20:
            flag += f" *** GARBLED TABLES ({num_clusters})"
        if numeric_only_lines > 20:
            flag += f" *** NUM-LINES ({numeric_only_lines})"

        first80 = " ".join(words[:80])
        last60 = " ".join(words[-60:])
        mid_start = wc // 2 - 60
        mid80 = " ".join(words[max(0,mid_start):mid_start+120])

        print(f"=== {tk} FY{fy} {sec} ({wc:,}w){flag} ===")
        print(f"  solo$={solo_dollar} solo%={solo_pct} toc={toc} html={html_tags} "
              f"nbsp={nbsp} clusters={num_clusters} pgfoot={page_footers} "
              f"exhibit={exhibit_refs} sig={sig_block} numlines={numeric_only_lines}")
        print(f"  START: {first80[:400]}")
        print(f"  MID:   {mid80[:400]}")
        print(f"  END:   {last60[-400:]}")
        print()
