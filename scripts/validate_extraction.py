"""AI-assisted validation of 10-K section extraction.

Samples 100 random filings, extracts Item 1A and Item 7 using our parser,
and sends the start/end of each extracted section to Gemini Flash for
independent quality judgment.

Usage:
    $env:GEMINI_API_KEY = "your-key-here"
    python validate_extraction.py
"""

import os
import sys
import ssl
import time
import random

import pandas as pd
try:
    from google import genai
except Exception:
    genai = None

# Re-use the Windows CA cert export from config (sets CURL_CA_BUNDLE / REQUESTS_CA_BUNDLE)
from config import DATA_DIR, REQUEST_SLEEP
from text_parser import (
    extract_filing_text, load_cik_map, _build_primary_doc_map,
)

# Also set SSL_CERT_FILE for httpx (used by Gemini SDK)
_ca = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
if _ca:
    os.environ["SSL_CERT_FILE"] = _ca

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Validation sample size
# With 20 RPD and 2 section calls per filing, max practical daily sample is 10 filings.
N_SAMPLE = 10
SEED = 42
MODEL = "gemini-2.5-flash"
GEMINI_API_KEY = "AIzaSyDS-OhwiU1IuN03Xd9mRaHmU-NT2Hypgwg"
MAX_API_RETRIES = 3
RETRY_BACKOFF_SECONDS = 12
# 5 RPM => one request every 12 seconds.
REQUEST_GAP_SECONDS = 13
CHECKPOINT_EVERY = 5

# Era boundaries for stratified sampling
_ERA_BINS = [(2010, 2014), (2015, 2019), (2020, 2024)]

PROMPT_TEMPLATE = """\
You are validating the quality of an automated text extraction from an SEC 10-K filing.

**Filing**: {ticker} (fiscal year {fy})
**Target section**: {section_name}

Below is the COMPLETE extracted text ({total_words:,} words).

--- START OF EXTRACTED TEXT ---
{full_text}
--- END OF EXTRACTED TEXT ---

Judge the extraction on ALL of these criteria:
1. **Correct section**: Does this text genuinely belong to {section_name}? (Yes/No)
2. **Start boundary**: Does the text start at or very near the section header, not with content from a prior section? (Yes/No)
3. **End boundary**: Does the text end where the section should end — before the next Item starts, not mid-sentence or mid-paragraph? (Yes/No)
4. **Content quality**: Is the text clean readable prose? Answer No if you find ANY of these artifacts:
   - "Table of Contents" running page headers or page numbers between paragraphs
   - Residual HTML tags or XBRL inline tags (e.g. <ix:nonfraction>, <div>, &nbsp;)
   - Large blocks of table data that are garbled (numbers without column context)
   - Exhibit lists or index pages instead of narrative content
5. **Completeness**: Does the section appear to contain the full expected content (not truncated, not just a stub or cross-reference to an exhibit)? (Yes/No)

Respond with EXACTLY this format (no extra text):
correct_section: Yes/No
start_boundary: Yes/No
end_boundary: Yes/No
content_quality: Yes/No
completeness: Yes/No
notes: <one-line explanation if any No, otherwise "OK">
"""


def _parse_response(text: str) -> dict:
    """Parse structured Gemini response into a dict."""
    result = {}
    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        result[key] = val.strip()
    return result


def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).upper()
    return ("429" in msg) or ("RESOURCE_EXHAUSTED" in msg) or ("RATE" in msg and "LIMIT" in msg)


def _generate_with_retry(client, prompt: str) -> str:
    """Call Gemini with basic retry/backoff for transient quota throttling."""
    last_err = None
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            return response.text or ""
        except Exception as e:
            last_err = e
            if _is_rate_limited(e) and attempt < MAX_API_RETRIES:
                sleep_s = RETRY_BACKOFF_SECONDS * attempt
                print(f"    Rate-limited (attempt {attempt}/{MAX_API_RETRIES}), sleeping {sleep_s}s ...")
                time.sleep(sleep_s)
                continue
            raise
    raise last_err


def main():
    if genai is None:
        print("ERROR: Missing Google GenAI SDK.")
        print("Install with: pip install google-genai")
        sys.exit(1)

    api_key = GEMINI_API_KEY
    if not api_key:
        print("ERROR: GEMINI_API_KEY is empty in script.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Load panel and build unique filing rows (same style as text_parser)
    df = pd.read_csv(DATA_DIR / "annual_panel.csv")
    if "form_type" in df.columns:
        df = df[df["form_type"] != "10-K/A"]
    rows = (
        df[["ticker", "accession_number", "fiscal_year"]]
        .dropna(subset=["accession_number"])
        .drop_duplicates()
    )

    out_path = DATA_DIR / "extraction_validation.csv"

    # Guard: exclude firm-years already validated in previous runs
    done_firm_years: set[tuple[str, int]] = set()
    if out_path.exists():
        try:
            prev = pd.read_csv(out_path, usecols=["ticker", "fiscal_year"], on_bad_lines="skip")
            prev = prev.dropna(subset=["ticker", "fiscal_year"]).drop_duplicates()
            done_firm_years = {
                (str(t), int(fy))
                for t, fy in zip(prev["ticker"], prev["fiscal_year"])
            }
        except Exception as e:
            print(f"WARNING: Could not read prior validation file for dedupe guard: {e}")

    if done_firm_years:
        before = len(rows)
        keys = list(zip(rows["ticker"].astype(str), rows["fiscal_year"].astype(int)))
        mask_new = [k not in done_firm_years for k in keys]
        rows = rows.loc[mask_new].copy()
        excluded = before - len(rows)
        print(f"Excluded {excluded} already validated firm-years from sampling pool")

    if rows.empty:
        print("No new firm-years left to validate. Delete/rename extraction_validation.csv to resample from scratch.")
        return

    random.seed(SEED)
    # Stratified sampling: equal draws from each era for diverse coverage
    per_era = max(1, N_SAMPLE // len(_ERA_BINS))
    strata = []
    for lo, hi in _ERA_BINS:
        era_rows = rows[(rows["fiscal_year"] >= lo) & (rows["fiscal_year"] <= hi)]
        n_draw = min(per_era, len(era_rows))
        if n_draw > 0:
            strata.append(era_rows.sample(n=n_draw, random_state=SEED))
    sample = pd.concat(strata, ignore_index=True)
    # Shuffle so eras are interleaved (avoids sequential doc-map cache misses)
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
    era_counts = {f"{lo}-{hi}": ((sample["fiscal_year"] >= lo) & (sample["fiscal_year"] <= hi)).sum()
                  for lo, hi in _ERA_BINS}
    print(f"Sampled {len(sample)} filings for validation (eras: {era_counts})")

    cik_map = load_cik_map()

    # Pre-fetch doc maps
    doc_maps: dict[str, dict] = {}
    unique_ciks = {cik_map[t] for t in sample["ticker"].unique() if t in cik_map}
    print(f"Fetching doc indices for {len(unique_ciks)} firms ...")
    for cik in sorted(unique_ciks):
        doc_maps[cik] = _build_primary_doc_map(cik)
        time.sleep(REQUEST_SLEEP)

    # Validate
    results = []
    sections = [("1a", "item_1a", "Item 1A (Risk Factors)"),
                ("7",  "item_7",  "Item 7 (MD&A)")]

    for idx, (_, row) in enumerate(sample.iterrows()):
        ticker = row["ticker"]
        fy = int(row["fiscal_year"])
        accn = row["accession_number"]
        cik = cik_map.get(ticker)
        if not cik:
            continue

        try:
            extracted = extract_filing_text(cik, accn, doc_maps.get(cik))
        except Exception as e:
            print(f"  [{idx+1}/{len(sample)}] {ticker} FY{fy}: extraction error — {e}")
            for _, out_key, sec_name in sections:
                results.append({
                    "ticker": ticker, "fiscal_year": fy, "section": out_key,
                    "total_words": 0, "correct_section": "ERROR",
                    "start_boundary": "ERROR", "end_boundary": "ERROR",
                    "content_quality": "ERROR", "completeness": "ERROR",
                    "notes": str(e),
                    "extracted_text": "",
                    "ai_raw_response": "",
                })
            continue

        for target, out_key, sec_name in sections:
            txt = extracted[out_key]
            if not txt:
                results.append({
                    "ticker": ticker, "fiscal_year": fy, "section": out_key,
                    "total_words": 0, "correct_section": "MISSING",
                    "start_boundary": "MISSING", "end_boundary": "MISSING",
                    "content_quality": "MISSING", "completeness": "MISSING",
                    "notes": "No text extracted",
                    "extracted_text": "",
                    "ai_raw_response": "",
                })
                continue

            words = txt.split()
            total_words = len(words)

            prompt = PROMPT_TEMPLATE.format(
                ticker=ticker, fy=fy, section_name=sec_name,
                full_text=txt, total_words=total_words,
            )

            try:
                raw = _generate_with_retry(client, prompt)
                parsed = _parse_response(raw)
            except Exception as e:
                raw = ""
                parsed = {
                    "correct_section": "API_ERROR",
                    "start_boundary": "API_ERROR",
                    "end_boundary": "API_ERROR",
                    "content_quality": "API_ERROR",
                    "completeness": "API_ERROR",
                    "notes": str(e),
                }

            results.append({
                "ticker": ticker,
                "fiscal_year": fy,
                "section": out_key,
                "total_words": total_words,
                "correct_section": parsed.get("correct_section", "PARSE_ERR"),
                "start_boundary": parsed.get("start_boundary", "PARSE_ERR"),
                "end_boundary": parsed.get("end_boundary", "PARSE_ERR"),
                "content_quality": parsed.get("content_quality", "PARSE_ERR"),
                "completeness": parsed.get("completeness", "PARSE_ERR"),
                "notes": parsed.get("notes", ""),
                "extracted_text": txt,
                "ai_raw_response": raw,
            })

            # Conservative pacing to avoid free-tier throttling
            time.sleep(REQUEST_GAP_SECONDS)

        # Checkpoint partial progress so long runs are resumable/inspectable
        if (idx + 1) % CHECKPOINT_EVERY == 0 or idx == len(sample) - 1:
            pd.DataFrame(results).to_csv(out_path, index=False)
            print(f"    checkpoint saved: {out_path} ({len(results)} rows)")

        if (idx + 1) % 10 == 0 or idx == len(sample) - 1:
            ok = sum(1 for r in results
                     if r["correct_section"] == "Yes"
                     and r["start_boundary"] == "Yes"
                     and r["end_boundary"] == "Yes"
                     and r["content_quality"] == "Yes"
                     and r["completeness"] == "Yes")
            total = sum(1 for r in results if r["correct_section"] not in
                        ("ERROR", "MISSING", "API_ERROR", "PARSE_ERR"))
            rate = f"{ok}/{total} ({ok/total:.0%})" if total else "N/A"
            print(f"  [{idx+1}/{len(sample)}] {ticker} FY{fy} — running accuracy: {rate}")

    # Save results
    out = pd.DataFrame(results)
    out.to_csv(out_path, index=False)

    # Summary
    judged = out[~out["correct_section"].isin(
        ["ERROR", "MISSING", "API_ERROR", "PARSE_ERR"])]
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY ({len(judged)} sections judged)")
    print(f"{'='*60}")
    if len(judged) == 0:
        print("  No sections judged (all rows were API_ERROR / MISSING / PARSE_ERR).")
    else:
        for col in ["correct_section", "start_boundary", "end_boundary", "content_quality", "completeness"]:
            yes = (judged[col] == "Yes").sum()
            print(f"  {col:20s}: {yes}/{len(judged)} ({yes/len(judged):.1%})")

    err_counts = out["correct_section"].value_counts()
    if any(k in err_counts.index for k in ["API_ERROR", "PARSE_ERR", "ERROR"]):
        print("\nError breakdown:")
        for k in ["API_ERROR", "PARSE_ERR", "ERROR", "MISSING"]:
            if k in err_counts.index:
                print(f"  {k:10s}: {int(err_counts[k])}")

    missed = out[out["correct_section"] == "MISSING"]
    if len(missed):
        print(f"\n  Missing extractions: {len(missed)}")
        for _, r in missed.iterrows():
            print(f"    {r.ticker} FY{r.fiscal_year} {r.section}")

    flagged = judged[judged.apply(
        lambda r: "No" in [r.correct_section, r.start_boundary,
                           r.end_boundary, r.content_quality], axis=1)]
    if len(flagged):
        print(f"\n  Flagged ({len(flagged)}):")
        for _, r in flagged.iterrows():
            print(f"    {r.ticker} FY{r.fiscal_year} {r.section}: {r.notes}")

    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
