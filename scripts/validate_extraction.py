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
import httpx

import pandas as pd
from google import genai

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
N_SAMPLE = 100
SEED = 42
MODEL = "gemini-2.0-flash"

PROMPT_TEMPLATE = """\
You are validating the quality of an automated text extraction from an SEC 10-K filing.

**Filing**: {ticker} (fiscal year {fy})
**Target section**: {section_name}

Below is the COMPLETE extracted text ({total_words:,} words).

--- START OF EXTRACTED TEXT ---
{full_text}
--- END OF EXTRACTED TEXT ---

Judge the extraction on these criteria:
1. **Correct section**: Does this text belong to {section_name}? (Yes/No)
2. **Start boundary**: Does the text start at or near the section header? (Yes/No)
3. **End boundary**: Does the text end where the section should end — before the next item starts, not mid-sentence? (Yes/No)
4. **Content quality**: Is the text clean prose (not table-of-contents, not exhibit list, not garbled HTML)? (Yes/No)

Respond with EXACTLY this format (no extra text):
correct_section: Yes/No
start_boundary: Yes/No
end_boundary: Yes/No
content_quality: Yes/No
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


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable first.")
        print('  $env:GEMINI_API_KEY = "your-key-here"')
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Load panel and sample
    df = pd.read_csv(DATA_DIR / "annual_panel.csv")
    if "form_type" in df.columns:
        df = df[df["form_type"] != "10-K/A"]
    df = df.dropna(subset=["accession_number"])

    random.seed(SEED)
    sample = df.sample(n=min(N_SAMPLE, len(df)), random_state=SEED)
    print(f"Sampled {len(sample)} filings for validation")

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
                    "content_quality": "ERROR", "notes": str(e),
                })
            continue

        for target, out_key, sec_name in sections:
            txt = extracted[out_key]
            if not txt:
                results.append({
                    "ticker": ticker, "fiscal_year": fy, "section": out_key,
                    "total_words": 0, "correct_section": "MISSING",
                    "start_boundary": "MISSING", "end_boundary": "MISSING",
                    "content_quality": "MISSING", "notes": "No text extracted",
                })
                continue

            words = txt.split()
            total_words = len(words)

            prompt = PROMPT_TEMPLATE.format(
                ticker=ticker, fy=fy, section_name=sec_name,
                full_text=txt, total_words=total_words,
            )

            try:
                response = client.models.generate_content(
                    model=MODEL, contents=prompt,
                )
                parsed = _parse_response(response.text)
            except Exception as e:
                parsed = {
                    "correct_section": "API_ERROR",
                    "start_boundary": "API_ERROR",
                    "end_boundary": "API_ERROR",
                    "content_quality": "API_ERROR",
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
                "notes": parsed.get("notes", ""),
            })

            # Gemini free tier: 15 RPM
            time.sleep(4.5)

        if (idx + 1) % 10 == 0 or idx == len(sample) - 1:
            ok = sum(1 for r in results
                     if r["correct_section"] == "Yes"
                     and r["start_boundary"] == "Yes"
                     and r["end_boundary"] == "Yes"
                     and r["content_quality"] == "Yes")
            total = sum(1 for r in results if r["correct_section"] not in
                        ("ERROR", "MISSING", "API_ERROR", "PARSE_ERR"))
            rate = f"{ok}/{total} ({ok/total:.0%})" if total else "N/A"
            print(f"  [{idx+1}/{len(sample)}] {ticker} FY{fy} — running accuracy: {rate}")

    # Save results
    out = pd.DataFrame(results)
    out_path = DATA_DIR / "extraction_validation.csv"
    out.to_csv(out_path, index=False)

    # Summary
    judged = out[~out["correct_section"].isin(
        ["ERROR", "MISSING", "API_ERROR", "PARSE_ERR"])]
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY ({len(judged)} sections judged)")
    print(f"{'='*60}")
    for col in ["correct_section", "start_boundary", "end_boundary", "content_quality"]:
        yes = (judged[col] == "Yes").sum()
        print(f"  {col:20s}: {yes}/{len(judged)} ({yes/len(judged):.1%})")

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
