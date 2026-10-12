"""AI-assisted validation of SEC 10-K section extraction quality.

Uses Gemini 2.5 Flash (via OpenAI-compatible endpoint) as an independent
judge to evaluate extraction quality on five criteria with a three-level
rubric (Pass / Minor / Fail).

Two-phase workflow
------------------
Phase 1 -- Build sample (extract texts from SEC, no API key needed):
    python validate_extraction.py --build-sample

Phase 2 -- Validate with AI (single paid key, runs all 200 in one go):
    python validate_extraction.py --api-key YOUR_KEY
    python validate_extraction.py --api-key YOUR_KEY --section item_1a

Design:
  - 100 randomly sampled sections per section type (Item 1A / Item 7)
  - Stratified by fiscal-year era
  - Full extracted text saved in sample -> SEC only hit once
  - On API error -> retry (never writes errors to results)
  - Wilson 95 % confidence intervals for all reported proportions
"""

import argparse
import math
import sys
import time

from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, REQUEST_SLEEP          # noqa: E402
from text_parser import (                            # noqa: E402
    extract_filing_text, load_cik_map, _build_primary_doc_map,
)

try:
    from openai import OpenAI as _OpenAI
except Exception:
    _OpenAI = None

# ── constants ─────────────────────────────────────────────────────────────

MODEL = "gemini-2.5-flash"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/"

TEMPERATURE = 0.0
MAX_RETRIES_PER_CALL = 5        # retries for a single API call
RETRY_BACKOFF_S = 20            # base backoff on rate-limit
REQUEST_GAP_S = 10               # paid tier: 2000 RPM, 4s is polite

_ERA_BINS = [(2010, 2012), (2013, 2015), (2016, 2018), (2019, 2021), (2022, 2024)]

CRITERIA = [
    "correct_section", "start_boundary", "end_boundary",
    "content_quality", "completeness",
]
VALID_SCORES = {"Pass", "Minor", "Fail"}

SAMPLE_PATH  = DATA_DIR / "validation_sample.csv"
RESULTS_PATH = DATA_DIR / "extraction_validation.csv"

# ── prompt ────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
You are an independent SEC financial-document quality auditor.

Filing: **{ticker}  FY{fy}**

Below is a section automatically extracted from this company's 10-K filing.
Evaluate it on five quality criteria.

**Important context**: Financial data tables (revenue breakdowns, balance sheets,
contractual obligations, etc.) are *intentionally removed* during extraction.
Only narrative prose is retained.  Do NOT penalize the extraction for missing
tables or for references to tables/figures that are absent.

{sections_block}
---

**Criteria:**

1. **correct_section** -- Does the text belong to the claimed section?
   Item 1A must contain risk-factor descriptions.
   Item 7 must contain management's discussion & analysis of financial
   condition and results of operations.

2. **start_boundary** -- Does the extraction start at or very near the correct
   section header?
   Minor: the literal "Item 1A" / "Item 7" header line is trimmed but
          the substantive content starts correctly.
   Fail:  significant text from a preceding section is included.

3. **end_boundary** -- Does the extraction stop before the next major Item?
   Minor: a few lines of the next Item's header are appended.
   Fail:  substantial text from Item 2, 7A, 8, or financial statements
          is included.

4. **content_quality** -- Is the text clean and usable for NLP analysis?
   Acceptable: imperfect table formatting, minor whitespace quirks.
   Fail: residual HTML/XML tags, garbled characters, binary data, or heavy
         formatting artefacts that would corrupt text analysis.

5. **completeness** -- Does the section appear complete (not truncated)?
   Typical word counts: Item 1A = 3,000-30,000; Item 7 = 5,000-25,000.
   Ignore missing tables -- they are removed by design.

**Scoring rubric:**
  Pass  -- criterion clearly satisfied
  Minor -- small issue; would NOT materially affect NLP analysis
           (sentiment, readability, topic modelling)
  Fail  -- issue that WOULD materially distort NLP results

Respond in **exactly** this format:

{response_format}
"""


# ── helpers ───────────────────────────────────────────────────────────────

def _response_format(section):
    """Build the expected response format string for one section."""
    lines = []
    for c in CRITERIA:
        lines.append(f"{section}_{c}: Pass/Minor/Fail")
    lines.append(f"{section}_notes: <one-line explanation or 'OK'>")
    return "\n".join(lines)


def _parse_response(text):
    out = {}
    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        val = val.strip()
        for v in VALID_SCORES:
            if val.lower().startswith(v.lower()):
                val = v
                break
        out[key] = val
    return out


# ── API ───────────────────────────────────────────────────────────────────

def _fetch_doc_map_safe(cik, max_retries=3):
    """Fetch doc map with retry for SEC timeouts."""
    for attempt in range(1, max_retries + 1):
        try:
            return _build_primary_doc_map(cik)
        except Exception as e:
            if "timeout" in str(e).lower() and attempt < max_retries:
                wait = 10 * attempt
                print(f"      SEC timeout (try {attempt}/{max_retries}), "
                      f"sleeping {wait}s ...")
                time.sleep(wait)
            else:
                raise
    return None


def _call_llm(client, prompt):
    """Call Gemini once (no retry). Returns response text or raises."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )
    return resp.choices[0].message.content or ""


# ── statistics ────────────────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    """Wilson-score 95 % confidence interval for proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# ── summary ───────────────────────────────────────────────────────────────

def _print_summary():
    if not RESULTS_PATH.exists():
        print("\n  No results file yet.")
        return
    try:
        df = pd.read_csv(RESULTS_PATH)
    except Exception:
        print("\n  No results yet.")
        return
    if df.empty:
        print("\n  No results yet.")
        return

    n_total = len(df)
    n_filings = df[["ticker", "fiscal_year"]].drop_duplicates().shape[0]
    print(f"\n{'=' * 70}")
    print(f"VALIDATION SUMMARY   {n_total} sections  |  {n_filings} filings")
    print(f"{'=' * 70}")

    print(f"\n  {'Criterion':<22}{'Strict':>10}{'Lenient':>10}"
          f"{'Fail':>8}  {'95% CI (lenient)'}")
    print(f"  {'-' * 66}")
    for c in CRITERIA:
        n = n_total
        n_p = int((df[c] == "Pass").sum())
        n_m = int((df[c] == "Minor").sum())
        n_f = int((df[c] == "Fail").sum())
        n_err = n - n_p - n_m - n_f
        lenient = n_p + n_m
        lo, hi = wilson_ci(lenient, n)
        err_note = f"  ({n_err} parse-err)" if n_err else ""
        print(f"  {c:<22}{n_p:>3}/{n} {n_p/n:>5.1%}"
              f"  {lenient:>3}/{n} {lenient/n:>5.1%}"
              f"  {n_f:>5}"
              f"   [{lo:.1%} - {hi:.1%}]{err_note}")

    # overall
    strict_all = int(df.apply(
        lambda r: all(r[c] == "Pass" for c in CRITERIA), axis=1).sum())
    lenient_all = int(df.apply(
        lambda r: all(r[c] in ("Pass", "Minor") for c in CRITERIA),
        axis=1).sum())
    n = n_total
    lo_s, hi_s = wilson_ci(strict_all, n)
    lo_l, hi_l = wilson_ci(lenient_all, n)
    print(f"\n  {'Overall strict':<22}{strict_all:>3}/{n} ({strict_all/n:.1%})"
          f"                 [{lo_s:.1%} - {hi_s:.1%}]")
    print(f"  {'Overall lenient':<22}{lenient_all:>3}/{n} ({lenient_all/n:.1%})"
          f"                 [{lo_l:.1%} - {hi_l:.1%}]")

    # by era
    print("\n  Lenient pass rate by era:")
    for lo_yr, hi_yr in _ERA_BINS:
        era = df[(df["fiscal_year"] >= lo_yr) & (df["fiscal_year"] <= hi_yr)]
        if era.empty:
            continue
        ok = int(era.apply(
            lambda r: all(r[c] in ("Pass", "Minor") for c in CRITERIA),
            axis=1).sum())
        print(f"    {lo_yr}-{hi_yr}: {ok}/{len(era)} ({ok/len(era):.1%})")

    # by section type
    print("\n  Lenient pass rate by section:")
    for sec in ("item_1a", "item_7"):
        sub = df[df["section"] == sec]
        if sub.empty:
            continue
        ok = int(sub.apply(
            lambda r: all(r[c] in ("Pass", "Minor") for c in CRITERIA),
            axis=1).sum())
        lo, hi = wilson_ci(ok, len(sub))
        print(f"    {sec}: {ok}/{len(sub)} ({ok/len(sub):.1%})"
              f"  [{lo:.1%} - {hi:.1%}]")

    # failures
    failed = df[df.apply(
        lambda r: any(r[c] == "Fail" for c in CRITERIA), axis=1)]
    if len(failed):
        print(f"\n  Failures ({len(failed)}):")
        for _, r in failed.iterrows():
            fails = [c for c in CRITERIA if r[c] == "Fail"]
            print(f"    {r['ticker']} FY{int(r['fiscal_year'])} {r['section']}"
                  f": {', '.join(fails)} -- {r.get('notes', '')}")

    print(f"\n  Results: {RESULTS_PATH}")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 1: BUILD SAMPLE
# ══════════════════════════════════════════════════════════════════════════

def build_sample(n_per_section, seed):
    """Sample filings, extract texts from SEC, save to CSV."""

    # ── load panel ────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "annual_panel.csv")
    if "form_type" in df.columns:
        df = df[df["form_type"] != "10-K/A"]
    pool = (df[["ticker", "accession_number", "fiscal_year"]]
            .dropna(subset=["accession_number"])
            .drop_duplicates())

    per_era = max(1, n_per_section // len(_ERA_BINS))

    # ── stratified sample for each section ────────────────────────────────
    rows = []
    for section, sec_seed in [("item_1a", seed), ("item_7", seed + 1000)]:
        parts = []
        for lo, hi in _ERA_BINS:
            era = pool[(pool["fiscal_year"] >= lo) & (pool["fiscal_year"] <= hi)]
            n_draw = min(per_era, len(era))
            if n_draw:
                parts.append(era.sample(n=n_draw, random_state=sec_seed))
        sec_sample = pd.concat(parts, ignore_index=True)
        sec_sample["section"] = section
        rows.append(sec_sample)

    sample = (pd.concat(rows, ignore_index=True)
              .sample(frac=1, random_state=seed)
              .reset_index(drop=True))
    print(f"Sampled {len(sample)} filings ({n_per_section} per section)")

    era_c = {f"{lo}-{hi}": int(((sample["fiscal_year"] >= lo)
                                 & (sample["fiscal_year"] <= hi)).sum())
             for lo, hi in _ERA_BINS}
    print(f"  Eras: {era_c}")

    # ── extract texts from SEC ────────────────────────────────────────────
    cik_map = load_cik_map()
    doc_maps = {}

    # prefetch doc maps
    needed_ciks = set()
    for _, row in sample.iterrows():
        cik = cik_map.get(str(row["ticker"]))
        if cik:
            needed_ciks.add(cik)
    print(f"Prefetching doc indices for {len(needed_ciks)} firms ...")
    for cik in sorted(needed_ciks):
        doc_maps[cik] = _fetch_doc_map_safe(cik)
        time.sleep(REQUEST_SLEEP)

    # extract each section
    texts = []
    word_counts = []
    extract_errors = []
    for i, row in sample.iterrows():
        ticker = str(row["ticker"])
        fy = int(row["fiscal_year"])
        accn = str(row["accession_number"])
        section = str(row["section"])
        cik = cik_map.get(ticker)

        if not cik:
            print(f"  [{i+1}/{len(sample)}] {ticker} FY{fy} {section}: "
                  f"no CIK -- empty")
            texts.append("")
            word_counts.append(0)
            extract_errors.append("no CIK")
            continue

        try:
            if cik not in doc_maps:
                doc_maps[cik] = _fetch_doc_map_safe(cik)
            ex = extract_filing_text(cik, accn, doc_maps.get(cik))
            text = ex.get(section) or ""
            wc = len(text.split()) if text else 0
            texts.append(text)
            word_counts.append(wc)
            extract_errors.append("")
            status = f"{wc:,} words" if text else "EMPTY"
            print(f"  [{i+1}/{len(sample)}] {ticker} FY{fy} {section}: "
                  f"{status}")
        except Exception as e:
            print(f"  [{i+1}/{len(sample)}] {ticker} FY{fy} {section}: "
                  f"ERROR -- {e}")
            texts.append("")
            word_counts.append(0)
            extract_errors.append(str(e))

        time.sleep(REQUEST_SLEEP)

    sample["text"] = texts
    sample["word_count"] = word_counts
    sample["extract_error"] = extract_errors

    # save
    sample.to_csv(SAMPLE_PATH, index=False)

    n_ok = int((sample["word_count"] > 0).sum())
    n_empty = int((sample["word_count"] == 0).sum())
    print(f"\nSample saved: {SAMPLE_PATH}")
    print(f"  {n_ok} with text, {n_empty} empty/error")

    # clear old results
    if RESULTS_PATH.exists():
        RESULTS_PATH.unlink()
        print(f"  Cleared old results: {RESULTS_PATH.name}")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 2: VALIDATE WITH AI
# ══════════════════════════════════════════════════════════════════════════

def validate(api_key, section_filter):
    """Validate pre-extracted texts with Gemini (paid key)."""

    if _OpenAI is None:
        sys.exit("ERROR: pip install openai")

    if not SAMPLE_PATH.exists():
        sys.exit("ERROR: run --build-sample first")

    client = _OpenAI(base_url=GEMINI_ENDPOINT, api_key=api_key)
    print(f"Model: {MODEL}")
    if section_filter:
        print(f"Section filter: {section_filter}")

    # load sample
    sample = pd.read_csv(SAMPLE_PATH)
    print(f"Loaded sample: {len(sample)} total rows")

    # filter to target section(s)
    if section_filter:
        sample = sample[sample["section"] == section_filter].reset_index(
            drop=True)
        print(f"  {len(sample)} rows for {section_filter}")

    # load existing results (checkpoint)
    results = []
    done_keys = set()
    if RESULTS_PATH.exists():
        try:
            prev = pd.read_csv(RESULTS_PATH)
            if not prev.empty:
                results = prev.to_dict("records")
                done_keys = {
                    (str(r["ticker"]), int(r["fiscal_year"]),
                     str(r["section"]))
                    for r in results}
                print(f"  Checkpoint: {len(done_keys)} already done")
        except Exception:
            pass

    # find remaining
    remaining = []
    for _, row in sample.iterrows():
        key = (str(row["ticker"]), int(row["fiscal_year"]),
               str(row["section"]))
        if key not in done_keys:
            remaining.append(row)

    if not remaining:
        print("\nAll sections validated!")
        _print_summary()
        return

    print(f"  Remaining: {len(remaining)} sections\n")

    # ── validation loop ──────────────────────────────────────────────────
    total_calls = 0

    for seq, row in enumerate(remaining, 1):
        ticker = str(row["ticker"])
        fy = int(row["fiscal_year"])
        section = str(row["section"])
        text = str(row["text"]) if pd.notna(row["text"]) else ""
        wc = int(row["word_count"])

        # record empty / extraction-error rows as Fail (no API call)
        if not text or wc == 0:
            err = str(row.get("extract_error", "")) if pd.notna(
                row.get("extract_error")) else ""
            reason = err if err else "empty extraction"
            result = {"ticker": ticker, "fiscal_year": fy,
                      "section": section, "word_count": 0}
            for c in CRITERIA:
                result[c] = "Fail"
            result["notes"] = f"extraction failure: {reason}"
            results.append(result)
            pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
            print(f"  [{seq}] {ticker} FY{fy} {section}=FAIL "
                  f"(extraction: {reason})")
            continue

        # determine label
        if section == "item_1a":
            label, name = "ITEM 1A", "Risk Factors"
        else:
            label, name = "ITEM 7", "MD&A"

        # build prompt
        sblock = (f"=== {label} ({name}) ===\n"
                  f"Word count: {wc:,}\n\n{text}\n")
        rfmt = _response_format(section)
        prompt = PROMPT_TEMPLATE.format(
            ticker=ticker, fy=fy,
            sections_block=sblock,
            response_format=rfmt,
        )

        # call LLM with retry
        success = False
        for attempt in range(1, MAX_RETRIES_PER_CALL + 1):
            try:
                raw = _call_llm(client, prompt)
                total_calls += 1
                success = True
                break
            except Exception as e:
                total_calls += 1
                wait = RETRY_BACKOFF_S * attempt
                print(f"    [{seq}] API error (try "
                      f"{attempt}/{MAX_RETRIES_PER_CALL}): "
                      f"{e} -- sleeping {wait}s")
                time.sleep(wait)

        if not success:
            print(f"  [{seq}] {ticker} FY{fy} {section}: "
                  f"failed all retries -- will retry on next run")
            continue

        # parse response
        parsed = _parse_response(raw)

        # build result row
        result = {"ticker": ticker, "fiscal_year": fy,
                  "section": section, "word_count": wc}
        all_valid = True
        for c in CRITERIA:
            v = parsed.get(f"{section}_{c}", "PARSE_ERR")
            if v not in VALID_SCORES:
                v = "PARSE_ERR"
                all_valid = False
            result[c] = v
        result["notes"] = parsed.get(f"{section}_notes", "")

        results.append(result)
        pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)

        # console status
        if all(result[c] == "Pass" for c in CRITERIA):
            tag = "PASS"
        elif all(result[c] in ("Pass", "Minor") for c in CRITERIA):
            tag = "MINOR"
        elif not all_valid:
            tag = "PARSE_ERR"
        else:
            tag = "FAIL"
        print(f"  [{seq}] {ticker} FY{fy} {section}={tag}"
              f"   ({total_calls} calls)")

        time.sleep(REQUEST_GAP_S)

    print(f"\nDone. Total API calls: {total_calls}")
    _print_summary()


# ── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="AI-based extraction validation (Gemini 2.5 Flash)")
    ap.add_argument("--build-sample", action="store_true",
                    help="Phase 1: sample filings & extract texts from SEC")
    ap.add_argument("--api-key",
                    help="Gemini API key (paid, from aistudio.google.com)")
    ap.add_argument("--section", choices=["item_1a", "item_7"],
                    default=None,
                    help="Which section to validate (default: both)")
    ap.add_argument("--n", type=int, default=100,
                    help="Sections per type to sample (default 100)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Sampling seed (default 42)")
    args = ap.parse_args()

    if args.build_sample:
        build_sample(args.n, args.seed)
    else:
        if not args.api_key:
            sys.exit("ERROR: --api-key required for validation")
        validate(args.api_key, args.section)


if __name__ == "__main__":
    main()
