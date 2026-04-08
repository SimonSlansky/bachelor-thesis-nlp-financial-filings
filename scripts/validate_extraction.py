"""AI-assisted validation of SEC 10-K section extraction quality.

Uses Gemini 2.5 Flash (via OpenAI-compatible endpoint) as an independent
judge to evaluate extraction quality on five criteria with a three-level
rubric (Pass / Minor / Fail).

Design for thesis-quality results:
  - One API call per filing (one section judged per call)
  - Full text sent to model (no truncation -- 1M token context)
  - Stratified random sampling across fiscal-year eras
  - Checkpoint / resume: stops after --max-calls; re-run to continue
  - Wilson 95 % confidence intervals for all reported proportions
  - Three-level scoring  →  strict *and* lenient pass rates

Usage:
    # All 100 Item 1A sections with key #1:
    python validate_extraction.py --section item_1a --api-key KEY_1

    # All 100 Item 7 sections with key #2:
    python validate_extraction.py --section item_7 --api-key KEY_2
"""

import argparse
import math
import shutil
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

TEMPERATURE = 0.0                   # deterministic for reproducibility
MAX_API_RETRIES = 3
RETRY_BACKOFF_S = 15
REQUEST_GAP_S = 4                 # seconds between API calls

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

def _section_block(label, name, text, wc):
    """Build the text block for one section in the prompt."""
    if not text:
        return f"=== {label} ({name}) ===\n[EMPTY -- extraction returned no text]\n"
    return f"=== {label} ({name}) ===\nWord count: {wc:,}\n\n{text}\n"


def _response_format(has_1a, has_7):
    """Build the expected response format string."""
    lines = []
    for key, present in [("item_1a", has_1a), ("item_7", has_7)]:
        if present:
            for c in CRITERIA:
                lines.append(f"{key}_{c}: Pass/Minor/Fail")
            lines.append(f"{key}_notes: <one-line explanation or 'OK'>")
    return "\n".join(lines)


# ── response parsing ─────────────────────────────────────────────────────

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


def _section_rows(parsed, prefix, ticker, fy, wc, has_text):
    """Build a result dict for one section."""
    row = {"ticker": ticker, "fiscal_year": fy, "section": prefix,
           "word_count": wc}
    if not has_text:
        for c in CRITERIA:
            row[c] = "EMPTY"
        row["notes"] = "No text extracted"
        return row
    for c in CRITERIA:
        v = parsed.get(f"{prefix}_{c}", "PARSE_ERR")
        row[c] = v if v in VALID_SCORES else "PARSE_ERR"
    row["notes"] = parsed.get(f"{prefix}_notes", "")
    return row


# ── API ───────────────────────────────────────────────────────────────────

def _is_rate_limited(exc):
    msg = str(exc).upper()
    return ("429" in msg or "RESOURCE_EXHAUSTED" in msg
            or ("RATE" in msg and "LIMIT" in msg))


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
    """Call Gemini via OpenAI-compatible endpoint."""
    last = None
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last = e
            if _is_rate_limited(e) and attempt < MAX_API_RETRIES:
                wait = RETRY_BACKOFF_S * attempt
                print(f"    rate-limited (try {attempt}/{MAX_API_RETRIES}), "
                      f"sleeping {wait}s ...")
                time.sleep(wait)
                continue
            raise
    raise last


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


# ── checkpoint / summary ─────────────────────────────────────────────────

_SKIP = frozenset({"EMPTY", "EXTRACT_ERR", "API_ERROR", "PARSE_ERR"})


def _save(results):
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)


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
    judged = df[~df["correct_section"].isin(_SKIP)].copy()

    n_filings = judged[["ticker", "fiscal_year"]].drop_duplicates().shape[0]
    print(f"\n{'=' * 70}")
    print(f"VALIDATION SUMMARY   {len(judged)} sections  |  {n_filings} filings")
    print(f"{'=' * 70}")

    if judged.empty:
        print("  (no sections judged yet)")
        return

    print(f"\n  {'Criterion':<22}{'Strict':>10}{'Lenient':>10}"
          f"{'Fail':>8}  {'95% CI (lenient)'}")
    print(f"  {'-' * 66}")
    for c in CRITERIA:
        n = len(judged)
        n_p = int((judged[c] == "Pass").sum())
        n_m = int((judged[c] == "Minor").sum())
        n_f = int((judged[c] == "Fail").sum())
        n_err = n - n_p - n_m - n_f
        lenient = n_p + n_m
        lo, hi = wilson_ci(lenient, n)
        err_note = f"  ({n_err} parse-err)" if n_err else ""
        print(f"  {c:<22}{n_p:>3}/{n} {n_p/n:>5.1%}"
              f"  {lenient:>3}/{n} {lenient/n:>5.1%}"
              f"  {n_f:>5}"
              f"   [{lo:.1%} - {hi:.1%}]{err_note}")

    # overall
    strict_all = int(judged.apply(
        lambda r: all(r[c] == "Pass" for c in CRITERIA), axis=1).sum())
    lenient_all = int(judged.apply(
        lambda r: all(r[c] in ("Pass", "Minor") for c in CRITERIA),
        axis=1).sum())
    n = len(judged)
    lo_s, hi_s = wilson_ci(strict_all, n)
    lo_l, hi_l = wilson_ci(lenient_all, n)
    print(f"\n  {'Overall strict':<22}{strict_all:>3}/{n} ({strict_all/n:.1%})"
          f"                 [{lo_s:.1%} - {hi_s:.1%}]")
    print(f"  {'Overall lenient':<22}{lenient_all:>3}/{n} ({lenient_all/n:.1%})"
          f"                 [{lo_l:.1%} - {hi_l:.1%}]")

    # by era
    print(f"\n  Lenient pass rate by era:")
    for lo_yr, hi_yr in _ERA_BINS:
        era = judged[(judged["fiscal_year"] >= lo_yr)
                     & (judged["fiscal_year"] <= hi_yr)]
        if era.empty:
            continue
        ok = int(era.apply(
            lambda r: all(r[c] in ("Pass", "Minor") for c in CRITERIA),
            axis=1).sum())
        print(f"    {lo_yr}-{hi_yr}: {ok}/{len(era)} ({ok/len(era):.1%})")

    # by section type
    print(f"\n  Lenient pass rate by section:")
    for sec in ("item_1a", "item_7"):
        sub = judged[judged["section"] == sec]
        if sub.empty:
            continue
        ok = int(sub.apply(
            lambda r: all(r[c] in ("Pass", "Minor") for c in CRITERIA),
            axis=1).sum())
        lo, hi = wilson_ci(ok, len(sub))
        print(f"    {sec}: {ok}/{len(sub)} ({ok/len(sub):.1%})"
              f"  [{lo:.1%} - {hi:.1%}]")

    # failures
    failed = judged[judged.apply(
        lambda r: any(r[c] == "Fail" for c in CRITERIA), axis=1)]
    if len(failed):
        print(f"\n  Failures ({len(failed)}):")
        for _, r in failed.iterrows():
            fails = [c for c in CRITERIA if r[c] == "Fail"]
            print(f"    {r['ticker']} FY{int(r['fiscal_year'])} {r['section']}"
                  f": {', '.join(fails)} -- {r.get('notes', '')}")

    # excluded
    empties = df[df["correct_section"] == "EMPTY"]
    errors = df[df["correct_section"].isin({"EXTRACT_ERR", "API_ERROR"})]
    if len(empties) or len(errors):
        print(f"\n  Excluded from scoring:")
        if len(empties):
            print(f"    Empty extractions: {len(empties)}")
        if len(errors):
            print(f"    Errors (extract/API): {len(errors)}")

    print(f"\n  Results: {RESULTS_PATH}")


# ── main ──────────────────────────────────────────────────────────────────

MAX_CALLS_DEFAULT = 200     # Gemini free tier: 500 RPD


def main():
    ap = argparse.ArgumentParser(
        description="AI-based extraction validation (Gemini 2.5 Flash)")
    ap.add_argument("--api-key", required=True,
                    help="Gemini API key (from aistudio.google.com)")
    ap.add_argument("--section", choices=["item_1a", "item_7"],
                    default=None,
                    help="Validate only this section (default: both)")
    ap.add_argument("--max-calls", type=int, default=MAX_CALLS_DEFAULT,
                    help=f"Max API calls this run (default {MAX_CALLS_DEFAULT})")
    ap.add_argument("--n", type=int, default=100,
                    help="Total filings to validate (default 100)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Sampling seed (default 42)")
    args = ap.parse_args()

    if _OpenAI is None:
        sys.exit("ERROR: pip install openai")

    client = _OpenAI(base_url=GEMINI_ENDPOINT, api_key=args.api_key)
    section_filter = args.section
    print(f"Model: {MODEL}"
          + (f"  Section: {section_filter}" if section_filter else ""))

    # ── load panel ────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "annual_panel.csv")
    if "form_type" in df.columns:
        df = df[df["form_type"] != "10-K/A"]
    pool = (df[["ticker", "accession_number", "fiscal_year"]]
            .dropna(subset=["accession_number"])
            .drop_duplicates())

    # ── create or load sample ─────────────────────────────────────────────
    fresh = not SAMPLE_PATH.exists()
    if fresh:
        per_era = max(1, args.n // len(_ERA_BINS))

        # Sample Item 1A independently
        parts_1a = []
        for lo, hi in _ERA_BINS:
            era = pool[(pool["fiscal_year"] >= lo) & (pool["fiscal_year"] <= hi)]
            n_draw = min(per_era, len(era))
            if n_draw:
                parts_1a.append(era.sample(n=n_draw, random_state=args.seed))
        sample_1a = pd.concat(parts_1a, ignore_index=True).reset_index(drop=True)
        sample_1a["section"] = "item_1a"

        # Sample Item 7 independently (different firms)
        parts_7 = []
        for lo, hi in _ERA_BINS:
            era = pool[(pool["fiscal_year"] >= lo) & (pool["fiscal_year"] <= hi)]
            n_draw = min(per_era, len(era))
            if n_draw:
                parts_7.append(era.sample(n=n_draw, random_state=args.seed + 1000))
        sample_7 = pd.concat(parts_7, ignore_index=True).reset_index(drop=True)
        sample_7["section"] = "item_7"

        # Combine both samples
        sample = (pd.concat([sample_1a, sample_7], ignore_index=True)
                  .sample(frac=1, random_state=args.seed)
                  .reset_index(drop=True))
        sample.to_csv(SAMPLE_PATH, index=False)
        # archive any stale results from a previous run
        if RESULTS_PATH.exists():
            bak = RESULTS_PATH.with_suffix(".csv.bak")
            shutil.copy2(RESULTS_PATH, bak)
            RESULTS_PATH.unlink()
            print(f"Backed up old results -> {bak.name}")
        print(f"Created sample: {len(sample)} filings -> {SAMPLE_PATH.name}")
    else:
        sample = pd.read_csv(SAMPLE_PATH)
        print(f"Loaded sample: {len(sample)} filings from {SAMPLE_PATH.name}")

    era_c = {f"{lo}-{hi}": int(((sample["fiscal_year"] >= lo)
                                 & (sample["fiscal_year"] <= hi)).sum())
             for lo, hi in _ERA_BINS}
    print(f"  Eras: {era_c}")

    # ── checkpoint ────────────────────────────────────────────────────────
    results = []
    done_sections = set()
    if RESULTS_PATH.exists():
        try:
            prev = pd.read_csv(RESULTS_PATH)
        except Exception:
            prev = pd.DataFrame()
        if not prev.empty:
            # deduplicate (ticker, fy, section)
            seen = set()
            for r in prev.to_dict("records"):
                key = (str(r["ticker"]), int(r["fiscal_year"]),
                       str(r["section"]))
                if key not in seen:
                    seen.add(key)
                    results.append(r)
            done_sections = seen
            print(f"  Checkpoint: {len(done_sections)} sections already processed")

    remaining = [(i, row) for i, row in sample.iterrows()
                 if (str(row["ticker"]), int(row["fiscal_year"]), str(row["section"])) not in done_sections
                 and (section_filter is None or str(row["section"]) == section_filter)]

    if not remaining:
        print("\nAll sections validated!")
        _print_summary()
        return

    to_process = min(len(remaining), args.max_calls)
    print(f"  Remaining: {len(remaining)} filings"
          f"  (will process up to {to_process} this run)\n")

    # ── CIK map + doc maps ───────────────────────────────────────────────
    cik_map = load_cik_map()
    doc_maps = {}
    batch = remaining[:to_process]
    needed = {cik_map[str(row["ticker"])]
              for _, row in batch if str(row["ticker"]) in cik_map}
    print(f"Prefetching doc indices for {len(needed)} firms ...")
    for cik in sorted(needed):
        doc_maps[cik] = _build_primary_doc_map(cik)
        time.sleep(REQUEST_SLEEP)
    print()

    # ── validation loop ──────────────────────────────────────────────────
    calls = 0
    for seq, (_, row) in enumerate(remaining, 1):
        if calls >= args.max_calls:
            print(f"\nReached {args.max_calls}-call limit. "
                  f"Re-run with a new --api-key to continue.")
            break

        ticker = str(row["ticker"])
        fy = int(row["fiscal_year"])
        accn = str(row["accession_number"])
        target_section = str(row["section"])  # item_1a or item_7
        cik = cik_map.get(ticker)
        if not cik:
            print(f"  [{seq}] {ticker} FY{fy} {target_section}: no CIK -- skipped")
            continue

        # extract
        try:
            # fetch doc map if not already cached
            if cik not in doc_maps:
                doc_maps[cik] = _fetch_doc_map_safe(cik)
            ex = extract_filing_text(cik, accn, doc_maps.get(cik))
        except Exception as e:
            print(f"  [{seq}] {ticker} FY{fy} {target_section}: extraction error -- {e}")
            results.append({"ticker": ticker, "fiscal_year": fy,
                            "section": target_section, "word_count": 0,
                            **{c: "EXTRACT_ERR" for c in CRITERIA},
                            "notes": str(e)})
            _save(results)
            continue

        # Extract only the target section
        if target_section == "item_1a":
            text = ex.get("item_1a") or ""
            label = "ITEM 1A"
            name = "Risk Factors"
        else:  # item_7
            text = ex.get("item_7") or ""
            label = "ITEM 7"
            name = "MD&A"

        wc = len(text.split()) if text else 0

        if not text:
            print(f"  [{seq}] {ticker} FY{fy} {target_section}: empty -- no API call")
            results.append({"ticker": ticker, "fiscal_year": fy,
                            "section": target_section, "word_count": wc,
                            **{c: "EMPTY" for c in CRITERIA},
                            "notes": "No text extracted"})
            _save(results)
            continue

        # build prompt (single section only)
        sblock = _section_block(label, name, text, wc)
        rfmt = _response_format(target_section == "item_1a", target_section == "item_7")

        prompt = PROMPT_TEMPLATE.format(
            ticker=ticker, fy=fy,
            sections_block=sblock,
            response_format=rfmt,
        )

        # call LLM
        try:
            raw = _call_llm(client, prompt)
            parsed = _parse_response(raw)
            calls += 1
        except Exception as e:
            print(f"  [{seq}] {ticker} FY{fy}: API error -- {e}")
            results.append({"ticker": ticker, "fiscal_year": fy,
                            "section": target_section, "word_count": wc,
                            **{c: "API_ERROR" for c in CRITERIA},
                            "notes": str(e)})
            _save(results)
            if _is_rate_limited(e):
                print("  Rate-limit hit -- stopping. "
                      "Re-run with a new --api-key.")
                break
            continue

        # record single-section results
        results.append(_section_rows(parsed, target_section, ticker, fy, wc, True))

        # console status
        scores = [parsed.get(f"{target_section}_{c}", "?") for c in CRITERIA]
        if all(v == "Pass" for v in scores):
            tag = "PASS"
        elif all(v in ("Pass", "Minor") for v in scores):
            tag = "MINOR"
        else:
            tag = "FAIL"
        print(f"  [{seq}] {ticker} FY{fy} {target_section}={tag}"
              f"   ({calls}/{args.max_calls} calls)")

        _save(results)
        if calls < args.max_calls:
            time.sleep(REQUEST_GAP_S)

    # final (deduplicate before saving)
    seen = set()
    deduped_final = []
    for r in results:
        key = (str(r.get("ticker")), int(r.get("fiscal_year")),
               str(r.get("section")))
        if key not in seen:
            seen.add(key)
            deduped_final.append(r)

    pd.DataFrame(deduped_final).to_csv(RESULTS_PATH, index=False)
    print(f"\nThis run: {calls} API calls")
    _print_summary()


if __name__ == "__main__":
    main()
