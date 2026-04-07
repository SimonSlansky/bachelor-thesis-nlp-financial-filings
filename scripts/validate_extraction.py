"""AI-assisted validation of SEC 10-K section extraction quality.

Uses Google Gemini 2.5 Flash as an independent judge to evaluate extraction
quality on five criteria with a three-level rubric (Pass / Minor / Fail).

Design for thesis-quality results:
  - One API call per filing (both Item 1A and Item 7 judged together)
  - Stratified random sampling across fiscal-year eras
  - Checkpoint / resume: stops after --max-calls; re-run with a new key
  - Wilson 95 % confidence intervals for all reported proportions
  - Three-level scoring  →  strict *and* lenient pass rates

Usage:
    # Put one Gemini API key per line in a text file:
    python validate_extraction.py --key-file keys.txt
    # The script rotates keys automatically (20 calls each).
"""

import argparse
import math
import os
import random
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
    from google import genai
    from google.genai import types as genai_types
except Exception:
    genai = None
    genai_types = None

# SSL cert forwarding for corporate proxies / httpx (Gemini SDK)
for _env in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    _val = os.environ.get(_env)
    if _val and os.path.isfile(_val):
        os.environ.setdefault("SSL_CERT_FILE", _val)
        break

# ── constants ─────────────────────────────────────────────────────────────

MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.0                   # deterministic for reproducibility
MAX_API_RETRIES = 3
RETRY_BACKOFF_S = 15
REQUEST_GAP_S = 4                   # seconds between API calls

TRUNC_THRESHOLD = 12_000            # send full text below this word count
TRUNC_HEAD = 3_000                  # first N words when truncating
TRUNC_TAIL = 3_000                  # last N words when truncating

_ERA_BINS = [(2010, 2014), (2015, 2019), (2020, 2024)]

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

Below are sections automatically extracted from this company's 10-K filing.
Evaluate each NON-EMPTY section on five quality criteria.

**Important context**: Financial data tables (revenue breakdowns, balance sheets,
contractual obligations, etc.) are *intentionally removed* during extraction.
Only narrative prose is retained.  Do NOT penalize the extraction for missing
tables or for references to tables/figures that are absent.

{sections_block}
---

**Criteria** (evaluate each for every non-empty section):

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

Respond in **exactly** this format (omit any section marked [EMPTY]):

{response_format}
"""


# ── helpers ───────────────────────────────────────────────────────────────

def _truncate(text):
    """Return (possibly truncated text, was_truncated)."""
    words = text.split()
    if len(words) <= TRUNC_THRESHOLD:
        return text, False
    head = " ".join(words[:TRUNC_HEAD])
    tail = " ".join(words[-TRUNC_TAIL:])
    omitted = len(words) - TRUNC_HEAD - TRUNC_TAIL
    return (
        f"[FIRST {TRUNC_HEAD:,} WORDS]:\n{head}\n\n"
        f"[... {omitted:,} words omitted for brevity ...]\n\n"
        f"[LAST {TRUNC_TAIL:,} WORDS]:\n{tail}"
    ), True


def _section_block(label, name, text, wc):
    """Build the text block for one section in the prompt."""
    if not text:
        return f"=== {label} ({name}) ===\n[EMPTY -- extraction returned no text]\n"
    body, trunc = _truncate(text)
    note = (f"  (showing first {TRUNC_HEAD:,} + last {TRUNC_TAIL:,}"
            f" of {wc:,} total words)") if trunc else ""
    return f"=== {label} ({name}) ===\nWord count: {wc:,}{note}\n\n{body}\n"


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


def _call_gemini(client, prompt):
    last = None
    cfg = (genai_types.GenerateContentConfig(temperature=TEMPERATURE)
           if genai_types else None)
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=prompt, config=cfg,
            )
            return resp.text or ""
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
        lenient = n_p + n_m
        lo, hi = wilson_ci(lenient, n)
        print(f"  {c:<22}{n_p:>3}/{n} {n_p/n:>5.1%}"
              f"  {lenient:>3}/{n} {lenient/n:>5.1%}"
              f"  {n_f:>5}"
              f"   [{lo:.1%} - {hi:.1%}]")

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

CALLS_PER_KEY = 20          # free-tier daily limit per key


def _load_keys(path):
    """Read API keys from a text file (one per line, # comments ok)."""
    keys = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                keys.append(line)
    if not keys:
        sys.exit(f"ERROR: no keys found in {path}")
    return keys


def main():
    ap = argparse.ArgumentParser(
        description="AI-based extraction validation (Gemini 2.5 Flash)")
    key_grp = ap.add_mutually_exclusive_group(required=True)
    key_grp.add_argument("--key-file",
                         help="Path to text file with one Gemini API key "
                              "per line (# comments allowed)")
    key_grp.add_argument("--api-key",
                         help="Single Gemini API key (processes up to "
                              f"{CALLS_PER_KEY} filings then stops)")
    ap.add_argument("--calls-per-key", type=int, default=CALLS_PER_KEY,
                    help=f"API calls per key (default {CALLS_PER_KEY})")
    ap.add_argument("--n", type=int, default=100,
                    help="Total filings to validate (default 100)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Sampling seed (default 42)")
    args = ap.parse_args()

    if genai is None:
        sys.exit("ERROR: pip install google-genai")

    # build key list
    if args.key_file:
        keys = _load_keys(args.key_file)
    else:
        keys = [args.api_key]
    max_calls = len(keys) * args.calls_per_key
    print(f"Loaded {len(keys)} API key(s) -> up to {max_calls} calls")

    key_idx = 0
    key_calls = 0
    client = genai.Client(api_key=keys[key_idx])

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
        random.seed(args.seed)
        per_era = max(1, args.n // len(_ERA_BINS))
        parts = []
        for lo, hi in _ERA_BINS:
            era = pool[(pool["fiscal_year"] >= lo) & (pool["fiscal_year"] <= hi)]
            n_draw = min(per_era, len(era))
            if n_draw:
                parts.append(era.sample(n=n_draw, random_state=args.seed))
        sample = (pd.concat(parts, ignore_index=True)
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
    done = set()
    results = []
    if RESULTS_PATH.exists():
        try:
            prev = pd.read_csv(RESULTS_PATH)
        except Exception:
            prev = pd.DataFrame()
        if not prev.empty:
            results = prev.to_dict("records")
            done = {(str(r["ticker"]), int(r["fiscal_year"])) for r in results
                    if r.get("correct_section") not in _SKIP}
            # deduplicate (ticker, fy, section)
            seen = set()
            deduped = []
            for r in results:
                key = (str(r["ticker"]), int(r["fiscal_year"]),
                       str(r["section"]))
                if key not in seen:
                    seen.add(key)
                    deduped.append(r)
            results = deduped
            print(f"  Checkpoint: {len(done)} filings already validated")

    remaining = [(i, row) for i, row in sample.iterrows()
                 if (str(row["ticker"]), int(row["fiscal_year"])) not in done]

    if not remaining:
        print("\nAll filings validated!")
        _print_summary()
        return

    to_process = min(len(remaining), max_calls)
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
        if calls >= max_calls:
            print(f"\nAll {len(keys)} key(s) exhausted "
                  f"({calls} calls). Add more keys to continue.")
            break

        ticker = str(row["ticker"])
        fy = int(row["fiscal_year"])
        accn = str(row["accession_number"])
        cik = cik_map.get(ticker)
        if not cik:
            print(f"  [{seq}] {ticker} FY{fy}: no CIK -- skipped")
            continue

        # extract
        try:
            ex = extract_filing_text(cik, accn, doc_maps.get(cik))
        except Exception as e:
            print(f"  [{seq}] {ticker} FY{fy}: extraction error -- {e}")
            for s in ("item_1a", "item_7"):
                results.append({"ticker": ticker, "fiscal_year": fy,
                                "section": s, "word_count": 0,
                                **{c: "EXTRACT_ERR" for c in CRITERIA},
                                "notes": str(e)})
            _save(results)
            continue

        t1a = ex.get("item_1a") or ""
        t7  = ex.get("item_7")  or ""
        wc1a = len(t1a.split()) if t1a else 0
        wc7  = len(t7.split())  if t7  else 0

        if not t1a and not t7:
            print(f"  [{seq}] {ticker} FY{fy}: both empty -- no API call")
            for s, wc in [("item_1a", wc1a), ("item_7", wc7)]:
                results.append({"ticker": ticker, "fiscal_year": fy,
                                "section": s, "word_count": wc,
                                **{c: "EMPTY" for c in CRITERIA},
                                "notes": "No text extracted"})
            _save(results)
            continue

        # build prompt (both sections in one call)
        sblock = (_section_block("ITEM 1A", "Risk Factors", t1a, wc1a)
                  + "\n"
                  + _section_block("ITEM 7", "MD&A", t7, wc7))
        rfmt = _response_format(bool(t1a), bool(t7))
        prompt = PROMPT_TEMPLATE.format(
            ticker=ticker, fy=fy,
            sections_block=sblock,
            response_format=rfmt,
        )

        # call Gemini (with automatic key rotation)
        try:
            raw = _call_gemini(client, prompt)
            parsed = _parse_response(raw)
            calls += 1
            key_calls += 1
            # rotate key if this one is exhausted
            if (key_calls >= args.calls_per_key
                    and key_idx + 1 < len(keys)):
                key_idx += 1
                key_calls = 0
                client = genai.Client(api_key=keys[key_idx])
                print(f"    -> rotated to key {key_idx + 1}/{len(keys)}")
        except Exception as e:
            if _is_rate_limited(e) and key_idx + 1 < len(keys):
                key_idx += 1
                key_calls = 0
                client = genai.Client(api_key=keys[key_idx])
                print(f"    rate-limited -> rotated to key "
                      f"{key_idx + 1}/{len(keys)}, retrying ...")
                try:
                    time.sleep(REQUEST_GAP_S)
                    raw = _call_gemini(client, prompt)
                    parsed = _parse_response(raw)
                    calls += 1
                    key_calls += 1
                except Exception as e2:
                    print(f"  [{seq}] {ticker} FY{fy}: "
                          f"API error after rotation -- {e2}")
                    for s, wc in [("item_1a", wc1a), ("item_7", wc7)]:
                        results.append({"ticker": ticker, "fiscal_year": fy,
                                        "section": s, "word_count": wc,
                                        **{c: "API_ERROR" for c in CRITERIA},
                                        "notes": str(e2)})
                    _save(results)
                    continue
            else:
                print(f"  [{seq}] {ticker} FY{fy}: API error -- {e}")
                for s, wc in [("item_1a", wc1a), ("item_7", wc7)]:
                    results.append({"ticker": ticker, "fiscal_year": fy,
                                    "section": s, "word_count": wc,
                                    **{c: "API_ERROR" for c in CRITERIA},
                                    "notes": str(e)})
                _save(results)
                if _is_rate_limited(e):
                    print("  All keys exhausted. Add more to continue.")
                    break
                continue

        # record per-section results
        for s, wc, has in [("item_1a", wc1a, bool(t1a)),
                           ("item_7",  wc7,  bool(t7))]:
            results.append(_section_rows(parsed, s, ticker, fy, wc, has))

        # console status
        tags = []
        for s in ("item_1a", "item_7"):
            scores = [parsed.get(f"{s}_{c}", "?") for c in CRITERIA]
            if all(v == "Pass" for v in scores):
                tags.append(f"{s}=PASS")
            elif all(v in ("Pass", "Minor") for v in scores):
                tags.append(f"{s}=MINOR")
            else:
                tags.append(f"{s}=FAIL")
        print(f"  [{seq}] {ticker} FY{fy}  {' '.join(tags)}"
              f"   ({calls}/{max_calls} calls,"
              f" key {key_idx + 1}/{len(keys)})")

        _save(results)
        if calls < max_calls:
            time.sleep(REQUEST_GAP_S)

    # final
    _save(results)
    print(f"\nThis run: {calls} API calls")
    _print_summary()


if __name__ == "__main__":
    main()
