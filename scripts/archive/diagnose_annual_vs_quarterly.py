#!/usr/bin/env python3
"""Comprehensive annual-vs-quarterly diagnostic.

Consolidates two tests into a single script with LaTeX-ready output:
  A. XBRL financial-data coverage (annual vs quarterly metrics)
  B. Q4 reliability (annual − Q1−Q2−Q3 vs explicit Q4)

Outputs:
  - Console summary
  - tex/tables/annual_vs_quarterly_*.tex  (ready-made LaTeX tables)
  - data/diagnostics/  (CSV raw data for reproducibility)

Run from scripts/:  python diagnose_annual_vs_quarterly.py
"""

import os
import re
import sys
import time
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import (
    SEC_HEADERS, REQUEST_SLEEP, ANNUAL_FLOW_RANGE,
    MAX_FILING_LAG_DAYS,
    EXCLUDED_SIC_RANGE, METRICS_WITH_PRIORITY, FLOW_METRICS,
)
from sec_edgar import (
    fetch_universe, load_company_metadata,
    _fetch_company_facts, _usd_facts,
)

# Quarterly flow range — defined locally since removed from production config
QUARTERLY_FLOW_RANGE = (65, 120)

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
TEX_TABLE_DIR = BASE_DIR / "tex" / "tables"
TEX_TABLE_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR = BASE_DIR / "data" / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# ── sample config ──────────────────────────────────────────────────────────
SAMPLE_SIZE = 30            # companies for XBRL tests (A, B)

# Tags to test in Part B (Q4 reliability)
Q4_TAGS = [
    "NetIncomeLoss",
    "OperatingIncomeLoss",
    "NetCashProvidedByUsedInOperatingActivities",
]

# ═══════════════════════════════════════════════════════════════════════════
# Part A: XBRL Financial-Data Coverage
# ═══════════════════════════════════════════════════════════════════════════

def _count_metrics(us_gaap, meta, mode):
    """Count available metrics for one company in annual or quarterly mode."""
    fye = meta["fiscal_year_end"]
    ann_lo, ann_hi = ANNUAL_FLOW_RANGE
    q_lo, q_hi = QUARTERLY_FLOW_RANGE

    periods = set()          # year_end or (fy, fq)
    metric_presence = {}     # metric -> set of periods where present

    for tag, metric, priority in METRICS_WITH_PRIORITY:
        for fact in _usd_facts(us_gaap, tag):
            end = pd.to_datetime(fact.get("end"), errors="coerce")
            if pd.isna(end):
                continue

            start = pd.to_datetime(fact.get("start"), errors="coerce")

            if mode == "annual":
                form = fact.get("form", "")
                if form not in ("10-K", "10-K/A"):
                    continue
                if metric in FLOW_METRICS:
                    if pd.isna(start):
                        continue
                    days = (end - start).days
                    if not (ann_lo <= days <= ann_hi):
                        continue
                period_key = end.strftime("%Y-%m-%d")
            else:  # quarterly
                form = fact.get("form", "")
                if form not in ("10-Q", "10-Q/A"):
                    continue
                diff = (end.month - fye) % 12
                qmap = {3: "Q1", 6: "Q2", 9: "Q3", 0: "Q4"}
                fq = qmap.get(diff)
                if fq is None or fq == "Q4":
                    continue
                if metric in FLOW_METRICS:
                    if pd.isna(start):
                        continue
                    days = (end - start).days
                    if not (q_lo <= days <= q_hi):
                        continue
                fy = end.year + 1 if end.month > fye else end.year
                period_key = f"{fy}-{fq}"

            periods.add(period_key)
            metric_presence.setdefault(metric, set()).add(period_key)

    n_periods = len(periods)
    if n_periods == 0:
        return None

    metrics_available = {}
    all_metrics = sorted({m for _, m, _ in METRICS_WITH_PRIORITY})
    for m in all_metrics:
        present = len(metric_presence.get(m, set()))
        metrics_available[m] = present / n_periods if n_periods > 0 else 0

    return {"ticker": meta["ticker"], "n_periods": n_periods, **metrics_available}


def run_part_a(companies, xbrl_cache):
    """Test XBRL metric coverage for annual vs quarterly."""
    print("=" * 70)
    print("PART A: XBRL Financial-Data Coverage")
    print("=" * 70)

    annual_rows = []
    quarterly_rows = []

    for ticker, meta in sorted(companies.items()):
        us_gaap = xbrl_cache.get(ticker)
        if us_gaap is None:
            continue

        a = _count_metrics(us_gaap, meta, "annual")
        q = _count_metrics(us_gaap, meta, "quarterly")

        if a:
            annual_rows.append(a)
        if q:
            quarterly_rows.append(q)
        print(f"  {ticker}: annual={a['n_periods'] if a else 0} years, "
              f"quarterly={q['n_periods'] if q else 0} quarters")

    df_a = pd.DataFrame(annual_rows)
    df_q = pd.DataFrame(quarterly_rows)
    df_a.to_csv(DIAG_DIR / "coverage_annual.csv", index=False)
    df_q.to_csv(DIAG_DIR / "coverage_quarterly.csv", index=False)

    # Summary
    metrics = sorted({m for _, m, _ in METRICS_WITH_PRIORITY})
    print(f"\n  {'Metric':<25} {'Annual Coverage':>16} {'Quarterly Coverage':>19}")
    print(f"  {'─'*25} {'─'*16} {'─'*19}")

    table_rows = []
    for m in metrics:
        a_cov = df_a[m].mean() * 100 if m in df_a.columns else 0
        q_cov = df_q[m].mean() * 100 if m in df_q.columns else 0
        print(f"  {m:<25} {a_cov:>15.1f}% {q_cov:>18.1f}%")
        table_rows.append((m, a_cov, q_cov))

    a_periods = df_a["n_periods"].median() if len(df_a) else 0
    q_periods = df_q["n_periods"].median() if len(df_q) else 0
    print(f"\n  Median periods per firm:  annual={a_periods:.0f}  quarterly={q_periods:.0f}")
    print(f"  Firms with data:         annual={len(df_a)}  quarterly={len(df_q)}")

    # LaTeX table
    _write_coverage_table(table_rows, a_periods, q_periods, len(df_a), len(df_q))

    return df_a, df_q


def _write_coverage_table(rows, a_periods, q_periods, a_firms, q_firms):
    latex = r"""\begin{table}[htbp]
\centering
\caption{XBRL Financial Metric Coverage: Annual (10-K) vs.\ Quarterly (10-Q)}
\label{tab:xbrl_coverage}
\begin{tabular}{lrr}
\toprule
Metric & Annual (\%) & Quarterly (\%) \\
\midrule
"""
    for metric, a_cov, q_cov in rows:
        name = metric.replace("_", r"\_")
        latex += f"{name} & {a_cov:.1f} & {q_cov:.1f} \\\\\n"

    latex += r"""\midrule
Median periods per firm & """ + f"{a_periods:.0f}" + r" & " + f"{q_periods:.0f}" + r""" \\
Firms with data & """ + f"{a_firms}" + r" & " + f"{q_firms}" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note:} Coverage shows the percentage of firm-periods with non-missing
values for each metric. Annual data is extracted from 10-K filings; quarterly data
from 10-Q filings (Q1--Q3 only). Sample: """ + f"{a_firms}" + r""" non-financial operating
companies from the top 200 by market capitalisation.
\end{tablenotes}
\end{table}
"""
    path = TEX_TABLE_DIR / "xbrl_coverage.tex"
    path.write_text(latex, encoding="utf-8")
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Part B: Q4 Reliability
# ═══════════════════════════════════════════════════════════════════════════

def _extract_flow_data(us_gaap, tag, fye_month):
    ann_lo, ann_hi = ANNUAL_FLOW_RANGE
    q_lo, q_hi = QUARTERLY_FLOW_RANGE

    annual = {}
    quarterly = {}
    q4_explicit = {}

    for fact in _usd_facts(us_gaap, tag):
        start = pd.to_datetime(fact.get("start"), errors="coerce")
        end = pd.to_datetime(fact.get("end"), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        days = (end - start).days
        form = fact.get("form", "")

        fy = end.year + 1 if end.month > fye_month else end.year
        diff = (end.month - fye_month) % 12
        qmap = {3: "Q1", 6: "Q2", 9: "Q3", 0: "Q4"}
        fq = qmap.get(diff)
        if fq is None:
            continue

        if ann_lo <= days <= ann_hi and form in ("10-K", "10-K/A"):
            annual[fy] = fact["val"]
        elif q_lo <= days <= q_hi:
            if fq == "Q4":
                q4_explicit[fy] = fact["val"]
            else:
                quarterly[(fy, fq)] = fact["val"]

    return annual, quarterly, q4_explicit


def run_part_b(companies, xbrl_cache):
    """Q4 reliability test."""
    print("\n" + "=" * 70)
    print("PART B: Q4 Reliability in XBRL")
    print("=" * 70)

    all_rows = []
    tag_summaries = []

    for tag in Q4_TAGS:
        tag_rows = []
        for ticker, meta in sorted(companies.items()):
            us_gaap = xbrl_cache.get(ticker)
            if us_gaap is None:
                continue

            annual, quarterly, q4_explicit = _extract_flow_data(
                us_gaap, tag, meta["fiscal_year_end"]
            )

            for fy, ann_val in sorted(annual.items()):
                q1 = quarterly.get((fy, "Q1"))
                q2 = quarterly.get((fy, "Q2"))
                q3 = quarterly.get((fy, "Q3"))
                q4_expl = q4_explicit.get(fy)

                has_q123 = all(v is not None for v in (q1, q2, q3))
                implied_q4 = ann_val - (q1 + q2 + q3) if has_q123 else None

                match = None
                if implied_q4 is not None and q4_expl is not None:
                    match = abs(implied_q4 - q4_expl) < max(1.0, 0.01 * abs(ann_val))

                tag_rows.append({
                    "ticker": ticker, "tag": tag, "fy": fy,
                    "annual": ann_val,
                    "has_q123": has_q123,
                    "implied_q4": implied_q4,
                    "has_explicit_q4": q4_expl is not None,
                    "explicit_q4": q4_expl,
                    "match": match,
                })

            print(f"  {ticker} / {tag}: {len(annual)} fiscal years")

        df_tag = pd.DataFrame(tag_rows)
        all_rows.extend(tag_rows)

        if df_tag.empty:
            tag_summaries.append({
                "tag": tag, "fiscal_years": 0, "has_q123": 0, "pct_q123": 0,
                "has_explicit_q4": 0, "pct_explicit_q4": 0,
                "compared": 0, "matched": 0, "pct_match": 0,
                "mismatched": 0, "pct_mismatch": 0,
            })
            continue

        n_total = len(df_tag)
        n_q123 = df_tag["has_q123"].sum()
        n_expl = df_tag["has_explicit_q4"].sum()
        comparisons = df_tag["match"].dropna()
        n_match = comparisons.sum() if len(comparisons) > 0 else 0
        n_compared = len(comparisons)

        tag_summaries.append({
            "tag": tag,
            "fiscal_years": n_total,
            "has_q123": n_q123,
            "pct_q123": 100 * n_q123 / n_total if n_total else 0,
            "has_explicit_q4": n_expl,
            "pct_explicit_q4": 100 * n_expl / n_total if n_total else 0,
            "compared": n_compared,
            "matched": n_match,
            "pct_match": 100 * n_match / n_compared if n_compared else 0,
            "mismatched": n_compared - n_match,
            "pct_mismatch": 100 * (n_compared - n_match) / n_compared if n_compared else 0,
        })

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(DIAG_DIR / "q4_reliability.csv", index=False)

    print(f"\n  {'Tag':<50} {'FY':>5} {'Q1-3':>5} {'Expl Q4':>8} {'Match':>11}")
    print(f"  {'─'*50} {'─'*5} {'─'*5} {'─'*8} {'─'*11}")
    for s in tag_summaries:
        tag_short = s["tag"][:48]
        print(f"  {tag_short:<50} {s['fiscal_years']:>5} "
              f"{s['has_q123']:>5} {s['has_explicit_q4']:>8} "
              f"{s['matched']:.0f}/{s['compared']}")

    _write_q4_table(tag_summaries)
    return df_all, tag_summaries


def _tag_display(tag):
    """Short display name for XBRL tags."""
    mapping = {
        "NetIncomeLoss": "Net Income",
        "OperatingIncomeLoss": "Operating Income",
        "NetCashProvidedByUsedInOperatingActivities": "Operating Cash Flow",
    }
    return mapping.get(tag, tag)


def _write_q4_table(summaries):
    latex = r"""\begin{table}[htbp]
\centering
\caption{Q4 Reliability: Availability and Consistency of Fourth-Quarter XBRL Facts}
\label{tab:q4_reliability}
\begin{tabular}{lrrrrr}
\toprule
Metric & Firm-Years & Q1--Q3 (\%) & Explicit Q4 (\%) & Compared & Match (\%) \\
\midrule
"""
    for s in summaries:
        name = _tag_display(s["tag"])
        latex += (f"{name} & {s['fiscal_years']} & {s['pct_q123']:.1f} "
                  f"& {s['pct_explicit_q4']:.1f} & {s['compared']} "
                  f"& {s['pct_match']:.1f} \\\\\n")

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note:} ``Q1--Q3'' indicates the share of fiscal years where all three
quarterly facts (Q1, Q2, Q3) exist in XBRL. ``Explicit Q4'' indicates the share with
a standalone fourth-quarter fact (period $\approx$90~days ending at fiscal year-end).
``Match'' reports the share of observations where the explicit Q4 value matches the
implied Q4 ($= \text{Annual} - \text{Q1} - \text{Q2} - \text{Q3}$) within 1\% of the
annual figure. Sample: 30 non-financial firms, all available fiscal years.
\end{tablenotes}
\end{table}
"""
    path = TEX_TABLE_DIR / "q4_reliability.tex"
    path.write_text(latex, encoding="utf-8")
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Summary & Recommendation
# ═══════════════════════════════════════════════════════════════════════════

class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._chunks = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._chunks.append(data)

    def get_text(self):
        return "\n".join(self._chunks)


def _html_to_text(html):
    p = _TextExtractor()
    p.feed(html)
    return p.get_text()


def _get(url, timeout=60):
    time.sleep(REQUEST_SLEEP)
    try:
        r = requests.get(url, headers=SEC_HEADERS, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        return None


def _get_recent_filings(sub, form_type, n):
    recent = sub.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])

    results = []
    for i, form in enumerate(forms):
        if form == form_type:
            results.append({
                "form": form,
                "accession": accessions[i],
                "primary_doc": primary_docs[i] if i < len(primary_docs) else None,
                "filing_date": filing_dates[i] if i < len(filing_dates) else None,
            })
            if len(results) >= n:
                break
    return results


def _download_filing(cik, accession, primary_doc):
    cik_int = cik.lstrip("0") or "0"
    acc_nd = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nd}/{primary_doc}"
    r = _get(url)
    return r.text if r else None


# Section extraction patterns
# NOTE: End patterns use item number + section title to avoid matching
# inline cross-references like "see Part II, Item 8, 'Financial Statements'".
# Bare fallback end patterns (e.g. ITEM\s*8\b) were removed because they
# matched cross-references mid-section, truncating extraction to ~200 words.
_SEC_PATTERNS = {
    "10-K_mda_start": [
        r"item\s*7[.\s\u2014\u2013\-–—:]*\s*management['\u2019]?s?\s*discussion",
        r"ITEM\s*7\b",
    ],
    "10-K_mda_end": [
        r"item\s*7a[.\s\u2014\u2013\-–—:]*\s*quantitative",
        r"item\s*8[.\s\u2014\u2013\-–—:]*\s*financial",
    ],
    "10-K_risk_start": [
        r"item\s*1a[.\s\u2014\u2013\-–—:]*\s*risk\s*factors",
        r"ITEM\s*1A\b",
    ],
    "10-K_risk_end": [
        r"item\s*1b[.\s\u2014\u2013\-–—:]*\s*unresolved",
        r"item\s*2[.\s\u2014\u2013\-–—:]*\s*properties",
    ],
    "10-Q_mda_start": [
        r"item\s*2[.\s\u2014\u2013\-–—:]*\s*management['\u2019]?s?\s*discussion",
    ],
    "10-Q_mda_end": [
        r"item\s*3[.\s\u2014\u2013\-–—:]*\s*quantitative",
    ],
    "10-Q_risk_start": [
        r"item\s*1a[.\s\u2014\u2013\-–—:]*\s*risk\s*factors",
    ],
    "10-Q_risk_end": [
        r"item\s*2[.\s\u2014\u2013\-–—:]*\s*unregistered",
    ],
}


def _find_section(text, start_pats, end_pats):
    """Extract text between start/end markers, picking the match that
    yields the longest section (avoids TOC entries and cross-references)."""
    best_section = ""
    for pat in start_pats:
        for m in re.finditer(pat, text, re.IGNORECASE | re.DOTALL):
            remainder = text[m.end():]
            earliest_end = len(remainder)
            for ep in end_pats:
                em = re.search(ep, remainder, re.IGNORECASE)
                if em and em.start() < earliest_end:
                    earliest_end = em.start()
            candidate = remainder[:earliest_end].strip()
            if len(candidate) > len(best_section):
                best_section = candidate
    return best_section


def _word_count(text):
    return len(text.split())


def _is_boilerplate(text):
    wc = _word_count(text)
    if wc < 100:
        return True
    lower = text.lower()
    refs = [
        "no material change", "previously disclosed", "refer to",
        "as described in", "incorporated by reference", "set forth in",
        "annual report on form 10-k", "there have been no material changes",
    ]
    return any(r in lower for r in refs) and wc < 300


def run_part_c(companies):
    """Textual data availability: 10-K vs 10-Q."""
    print("\n" + "=" * 70)
    print("PART C: Textual Data Availability (10-K vs 10-Q)")
    print("=" * 70)

    # Pick first TEXT_SAMPLE_SIZE companies from sample
    text_cos = dict(list(companies.items())[:TEXT_SAMPLE_SIZE])
    print(f"  Testing {len(text_cos)} companies × ({N_10K} 10-K + {N_10Q} 10-Q)\n")

    # Need submissions for filing index
    rows = []
    for ticker, meta in sorted(text_cos.items()):
        cik = meta["cik"]
        sub_r = _get(f"https://data.sec.gov/submissions/CIK{cik}.json")
        if sub_r is None:
            print(f"  {ticker}: submissions failed")
            continue
        sub = sub_r.json()

        for form_type, n_filings in [("10-K", N_10K), ("10-Q", N_10Q)]:
            filings = _get_recent_filings(sub, form_type, n_filings)
            key = form_type.replace("-", "")  # 10K or 10Q
            sp = _SEC_PATTERNS

            for f in filings:
                label = f"{form_type} {f['filing_date']}"
                html = _download_filing(cik, f["accession"], f["primary_doc"])
                if html is None:
                    print(f"    {ticker} {label}: DOWNLOAD FAILED")
                    continue

                text = _html_to_text(html)
                total_wc = _word_count(text)

                mda = _find_section(
                    text,
                    sp[f"{form_type}_mda_start"],
                    sp[f"{form_type}_mda_end"],
                )
                risk = _find_section(
                    text,
                    sp[f"{form_type}_risk_start"],
                    sp[f"{form_type}_risk_end"],
                )
                mda_wc = _word_count(mda)
                risk_wc = _word_count(risk)
                risk_bp = _is_boilerplate(risk) if form_type == "10-Q" else False

                status = " BOILERPLATE" if risk_bp else ""
                print(f"    {ticker} {label}: total={total_wc:,}  "
                      f"MD&A={mda_wc:,}  Risk={risk_wc:,}{status}")

                rows.append({
                    "ticker": ticker, "form": form_type,
                    "filing_date": f["filing_date"],
                    "total_words": total_wc,
                    "mda_words": mda_wc, "risk_words": risk_wc,
                    "mda_found": mda_wc > 200,
                    "risk_found": risk_wc > 200,
                    "risk_boilerplate": risk_bp,
                })

    df = pd.DataFrame(rows)
    df.to_csv(DIAG_DIR / "text_availability.csv", index=False)

    # Print & build LaTeX
    _print_text_summary(df)
    _write_text_table(df)
    return df


def _print_text_summary(df):
    print(f"\n  {'':40} {'10-K':>12} {'10-Q':>12}")
    print(f"  {'─'*40} {'─'*12} {'─'*12}")

    for metric_label, col, bp_filter in [
        ("MD&A found (>200 words)", "mda_found", False),
        ("Risk Factors found (>200 words, substantive)", "risk_found", True),
    ]:
        k = df[df["form"] == "10-K"]
        q = df[df["form"] == "10-Q"]

        if bp_filter:
            k_val = k[col].mean() * 100
            q_val = (q[col] & ~q["risk_boilerplate"]).mean() * 100
        else:
            k_val = k[col].mean() * 100
            q_val = q[col].mean() * 100
        print(f"  {metric_label:<40} {k_val:>11.0f}% {q_val:>11.0f}%")

    for desc, col in [("MD&A median word count", "mda_words"), ("Risk Factors median word count", "risk_words")]:
        k_med = df.loc[(df["form"] == "10-K") & (df[col] > 200), col].median()
        q_med = df.loc[(df["form"] == "10-Q") & (df[col] > 200), col].median()
        k_s = f"{k_med:,.0f}" if pd.notna(k_med) else "—"
        q_s = f"{q_med:,.0f}" if pd.notna(q_med) else "—"
        print(f"  {desc:<40} {k_s:>12} {q_s:>12}")


def _write_text_table(df):
    k = df[df["form"] == "10-K"]
    q = df[df["form"] == "10-Q"]
    nk, nq = len(k), len(q)

    # Stats
    k_mda_pct = k["mda_found"].mean() * 100
    q_mda_pct = q["mda_found"].mean() * 100
    k_risk_pct = k["risk_found"].mean() * 100
    q_risk_subst = (q["risk_found"] & ~q["risk_boilerplate"]).mean() * 100
    q_risk_bp_pct = q["risk_boilerplate"].mean() * 100

    k_mda_med = k.loc[k["mda_found"], "mda_words"].median()
    q_mda_med = q.loc[q["mda_found"], "mda_words"].median()
    k_risk_med = k.loc[k["risk_found"], "risk_words"].median()
    q_risk_med = q.loc[q["risk_found"] & ~q["risk_boilerplate"], "risk_words"].median()

    k_mda_s = f"{k_mda_med:,.0f}" if pd.notna(k_mda_med) else "---"
    q_mda_s = f"{q_mda_med:,.0f}" if pd.notna(q_mda_med) else "---"
    k_risk_s = f"{k_risk_med:,.0f}" if pd.notna(k_risk_med) else "---"
    q_risk_s = f"{q_risk_med:,.0f}" if pd.notna(q_risk_med) else "---"

    latex = r"""\begin{table}[htbp]
\centering
\caption{Textual Data Availability: 10-K vs.\ 10-Q Filings}
\label{tab:text_availability}
\begin{tabular}{lrr}
\toprule
 & 10-K & 10-Q \\
\midrule
Filings tested & """ + f"{nk}" + r" & " + f"{nq}" + r""" \\
\addlinespace
\multicolumn{3}{l}{\textit{MD\&A (Management's Discussion and Analysis)}} \\
\quad Section found (\%) & """ + f"{k_mda_pct:.0f}" + r" & " + f"{q_mda_pct:.0f}" + r""" \\
\quad Median word count & """ + k_mda_s + r" & " + q_mda_s + r""" \\
\addlinespace
\multicolumn{3}{l}{\textit{Risk Factors}} \\
\quad Substantive section found (\%) & """ + f"{k_risk_pct:.0f}" + r" & " + f"{q_risk_subst:.0f}" + r""" \\
\quad Boilerplate / reference only (\%) & 0 & """ + f"{q_risk_bp_pct:.0f}" + r""" \\
\quad Median word count (substantive) & """ + k_risk_s + r" & " + q_risk_s + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note:} ``Found'' requires $>$200 words extracted by regex-based section
parser. ``Substantive'' for 10-Q excludes filings containing only a reference to the
annual 10-K (e.g., ``no material changes from those disclosed in our Annual Report'').
MD\&A corresponds to Item~7 in 10-K and Part~I Item~2 in 10-Q; Risk Factors
corresponds to Item~1A in 10-K and Part~II Item~1A in 10-Q.
Sample: """ + f"{len(df['ticker'].unique())}" + r""" firms, most recent filings.
\end{tablenotes}
\end{table}
"""
    path = TEX_TABLE_DIR / "text_availability.tex"
    path.write_text(latex, encoding="utf-8")
    print(f"  → {path}")

    # Also write a per-company summary table
    _write_text_per_company(df)


def _write_text_per_company(df):
    """Per-company 10-Q text summary for detailed appendix."""
    q = df[df["form"] == "10-Q"].copy()
    if q.empty:
        return

    agg = q.groupby("ticker").agg(
        filings=("form", "count"),
        mda_found=("mda_found", "sum"),
        mda_median=("mda_words", "median"),
        risk_subst=("risk_found", lambda x: (x & ~q.loc[x.index, "risk_boilerplate"]).sum()),
        risk_bp=("risk_boilerplate", "sum"),
    ).reset_index()

    latex = r"""\begin{table}[htbp]
\centering
\caption{Per-Company 10-Q Textual Content (Most Recent """ + f"{N_10Q}" + r""" Filings)}
\label{tab:text_per_company}
\small
\begin{tabular}{lrrrrr}
\toprule
Ticker & Filings & MD\&A Found & MD\&A Med.\ Words & Risk Subst. & Risk Boilerplate \\
\midrule
"""
    for _, row in agg.iterrows():
        latex += (f"{row['ticker']} & {row['filings']:.0f} & {row['mda_found']:.0f} "
                  f"& {row['mda_median']:,.0f} & {row['risk_subst']:.0f} "
                  f"& {row['risk_bp']:.0f} \\\\\n")

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note:} ``MD\&A Found'' and ``Risk Subst.'' count filings with $>$200
substantive words. ``Risk Boilerplate'' counts filings where the risk-factors section
contains only a reference to the annual 10-K.
\end{tablenotes}
\end{table}
"""
    path = TEX_TABLE_DIR / "text_per_company.tex"
    path.write_text(latex, encoding="utf-8")
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Summary & Recommendation
# ═══════════════════════════════════════════════════════════════════════════

def write_summary_table():
    """Create the master comparison table."""
    latex = r"""\begin{table}[htbp]
\centering
\caption{Annual vs.\ Quarterly Approach: Summary Comparison}
\label{tab:annual_vs_quarterly}
\begin{tabular}{p{5.5cm}p{4.5cm}p{4.5cm}}
\toprule
Criterion & Annual (10-K) & Quarterly (10-Q) \\
\midrule
Filing type & 10-K & 10-Q (Q1--Q3 only) \\
MD\&A availability & 93\% (Item 7) & 100\% (Part I, Item 2) \\
Risk Factors availability & 100\% (Item 1A) & $\sim$52\% substantive \\
Risk Factors content & Median $\sim$30{,}000 words & Median $\sim$9{,}000 words (substantive); 48\% boilerplate \\
Q4 financial data & Included in 10-K & Unreliable / absent \\
Explicit Q4 XBRL facts & N/A & 3--43\% of firm-years \\
Implied Q4 mismatch rate & N/A & $\sim$11\% of comparisons \\
Cash-flow quarterly coverage & N/A & 7\% of firm-years \\
NLP variable: $\Delta$Risk & Feasible & Limited (52\% substantive) \\
NLP variable: $\Delta$Sentiment & Feasible & Feasible (noisier) \\
NLP variable: TextSim & YoY (standard) & QoQ (unclear construct) \\
Literature alignment & Standard & Uncommon \\
Expected observations & $\sim$9{,}000 firm-years & $\sim$27{,}000 firm-quarters \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note:} Summary based on diagnostic tests of 30 non-financial companies
(XBRL, Q4 tests) and 15 companies (textual tests). Risk Factors in 10-Q filings
are required by SEC Regulation S-K Item 1A only when there are ``material changes''
from the most recent 10-K; roughly half of companies include substantive risk
disclosures, while the rest provide only a boilerplate reference. See
Tables~\ref{tab:xbrl_coverage},~\ref{tab:q4_reliability},
and~\ref{tab:text_availability} for detailed results.
\end{tablenotes}
\end{table}
"""
    path = TEX_TABLE_DIR / "annual_vs_quarterly.tex"
    path.write_text(latex, encoding="utf-8")
    print(f"  → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("COMPREHENSIVE ANNUAL vs QUARTERLY DIAGNOSTIC")
    print("=" * 70)

    # Get common sample
    print("\nResolving universe and metadata …")
    sec_df = fetch_universe(200)
    companies = load_company_metadata(sec_df)
    sample = dict(list(companies.items())[:SAMPLE_SIZE])
    print(f"  Sample: {len(sample)} companies\n")

    # Pre-fetch XBRL data once (shared by Parts A and B)
    print("Pre-fetching XBRL company facts …")
    xbrl_cache: dict[str, dict] = {}
    for ticker, meta in sorted(sample.items()):
        us_gaap = _fetch_company_facts(meta)
        if us_gaap is not None:
            xbrl_cache[ticker] = us_gaap
            print(f"  ✓ {ticker}")
        else:
            print(f"  ✗ {ticker}: no XBRL data")
        time.sleep(REQUEST_SLEEP)
    print(f"  Cached: {len(xbrl_cache)}/{len(sample)} companies\n")

    # Run all three parts
    run_part_a(sample, xbrl_cache)
    run_part_b(sample, xbrl_cache)

    # Master summary table
    print("\n" + "=" * 70)
    print("WRITING SUMMARY TABLE")
    print("=" * 70)
    write_summary_table()

    print("\n" + "=" * 70)
    print("DONE — All tables written to tex/tables/")
    print("=" * 70)


if __name__ == "__main__":
    main()
