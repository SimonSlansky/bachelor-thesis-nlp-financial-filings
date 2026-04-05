"""Verify that equivalent-tag firms show smooth, consistent time series.

For every firm that uses >1 tag from an equivalence group, this script:
  1. Prints the full time series with tag annotations
  2. Computes YoY % changes and flags tag-switch boundaries
  3. Compares YoY volatility at switch points vs. non-switch points
  4. Reports overall statistics across all switchers

Run:  python tests/test_tag_equivalence.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import numpy as np
import pandas as pd

from config import DATA_DIR, EQUIVALENT_TAG_GROUPS

raw = pd.read_csv(DATA_DIR / "annual_financials.csv")


def analyse_metric(metric: str, equiv_tags: set[str]) -> None:
    tag_col = f"_tag_{metric}"
    if tag_col not in raw.columns:
        print(f"  {tag_col} not in raw data — skipping\n")
        return

    # Identify firms that actually mix equivalent tags
    firm_tags = raw.dropna(subset=[tag_col]).groupby("ticker")[tag_col].nunique()
    multi = firm_tags[firm_tags > 1].index.tolist()
    # Keep only those whose tags are ALL within the equivalence set
    switchers = []
    for t in multi:
        used = set(raw.loc[(raw.ticker == t) & raw[tag_col].notna(), tag_col].unique())
        if used.issubset(equiv_tags):
            switchers.append(t)

    print(f"  {len(switchers)} firms use multiple equivalent tags\n")
    if not switchers:
        return

    # ── Per-firm time-series printout (top 5) ─────────────────────────
    showcase = sorted(switchers)[:5]
    for ticker in showcase:
        ts = (
            raw.loc[(raw.ticker == ticker) & raw[tag_col].notna(),
                    ["fiscal_year", metric, tag_col]]
            .sort_values("fiscal_year")
            .reset_index(drop=True)
        )
        # Short tag labels
        tag_short = {tag: f"T{i+1}" for i, tag in enumerate(sorted(equiv_tags))}
        ts["tag_label"] = ts[tag_col].map(tag_short)
        ts["yoy_pct"] = ts[metric].pct_change() * 100
        ts["switch"] = ts[tag_col] != ts[tag_col].shift(1)
        ts.loc[0, "switch"] = False

        print(f"  ── {ticker} ──")
        for _, r in ts.iterrows():
            marker = " ◄ SWITCH" if r["switch"] else ""
            print(f"    FY{int(r['fiscal_year'])}  {r[metric]:>16,.0f}  "
                  f"[{r['tag_label']}]  "
                  f"YoY {r['yoy_pct']:+7.1f}%{marker}")
        print()

    # Print tag legend
    for tag in sorted(equiv_tags):
        label = f"T{sorted(equiv_tags).index(tag)+1}"
        print(f"    {label} = {tag}")
    print()

    # ── Aggregate: compare YoY change at switch vs non-switch ─────────
    switch_changes = []
    nonswitch_changes = []

    for ticker in switchers:
        ts = (
            raw.loc[(raw.ticker == ticker) & raw[tag_col].notna(),
                    ["fiscal_year", metric, tag_col]]
            .sort_values("fiscal_year")
            .reset_index(drop=True)
        )
        for i in range(1, len(ts)):
            prev_val = ts.iloc[i - 1][metric]
            curr_val = ts.iloc[i][metric]
            if pd.isna(prev_val) or pd.isna(curr_val) or prev_val == 0:
                continue
            pct = (curr_val - prev_val) / abs(prev_val) * 100
            is_switch = ts.iloc[i][tag_col] != ts.iloc[i - 1][tag_col]
            if is_switch:
                switch_changes.append(pct)
            else:
                nonswitch_changes.append(pct)

    sw = pd.Series(switch_changes)
    nsw = pd.Series(nonswitch_changes)

    def _stats(s: pd.Series) -> str:
        return (f"median = {s.median():+.1f}%,  "
                f"IQR = [{s.quantile(.25):+.1f}%, {s.quantile(.75):+.1f}%],  "
                f"median abs = {s.abs().median():.1f}%")

    print(f"  YoY % change at TAG-SWITCH boundaries (n={len(sw)}):")
    print(f"    {_stats(sw)}")
    print(f"  YoY % change at NON-SWITCH points      (n={len(nsw)}):")
    print(f"    {_stats(nsw)}")

    # YoY volatility test
    if len(sw) >= 5 and len(nsw) >= 5:
        diff = sw.abs().median() - nsw.abs().median()
        print(f"  Median |YoY| gap (switch − non-switch): {diff:+.1f}pp")

    # ── Level test (the one that really matters): NI/TA at boundaries ─
    # YoY volatility can differ due to selection (switches coincide with
    # restructurings, spin-offs, etc.).  The key question is whether the
    # LEVEL of the metric/total_assets is systematically biased.
    switch_roa = []
    nonswitch_roa = []
    for ticker in switchers:
        ts = (
            raw.loc[(raw.ticker == ticker) & raw[tag_col].notna(),
                    ["fiscal_year", metric, "total_assets", tag_col]]
            .sort_values("fiscal_year")
            .reset_index(drop=True)
        )
        for i in range(len(ts)):
            ta = ts.iloc[i]["total_assets"]
            val = ts.iloc[i][metric]
            if pd.isna(ta) or pd.isna(val) or ta == 0:
                continue
            ratio = val / ta
            is_boundary = (i > 0 and ts.iloc[i][tag_col] != ts.iloc[i - 1][tag_col])
            if is_boundary:
                switch_roa.append(ratio)
            else:
                nonswitch_roa.append(ratio)

    sr = pd.Series(switch_roa)
    nr = pd.Series(nonswitch_roa)
    if len(sr) >= 5 and len(nr) >= 5:
        level_diff = (sr.median() - nr.median()) * 100
        verdict = "PASS ✓" if abs(level_diff) < 1.0 else "WARN — level bias >1pp"
        print(f"\n  LEVEL TEST ({metric}/total_assets):")
        print(f"    At switch boundaries:  median = {sr.median()*100:.2f}%")
        print(f"    At non-switch points:  median = {nr.median()*100:.2f}%")
        print(f"    Difference:            {level_diff:+.2f}pp  → {verdict}")
    print()


# ── Run for each equivalence group ────────────────────────────────────────
print("=" * 70)
print("  TAG EQUIVALENCE VALIDATION")
print("=" * 70)

for metric, tags in EQUIVALENT_TAG_GROUPS.items():
    print(f"\n{'─' * 70}")
    print(f"  Metric: {metric}")
    print(f"  Equivalent tags: {tags}")
    print(f"{'─' * 70}\n")
    analyse_metric(metric, tags)

print("=" * 70)
print("  DONE")
print("=" * 70)
