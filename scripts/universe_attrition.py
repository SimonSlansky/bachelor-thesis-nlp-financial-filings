"""Reconstruct the 1,000 -> 548 universe attrition (#4, #20).

data/cik_map.csv is the SEC company_tickers.json list as stored in the repo
(first committed 2026-04-06, commit c0e9369; SEC file order ~ descending
market cap, no market-cap values).  The universe is its first 1,000 ticker
rows.  For each, the Submissions API (JSON) gives entity type, SIC and FYE;
for the survivors the Company Facts API (JSON) gives the number of 10-K
fiscal years (same extraction code as the panel).  Today's metadata is used,
so the reconstruction can differ slightly from the original run; the
comparison with data/universe.csv is reported.

Writes data/universe_attrition.csv (per ticker) and prepends the stages to
data/sample_attrition.csv as data/sample_attrition_full.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, EXCLUDED_SIC_RANGE, MIN_VALID_YEARS, NUM_YEARS, REQUEST_SLEEP  # noqa: E402
from sec_edgar import _fetch_submissions, extract_annual_facts  # noqa: E402


def main():
    top = pd.read_csv(DATA_DIR / "cik_map.csv", dtype=str).head(1000)
    uni = set(pd.read_csv(DATA_DIR / "universe.csv", dtype=str)["ticker"])
    rows = []
    for i, (tk, cik) in enumerate(zip(top["ticker"], top["cik"]), 1):
        sub = _fetch_submissions(cik)
        time.sleep(REQUEST_SLEEP)
        r = {"rank": i, "ticker": tk, "cik": cik, "in_universe": tk in uni}
        if sub is None:
            r["stage_failed"] = "submissions unavailable"
            rows.append(r)
            continue
        r["entity_type"] = sub.get("entityType")
        try:
            r["sic"] = int(sub.get("sic") or 0)
        except ValueError:
            r["sic"] = 0
        fye = sub.get("fiscalYearEnd") or ""
        if r["entity_type"] != "operating":
            r["stage_failed"] = "not an operating company"
        elif r["sic"] in EXCLUDED_SIC_RANGE:
            r["stage_failed"] = "financial (SIC 6000-6999)"
        elif len(fye) < 2 or not fye[:2].isdigit():
            r["stage_failed"] = "no fiscal-year end"
        else:
            facts = extract_annual_facts({"ticker": tk, "cik": cik,
                                          "company_name": sub.get("name", tk),
                                          "fiscal_year_end": int(fye[:2])}, NUM_YEARS)
            time.sleep(REQUEST_SLEEP)
            r["n_xbrl_years"] = len(facts)
            r["stage_failed"] = (None if len(facts) >= MIN_VALID_YEARS
                                 else f"< {MIN_VALID_YEARS} years of 10-K XBRL data")
        rows.append(r)
        if i % 100 == 0:
            print(f"  [{i}/1000]", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(DATA_DIR / "universe_attrition.csv", index=False)
    order = ["not an operating company", "financial (SIC 6000-6999)", "no fiscal-year end",
             f"< {MIN_VALID_YEARS} years of 10-K XBRL data"]
    stages, left = [{"stage": "SEC ticker list, first 1,000 rows", "firms": len(d)}], len(d)
    for s in ["submissions unavailable"] + order:
        k = int((d["stage_failed"] == s).sum())
        if k:
            left -= k
            stages.append({"stage": f"minus {s}", "firms": left})
    st = pd.DataFrame(stages)
    st.to_csv(DATA_DIR / "universe_stages.csv", index=False)
    ok = d["stage_failed"].isna()
    print(st.to_string(index=False))
    print(f"reconstructed survivors {int(ok.sum())}; of these in universe.csv "
          f"{int((ok & d['in_universe']).sum())}; universe firms not reconstructed "
          f"{int((~ok & d['in_universe']).sum())}")


if __name__ == "__main__":
    main()
