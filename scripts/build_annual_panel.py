#!/usr/bin/env python3
"""Build the annual (10-K) panel dataset.

Pipeline:
  1. Resolve tickers → CIK via SEC API
  2. Load company metadata (fiscal-year-end month)
  3. Extract XBRL facts from 10-K filings
  4. Save raw extraction → annual_financials.csv
  5. Clean: filter transitions, resolve duplicates
  6. Impute: component computation, time-series fill
  7. Compute post-filing stock returns and volatility (365-day window)
  8. Derive financial ratios, winsorize, drop lag year
  9. Save final panel → annual_panel.csv
"""

import time
import pandas as pd

from config import (
    SAMPLE_SIZE, DATA_DIR, NUM_YEARS, MIN_VALID_YEARS,
    ANNUAL_RETURN_WINDOW, MIN_TRADING_DAYS_ANNUAL, REQUEST_SLEEP,
    MIN_FIRM_COVERAGE,
)
from sec_edgar import (fetch_universe, load_company_metadata, extract_annual_facts,
                       relabel_fiscal_years)
from returns import compute_returns_and_volatility
from panel import (
    compute_missing_components, impute_balance_sheet, add_financial_ratios,
    filter_transitions_and_duplicates, drop_earliest_year,
    drop_low_return_coverage, cap_fiscal_year,
    save_panel, ANNUAL_COLS,
    lock_firm_tags, save_tag_diagnostics, tag_overlap_diagnostics,
)
from text_parser import load_cik_map, build_text_dataset

# ── pipeline tracker ──────────────────────────────────────────────────────

_KEY_COLS = ["total_assets", "net_income", "total_liabilities",
             "operating_income", "operating_cash_flow",
             "return_next_year", "vol_next_year"]
_prev_snapshot: dict | None = None


def _snap(df: pd.DataFrame) -> dict:
    """Capture current state for comparison."""
    missing = {c: int(df[c].isna().sum()) for c in _KEY_COLS if c in df.columns}
    return {"rows": len(df), "firms": int(df["ticker"].nunique()), "missing": missing}


def _track(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Print a one-line stage summary and optional delta from previous stage."""
    global _prev_snapshot
    cur = _snap(df)
    header = f"  [{label}]  {cur['firms']} firms x {cur['rows']} rows"

    if _prev_snapshot is not None:
        dr = cur["rows"] - _prev_snapshot["rows"]
        df_firms = cur["firms"] - _prev_snapshot["firms"]
        parts = []
        if dr:
            parts.append(f"rows {dr:+d}")
        if df_firms:
            parts.append(f"firms {df_firms:+d}")
        # report changes in missing counts
        for col in _KEY_COLS:
            prev_m = _prev_snapshot["missing"].get(col)
            cur_m = cur["missing"].get(col)
            if prev_m is not None and cur_m is not None and cur_m != prev_m:
                delta = cur_m - prev_m
                parts.append(f"{col} NaN {delta:+d}")
            elif prev_m is None and cur_m is not None and cur_m > 0:
                parts.append(f"{col} NaN={cur_m}")
        if parts:
            header += "  (" + ", ".join(parts) + ")"

    print(header)
    _prev_snapshot = cur
    ATTRITION.append({"stage": label, "firms": cur["firms"], "firm_years": cur["rows"]})
    return df


# Sample-attrition log (#20) -> data/sample_attrition.csv
ATTRITION: list[dict] = []
_ATTR_PRE = DATA_DIR / "sample_attrition_universe.json"


def _save_attrition() -> None:
    import json
    rows = []
    if _ATTR_PRE.exists():
        rows = json.loads(_ATTR_PRE.read_text())
    rows += ATTRITION
    pd.DataFrame(rows).to_csv(DATA_DIR / "sample_attrition.csv", index=False)
    print(f"  Sample attrition -> sample_attrition.csv ({len(rows)} stages)")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-raw", action="store_true",
                    help="rebuild from data/annual_financials.csv (no XBRL download)")
    ap.add_argument("--skip-text", action="store_true",
                    help="skip 10-K text extraction and text features")
    args = ap.parse_args()

    raw_path = DATA_DIR / "annual_financials.csv"
    if args.from_raw:
        df_all = pd.read_csv(raw_path, parse_dates=["year_end", "filing_date"])
        print(f"Loaded raw extraction: {df_all['ticker'].nunique()} firms, "
              f"{len(df_all)} firm-years\n")
    else:
        df_all = _extract_raw()
        if df_all is None:
            return
        df_all.to_csv(raw_path, index=False)
        print(f"  Raw data -> {raw_path.name}\n")
    _process(df_all, skip_text=args.skip_text)


def _extract_raw() -> pd.DataFrame | None:
    # 1. Universe.  The SEC company-tickers file changes every day, so a
    #    rerun from it would silently change the sample.  If a frozen list
    #    exists (data/universe.csv: ticker, cik) it is used instead.
    frozen = DATA_DIR / "universe.csv"
    meta_path = DATA_DIR / "universe_meta.json"
    if frozen.exists():
        sec_df = pd.read_csv(frozen, dtype=str)
        sec_df["cik"] = sec_df["cik"].str.zfill(10)
        print(f"Universe: {len(sec_df)} companies (frozen list {frozen.name})\n")
    else:
        sec_df = fetch_universe(SAMPLE_SIZE)
        print(f"Universe: {len(sec_df)} companies (top {SAMPLE_SIZE or 'all'} by market cap)\n")
        sec_df.to_csv(frozen, index=False)
    if not meta_path.exists():
        import json, datetime as dt
        snap = dt.date.fromtimestamp(frozen.stat().st_mtime).isoformat()
        meta_path.write_text(json.dumps({
            "source": "https://www.sec.gov/files/company_tickers.json",
            "ordering": "SEC file order (approx. descending market cap); "
                        "no market-cap values in file",
            "n": len(sec_df), "snapshot_date": snap,
            "snapshot_date_basis": "file modification time of data/universe.csv",
        }, indent=2))

    # 2. Load metadata
    print("Loading company metadata …")
    companies = load_company_metadata(sec_df)
    print(f"  Loaded {len(companies)}/{len(sec_df)} companies\n")
    stages = [{"stage": "universe (SEC list, top N)", "firms": len(sec_df), "firm_years": None},
              {"stage": "operating, non-financial, valid FYE", "firms": len(companies), "firm_years": None}]

    # 3. Extract XBRL
    print("Extracting annual XBRL data …")
    frames: list[pd.DataFrame] = []
    for ticker, meta in sorted(companies.items()):
        df = extract_annual_facts(meta, NUM_YEARS)
        if df.empty:
            print(f"  {ticker}: no data")
        elif len(df) < MIN_VALID_YEARS:
            print(f"  {ticker}: {len(df)} years (below minimum {MIN_VALID_YEARS})")
        else:
            frames.append(df)
            print(f"  {ticker}: {len(df)} years")
        time.sleep(REQUEST_SLEEP)

    if not frames:
        print("No data extracted — aborting.")
        return None
    df_all = pd.concat(frames, ignore_index=True)
    print(f"\n  Total: {df_all['ticker'].nunique()} firms, {len(df_all)} firm-years\n")
    import json
    stages.append({"stage": f">= {MIN_VALID_YEARS} years of 10-K XBRL data",
                   "firms": int(df_all["ticker"].nunique()), "firm_years": len(df_all)})
    _ATTR_PRE.write_text(json.dumps(stages, indent=2))

    # Merge SIC and fiscal_year_end from metadata (SIC = today's snapshot;
    # the historical SIC at each filing is added by backfill_filing_metadata.py)
    meta_df = pd.DataFrame(companies.values())[["ticker", "sic", "fiscal_year_end"]]
    return df_all.merge(meta_df, on="ticker", how="left")


def _process(df_all: pd.DataFrame, skip_text: bool = False) -> None:
    # ── Pipeline tracking starts here ──────────────────────────────────
    print("=" * 66)
    print("  PIPELINE STAGE TRACKER")
    print("=" * 66)
    df_all = relabel_fiscal_years(df_all)
    df_all = _track(df_all, "raw extraction")

    # 4b. Per-firm tag locking (ensures within-firm XBRL tag consistency)
    print("\nLocking per-firm XBRL tags …")
    df_all = lock_firm_tags(df_all)
    save_tag_diagnostics(df_all, DATA_DIR / "tag_provenance.csv")
    tag_overlap_diagnostics(df_all, DATA_DIR / "tag_overlap.csv")
    df_all = _track(df_all, "tag locking")

    # 5. Clean transitions & duplicates
    print("\nCleaning transitions & duplicates …")
    df_all = filter_transitions_and_duplicates(df_all)
    assert not df_all.duplicated(["ticker", "fiscal_year"]).any(), "duplicate (ticker, FY)"
    df_all = _track(df_all, "transitions & dedup")

    # 6. Impute
    print("\nImputing missing data …")
    df_all = compute_missing_components(df_all)
    df_all = impute_balance_sheet(df_all, "year_end")
    df_all = _track(df_all, "imputation")

    # 7. Stock returns & volatility
    print("\nComputing stock returns & volatility (365-day window) …")
    df_all = compute_returns_and_volatility(
        df_all,
        date_col="year_end",
        return_col="return_next_year",
        vol_col="vol_next_year",
        window_days=ANNUAL_RETURN_WINDOW,
        min_trading_days=MIN_TRADING_DAYS_ANNUAL,
        lag_col="lagged_vol",
    )
    df_all = _track(df_all, "returns & volatility")

    # 8. Financial ratios (raw, unwinsorized)
    print("\nDeriving financial ratios …")
    df_all = add_financial_ratios(df_all, "year_end")
    df_all = _track(df_all, "ratios")

    # Keep the filings of the dropped first year: they are needed as the
    # t-1 anchor for the short-horizon lagged volatilities.
    earliest = df_all.groupby("ticker")["year_end"].transform("min")
    df_all.loc[df_all["year_end"] == earliest,
               ["ticker", "fiscal_year", "filing_date"]].to_csv(
        DATA_DIR / "lag_filings.csv", index=False)
    df_all = drop_earliest_year(df_all)
    df_all = _track(df_all, "drop earliest year")

    # 8b. Cap fiscal year
    print("\nFiltering panel …")
    df_all = cap_fiscal_year(df_all, max_fy=2024)
    df_all = _track(df_all, "cap FY <= 2024")

    # No firm-level return-coverage filter.  The old filter dropped firms
    # with < 90% valid *post-filing* 365-day volatility, i.e. it selected on
    # the outcome period (delisting / suspension after the filing).  Rows
    # without a valid outcome now simply drop out of each regression.
    n_cov = df_all.groupby("ticker")["vol_next_year"].apply(
        lambda s: s.notna().mean() < MIN_FIRM_COVERAGE).sum()
    print(f"  (info) {n_cov} firms would have failed the old "
          f"{MIN_FIRM_COVERAGE:.0%} coverage rule; they are kept.")

    # Years per firm in the final panel (for min-years sensitivity, #6) and
    # explicit naming of today's SIC snapshot.
    df_all["n_years_firm"] = df_all.groupby("ticker")["fiscal_year"].transform("nunique")
    df_all["sic_snapshot"] = df_all["sic"]
    _save_attrition()

    # 8c. Winsorisation is NOT applied to the stored panel: it is applied to
    #     each regression sample (regressions.load_text_panel, silence tables)
    #     so the 1st/99th percentiles are those of the estimation sample.

    print("=" * 66)

    # 9. Save
    print("\nSaving final panel …")
    save_panel(df_all, DATA_DIR / "annual_panel.csv", ANNUAL_COLS)

    if skip_text:
        print("\n[skip] text extraction / features (--skip-text)")
        return

    # 10. Extract 10-K text sections (Item 1A, Item 7)
    print("\nExtracting 10-K text sections …")
    cik_map = load_cik_map()
    build_text_dataset(df_all, cik_map)

    # 11. Compute text features and merge into the panel (optional step;
    # requires data/lm_master.csv — see scripts/lm_loader.py for source).
    lm_path = DATA_DIR / "lm_master.csv"
    if lm_path.exists():
        print("\nBuilding text features …")
        from build_text_features import main as build_text_features_main
        build_text_features_main()
    else:
        print(f"\n[skip] {lm_path.name} not found — "
              "download the Loughran-McDonald master dictionary "
              "(see scripts/lm_loader.py) and rerun "
              "scripts/build_text_features.py to produce annual_panel_text.csv.")

    print("\nDone.")


if __name__ == "__main__":
    main()
